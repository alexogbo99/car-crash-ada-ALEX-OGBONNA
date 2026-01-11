import sys
from pathlib import Path
import json
import time
import gc
import pickle
import traceback
import numpy as np
import pandas as pd
import torch
import joblib 
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.preprocessing import StandardScaler

# PATHS
ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "C1_Direct_Heavy" 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = RESULTS_DIR / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = RESULTS_DIR / "train_val_metrics.csv"
TUNING_CSV = RESULTS_DIR / "tuning_log.csv"

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))

# Import Helpers
from model_definitions import get_optimized_model, predict_with_torch
from features_scripts.feature_tiers_C1 import TIERS, HISTORY_VARIANTS 

def save_to_csv(result_dict, path):
    df = pd.DataFrame([result_dict])
    if not path.exists():
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)

def compute_metrics(y_true, y_pred):
    y_pred_safe = np.maximum(y_pred, 1e-6)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "poisson_dev": float(mean_poisson_deviance(y_true, y_pred_safe)),
        "mae": float(np.mean(np.abs(y_true - y_pred)))
    }

def main():
    print("=== STARTING EXPERIMENT: C1 (Crash Counts) | DIRECT | HEAVY ===")
    
    # 1. Load Data
    print("Loading datasets...")
    train_df = pd.read_parquet(DATA_PROCESSED / "train_dataset.parquet")
    val_df   = pd.read_parquet(DATA_PROCESSED / "val_dataset.parquet")
    
    # 2. Config - PRESERVING EXACT STRUCTURE
    TARGETS = ["y"] 
    TIER_LIST = ["Tier1_Island", "Tier2_Weather", "Tier3_Full"]
    HISTORY_LIST = ["WithLag1"]
    ALGOS = ["GLM", "HGB", "RF", "MLP", "GRU"]
    
    # 3. Loops
    for target in TARGETS:
        print(f"\n>>> TARGET: {target}")
        
        for tier_name in TIER_LIST:
            features = TIERS[tier_name]
            print(f"  >> TIER: {tier_name} ({len(features)} feats)")
            
            for hist_name in HISTORY_LIST:
                hist_cols = HISTORY_VARIANTS[hist_name]
                full_feats = features + hist_cols
                
                # Full Data
                X_train = train_df[full_feats].fillna(0).values
                y_train = train_df[target].fillna(0).values
                X_val   = val_df[full_feats].fillna(0).values
                y_val   = val_df[target].fillna(0).values
                
                if np.isnan(X_train).any(): 
                    print("NaN in Train X, filling 0")
                    X_train = np.nan_to_num(X_train)
                    
                for algo in ALGOS:
                    print(f"    > ALGO: {algo} | HIST: {hist_name} ... ", end="")
                    t0 = time.time()
                    
                    try:
                        # Scaling
                        needs_scaling = algo in ["MLP", "GRU", "GLM"]
                        if needs_scaling:
                            scaler = StandardScaler()
                            X_train_in = scaler.fit_transform(X_train).astype(np.float32)
                            X_val_in   = scaler.transform(X_val).astype(np.float32)
                            
                            scaler_path = MODELS_DIR / f"scaler_{algo}_{tier_name}_{target}.pkl"
                            with open(scaler_path, "wb") as f:
                                pickle.dump(scaler, f)
                        else:
                            X_train_in = X_train
                            X_val_in   = X_val
                        
                        # Train
                        model = get_optimized_model(
                            algo, 
                            input_dim=X_train_in.shape[1], 
                            tuning_csv=TUNING_CSV, 
                            is_search=False 
                        )
                        
                        if algo in ["MLP", "GRU"]:
                            model.fit(X_train_in, y_train, X_val_in, y_val)
                            
                            torch.save({
                                "state_dict": model.state_dict(),
                                "input_dim": X_train_in.shape[1]
                            }, MODELS_DIR / f"regressor_direct_{algo}_{tier_name}_{target}.pt")
                            
                            y_pred = predict_with_torch(model, X_val_in, is_gru=(algo=="GRU"))
                        else:
                            model.fit(X_train_in, y_train)
                            joblib.dump(model, MODELS_DIR / f"regressor_direct_{algo}_{tier_name}_{target}.pkl")
                            y_pred = model.predict(X_val_in)
                        
                        metrics = compute_metrics(y_val, y_pred)
                        dt = time.time() - t0
                        print(f"RMSE={metrics['rmse']:.4f} ({dt:.1f}s)")
                        
                        res = {
                            "split": "train_val",
                            "method": "Direct",
                            "target": target, 
                            "algo": algo, 
                            "tier": tier_name,
                            "history": hist_name, 
                            **metrics, 
                            "time_sec": float(dt),
                            "requires_scaling": needs_scaling
                        }
                        save_to_csv(res, METRICS_CSV)
                        
                        # Updated JSON filename for C1
                        rpath = RESULTS_DIR / "train_val_results.json"
                        existing = json.load(open(rpath)) if rpath.exists() else []
                        existing.append(res)
                        with open(rpath, 'w') as f: json.dump(existing, f, indent=2)

                        del model, y_pred
                        if needs_scaling: del X_train_in, X_val_in

                    except Exception as e:
                        print(f"Crash on {algo}: {e}")
                        traceback.print_exc()
                        gc.collect()

                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                del X_train, y_train, X_val, y_val; gc.collect()

    print(f"\nAll experiments done. Final metrics saved to {METRICS_CSV}")

if __name__ == "__main__":
    main()