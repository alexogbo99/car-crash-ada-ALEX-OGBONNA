# scripts/run_experiment_scripts/run_C7/run_train_val_experiments_C7_hurdle_light.py

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
RESULTS_DIR = ROOT / "results" / "C7_Hurdle_Light"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = RESULTS_DIR / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = RESULTS_DIR / "train_val_metrics_hurdle.csv"

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))

from model_definitions import get_optimized_model, HurdleModel, predict_with_torch
from features_scripts.feature_tiers_C7 import TIERS, HISTORY_VARIANTS

def compute_metrics(y_true, y_pred):
    y_pred_nn = np.maximum(y_pred, 1e-6)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "poisson_dev": float(mean_poisson_deviance(y_true, y_pred_nn))
    }

def save_to_csv(data_dict, csv_path):
    df = pd.DataFrame([data_dict])
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode='a', header=write_header, index=False)

def main():
    print("=== TRAIN/VAL EXPERIMENTS (Hurdle | No Wrapper | Light Mode) ===")
    
    TARGETS = ["y_7d_sum"]
    TIER_LIST = ["Tier1_Island", "Tier2_Weather", "Tier3_Full"]
    HISTORY_LIST = ["WithLag1"]
    ALGOS = ["GLM", "HGB", "RF", "MLP", "GRU"]
    
    run_count = 0
    total_runs = len(TARGETS) * len(TIER_LIST) * len(HISTORY_LIST) * len(ALGOS)

    for target in TARGETS:
        print(f"\n>>> PROCESSING TARGET: {target} <<<")

        for tier in TIER_LIST:
            # 1. LOAD CHAMPION GATEKEEPER
            champ_path = RESULTS_DIR / f"champion_gatekeeper_{tier}_{target}.pkl"
            if champ_path.exists():
                with open(champ_path, "rb") as f:
                    tier_gatekeeper = pickle.load(f)
            else:
                print(f"Missing Gatekeeper for {tier}. Skipping."); continue

            for hist_var in HISTORY_LIST:
                feature_cols = TIERS[tier] + HISTORY_VARIANTS[hist_var]
                req_cols = list(set(feature_cols + [target, "cell_id", "date", "time_bin"]))
                input_dim = len(feature_cols)

                # 2. LOAD DATA ONCE PER CONFIG
                print(f"   Loading Data for {tier} + {hist_var}...")
                try:
                    train_df = pd.read_parquet(DATA_PROCESSED / "train_dataset.parquet", columns=req_cols)
                    val_df = pd.read_parquet(DATA_PROCESSED / "validation_dataset.parquet", columns=req_cols)
                    
                    for df in [train_df, val_df]:
                        fcols = df.select_dtypes(include=[np.float64]).columns
                        df[fcols] = df[fcols].astype(np.float32)
                        df.sort_values(["cell_id", "date", "time_bin"], inplace=True)

                    X_train = train_df[feature_cols].fillna(0).values
                    y_train = train_df[target].fillna(0).values
                    X_val = val_df[feature_cols].fillna(0).values
                    y_val = val_df[target].fillna(0).values
                    del train_df, val_df; gc.collect()
                except Exception as e:
                    print(f"Data Load Error: {e}"); continue

                # 3. TRAIN REGRESSORS
                for algo in ALGOS:
                    run_count += 1
                    model_id = f"{algo}_{tier}_{target}"
                    print(f"\n[{run_count}/{total_runs}] Running {algo} | {tier} | {target} ...")
                    t0 = time.time()
                    
                    try:
                        needs_scaling = algo in ["GLM", "MLP", "GRU"]
                        if needs_scaling:
                            scaler = StandardScaler()
                            X_train_in = scaler.fit_transform(X_train).astype(np.float32)
                            X_val_in = scaler.transform(X_val).astype(np.float32)
                        else:
                            X_train_in, X_val_in = X_train, X_val
                        
                        # --- BRANCH 1: NEURAL NETWORKS ---
                        if algo in ["MLP", "GRU"]:
                            regressor = get_optimized_model(
                                algo_name=algo, approach="Regressor", input_dim=input_dim,
                                X_train=X_train_in, y_train=y_train, 
                                light_mode=True, return_cv_results=False 
                            )
                            
                            # Manual Hurdle Prediction
                            # 1. Gate Probability (Raw X, pipeline handles scaling if needed)
                            prob_gate = tier_gatekeeper.predict_proba(X_val)[:, 1]
                            
                            # 2. Regressor Prediction (Scaled X if needed)
                            pred_reg = predict_with_torch(regressor, X_val_in, is_gru=(algo=="GRU"))
                            
                            y_pred = prob_gate * pred_reg
                            
                            # Save Torch Regressor
                            torch.save({
                                "state_dict": regressor.state_dict(),
                                "algo": algo, "tier": tier, "input_dim": input_dim
                            }, MODELS_DIR / f"regressor_hurdle_{model_id}.pt")

                        # --- BRANCH 2: STANDARD SKLEARN ---
                        else:
                            regressor = get_optimized_model(
                                algo_name=algo, approach="Regressor", input_dim=input_dim,
                                X_train=X_train_in, y_train=y_train, 
                                light_mode=True, return_cv_results=False
                            )
                            
                            if not hasattr(regressor, "predict"):
                                regressor.fit(X_train_in, y_train)

                            # Manual Hurdle Prediction
                            prob_gate = tier_gatekeeper.predict_proba(X_val)[:, 1]
                            pred_reg = regressor.predict(X_val_in)
                            y_pred = prob_gate * pred_reg
                            
                            # Save Pickle Regressor
                            with open(MODELS_DIR / f"regressor_hurdle_{model_id}.pkl", "wb") as f:
                                pickle.dump(regressor, f)

                        # Evaluate
                        metrics = compute_metrics(y_val, y_pred)
                        dt = time.time() - t0
                        print(f"     RMSE={metrics['rmse']:.4f}")
                        
                        res = {
                            "method": "Hurdle", "target": target, "algo": algo, "tier": tier,
                            "history": hist_var, **metrics, "time_sec": dt, 
                            "requires_scaling": needs_scaling
                        }
                        save_to_csv(res, METRICS_CSV)
                        
                        rpath = RESULTS_DIR / "train_val_results_C7_Hurdle_light.json"
                        existing = json.load(open(rpath)) if rpath.exists() else []
                        existing.append(res)
                        with open(rpath, 'w') as f: json.dump(existing, f, indent=2)
                        
                        if needs_scaling: del X_train_in, X_val_in
                        del regressor

                    except Exception as e:
                        print(f"Error {algo}: {e}")
                        traceback.print_exc()

                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                del X_train, y_train, X_val, y_val; gc.collect()

if __name__ == "__main__":
    main()