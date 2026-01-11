# scripts/run_experiment_scripts/run_C7/run_train_val_experiments_C7_direct_heavy.py

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
RESULTS_DIR = ROOT / "results" / "C7_Direct_Heavy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = RESULTS_DIR / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = RESULTS_DIR / "train_val_metrics.csv" 
TUNING_CSV = RESULTS_DIR / "tuning_log.csv"

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))

# Import Helpers
from model_definitions import get_optimized_model, predict_with_torch
from features_scripts.feature_tiers_C7 import TIERS, HISTORY_VARIANTS

# --- 1. MANIFEST SAVER ---
def save_run_manifest(targets, tiers, algos):
    manifest = {
        "target_family": "C7",
        "method": "Direct",
        "mode": "Heavy",
        "targets": targets,
        "tiers": tiers,
        "history_variants": ["WithLag1"],
        "algorithms": algos,
        "results_dir": str(RESULTS_DIR),
        "timestamp_utc": str(datetime.utcnow()) 
    }
    with open(RESULTS_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

def prepare_xy_subsampled(df, features, target_col, neg_fraction=0.05):
    df_clean = df.dropna(subset=features + [target_col])
    pos_idx = df_clean[df_clean[target_col] > 0].index
    neg_idx = df_clean[df_clean[target_col] == 0].index
    if len(neg_idx) > 0:
        neg_sample = np.random.choice(neg_idx, int(len(neg_idx) * neg_fraction), replace=False)
        final_idx = np.concatenate([pos_idx, neg_sample])
    else:
        final_idx = pos_idx
    np.random.shuffle(final_idx)
    df_sub = df_clean.loc[final_idx]
    return df_sub[features].values, df_sub[target_col].values

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
    print("=== TRAIN/VAL EXPERIMENTS (Direct | Heavy | Standardized) ===")

    TARGETS = ["y_7d_sum"]
    TIER_LIST = ["Tier1_Island", "Tier2_Weather", "Tier3_Full"]
    HISTORY_LIST = ["WithLag1"]
    ALGOS = ["GLM", "HGB", "RF", "MLP", "GRU"]

    # SAVE MANIFEST
    save_run_manifest(TARGETS, TIER_LIST, ALGOS)

    print("Loading datasets (Float32 forced)...")
    all_feats = list(set([c for t in TIERS.values() for c in t] + [c for h in HISTORY_VARIANTS.values() for c in h] + TARGETS + ["cell_id", "date", "time_bin"]))
    
    train_df = pd.read_parquet(DATA_PROCESSED / "train_dataset.parquet", columns=all_feats)
    val_df = pd.read_parquet(DATA_PROCESSED / "validation_dataset.parquet", columns=all_feats)

    for df in [train_df, val_df]:
        fcols = df.select_dtypes(include=[np.float64]).columns
        df[fcols] = df[fcols].astype(np.float32)
        df.sort_values(["cell_id", "date", "time_bin"], inplace=True)

    run_count = 0
    total_runs = len(TARGETS) * len(TIER_LIST) * len(HISTORY_LIST) * len(ALGOS)

    for target in TARGETS:
        print(f"\n\n>>> PROCESSING TARGET: {target} <<<")
        
        for tier in TIER_LIST:
            for hist_var in HISTORY_LIST:
                feature_cols = TIERS[tier] + HISTORY_VARIANTS[hist_var]
                input_dim = len(feature_cols)
                
                try:
                    X_train, y_train = prepare_xy_subsampled(train_df, feature_cols, target, neg_fraction=0.05)
                    X_val = val_df[feature_cols].fillna(0).values
                    y_val = val_df[target].fillna(0).values
                except Exception as e:
                    print(f"Data Prep Failed: {e}"); continue

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
                        
                        # --- TRAIN ---
                        # HEAVY MODE IS ON
                        model, tuning_info = get_optimized_model(
                            algo_name=algo, approach="Direct", input_dim=input_dim,
                            X_train=X_train_in, y_train=y_train, light_mode=False, return_cv_results=True
                        )

                        # --- PREDICT ---
                        if algo in ["MLP", "GRU"]:
                            is_gru = (algo == "GRU")
                            y_pred = predict_with_torch(model, X_val_in, is_gru=is_gru)
                            
                            ckpt = {
                                "state_dict": model.state_dict(),
                                "config": getattr(model, "_best_cfg", None),
                                "algo": algo, "tier": tier, "target": target
                            }
                            torch.save(ckpt, MODELS_DIR / f"regressor_direct_{model_id}.pt")

                        else:
                            model.fit(X_train_in, y_train)
                            y_pred = model.predict(X_val_in)
                            
                            with open(MODELS_DIR / f"regressor_direct_{model_id}.pkl", "wb") as f:
                                pickle.dump(model, f)

                        # Save Tuning Log
                        if tuning_info:
                            try:
                                tdf = pd.DataFrame(tuning_info) if isinstance(tuning_info, list) else pd.DataFrame([tuning_info])
                                tdf["algo"] = algo; tdf["tier"] = tier
                                tdf.to_csv(TUNING_CSV, mode='a', header=not TUNING_CSV.exists(), index=False)
                            except: pass

                        # Metrics
                        metrics = compute_metrics(y_val, y_pred)
                        dt = time.time() - t0
                        print(f"     RMSE={metrics['rmse']:.4f}")

                        # --- STANDARD RESULT DICT ---
                        res = {
                            "target_family": "C7",   
                            "mode": "Heavy",         
                            "split": "train_val",    
                            "method": "Direct", 
                            "target": target, 
                            "algo": algo, 
                            "tier": tier,
                            "history": hist_var, 
                            **metrics, 
                            "time_sec": float(dt),
                            "requires_scaling": needs_scaling
                        }
                        
                        # Save to CSV
                        save_to_csv(res, METRICS_CSV)
                        
                        # Save to JSON (Standard Name)
                        rpath = RESULTS_DIR / "train_val_results.json"
                        existing = json.load(open(rpath)) if rpath.exists() else []
                        existing.append(res)
                        with open(rpath, 'w') as f: json.dump(existing, f, indent=2)

                        # Cleanup
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