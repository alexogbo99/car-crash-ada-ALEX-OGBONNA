# scripts/run_experiment_scripts/run_C7/run_gatekeeper_selection_C7_heavy.py

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import time
import shutil
import gc
import joblib 
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# PATHS
ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "C7_Hurdle_Heavy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "gatekeeper_metrics_heavy.csv"

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))

from model_definitions import get_optimized_model
from features_scripts.feature_tiers_C7 import TIERS, HISTORY_VARIANTS

def compute_lift_1pct(y_true, y_prob):
    n = len(y_true)
    if n == 0: return 0.0
    k = max(1, int(0.01 * n))
    df = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p', ascending=False)
    base_rate = y_true.mean()
    top_rate = df.iloc[:k]['y'].mean()
    return top_rate / base_rate if base_rate > 0 else 0.0

def main():
    print("=== GATEKEEPER SELECTION (All Tiers | Safe Memory | Pipeline Fixed) ===")
    
    TARGETS = ["y_7d_sum"]
    TIER_LIST = ["Tier1_Island", "Tier2_Weather", "Tier3_Full"]
    hist_name = "WithLag1"
    ALGOS = ["HGB", "RF", "LogReg"]

    for target in TARGETS:
        print(f"\n\n################################################")
        print(f"   TARGET: {target}")
        print(f"################################################")

        for tier_name in TIER_LIST:
            print(f"\n>>> PROCESSING {tier_name} <<<")
            
            feature_cols = TIERS[tier_name] + HISTORY_VARIANTS[hist_name]
            req_cols = list(set(feature_cols + [target]))
            
            # 1. Load & Subsample
            print("Loading TRAIN dataset (Float32)...")
            train_df = pd.read_parquet(DATA_PROCESSED / "train_dataset.parquet", columns=req_cols)
            
            fcols = train_df.select_dtypes(include=[np.float64]).columns
            train_df[fcols] = train_df[fcols].astype(np.float32)
            
            pos_mask = train_df[target] > 0
            neg_mask = train_df[target] == 0
            
            pos_idx = train_df[pos_mask].index
            n_pos = len(pos_idx)
            if n_pos < 10:
                print(f"⚠️ SKIPPING {tier_name} | {target}: Too few positives ({n_pos}).")
                del train_df; gc.collect()
                continue

            # Downsample Negatives to 10% for training speed/balance
            if neg_mask.sum() > 0:
                neg_idx = train_df[neg_mask].sample(frac=0.10, random_state=42).index
                train_idx = pos_idx.union(neg_idx).tolist()
            else:
                train_idx = pos_idx.tolist()
            
            # Reduce DataFrame
            train_df = train_df.loc[train_idx].sample(frac=1.0, random_state=42)
            print(f"   Train Size Reduced to: {len(train_df)} rows")
            
            # Create Arrays
            X_train = train_df[feature_cols].fillna(0).values
            y_train_bin = (train_df[target] > 0).astype(int).values
            
            del train_df, pos_idx, train_idx, pos_mask, neg_mask
            gc.collect()

            print("Loading VAL dataset (Float32)...")
            val_df = pd.read_parquet(DATA_PROCESSED / "validation_dataset.parquet", columns=req_cols)
            fcols_val = val_df.select_dtypes(include=[np.float64]).columns
            val_df[fcols_val] = val_df[fcols_val].astype(np.float32)
            
            X_val = val_df[feature_cols].fillna(0).values
            y_val_bin = (val_df[target] > 0).astype(int).values
            del val_df; gc.collect()

            # 2. Train Candidates
            for algo in ALGOS:
                print(f"   Training {algo} on {tier_name}...")
                t0 = time.time()
                
                try:
                    # For LogReg, we fit on RAW data here, but wrapped in a Pipeline 
                                        
                    if algo == "LogReg":
                        # Standard scaler for optimization input
                        scaler = StandardScaler()
                        X_train_in = scaler.fit_transform(X_train).astype(np.float32)
                        X_val_in = scaler.transform(X_val).astype(np.float32)
                    else:
                        X_train_in, X_val_in = X_train, X_val

                    # Get optimized base model
                    base_model = get_optimized_model(
                        algo_name=algo, approach="Classifier",
                        input_dim=X_train_in.shape[1], X_train=X_train_in, y_train=y_train_bin, light_mode=False 
                    )
                    
                    # Explicit Fit
                    if algo == "RF" and len(X_train_in) > 500_000:
                        idx = np.random.choice(len(X_train_in), 500_000, replace=False)
                        base_model.fit(X_train_in[idx], y_train_bin[idx])
                    else:
                        base_model.fit(X_train_in, y_train_bin)
                    
                    # Predict
                    y_prob = base_model.predict_proba(X_val_in)[:, 1]
                    auc = roc_auc_score(y_val_bin, y_prob)
                    lift = compute_lift_1pct(y_val_bin, y_prob)
                    dt = time.time() - t0
                    
                    print(f"     [VAL] AUC={auc:.4f} | Lift={lift:.2f}")

                    # Save Model - WRAP LOGREG IN PIPELINE
                    model_to_save = base_model
                    if algo == "LogReg":
                        # Re-create pipeline so loading it later handles scaling automatically
                        # We must recreate the pipeline with the fitted scaler
                        pipe = make_pipeline(scaler, base_model)
                        model_to_save = pipe
                        # Clean up temp arrays
                        del X_train_in, X_val_in

                    model_filename = f"gatekeeper_{algo}_{tier_name}_{target}.pkl"
                    with open(RESULTS_DIR / model_filename, "wb") as f:
                        pickle.dump(model_to_save, f)
                    
                    # Save Metrics
                    row = {
                        "algo": algo, "tier": tier_name, "target": target,
                        "auc": auc, "lift_1pct": lift, "time_sec": dt, 
                        "model_file": model_filename
                    }
                    pd.DataFrame([row]).to_csv(CSV_PATH, mode='a', header=not CSV_PATH.exists(), index=False)
                
                except Exception as e:
                    print(f"Error training {algo}: {e}")
                    # traceback.print_exc()

            # 3. Pick Champion
            print(f"   Picking Champion for {tier_name} ({target})...")
            if CSV_PATH.exists():
                full_df = pd.read_csv(CSV_PATH)
                tier_df = full_df[(full_df["tier"] == tier_name) & (full_df["target"] == target)]
                
                if not tier_df.empty:
                    best_row = tier_df.sort_values(["auc", "lift_1pct"], ascending=[False, False]).iloc[0]
                    champ_file = best_row["model_file"]
                    
                    src = RESULTS_DIR / champ_file
                    dst = RESULTS_DIR / f"champion_gatekeeper_{tier_name}_{target}.pkl"
                    
                    if src.exists():
                        shutil.copy(src, dst)
                        print(f"Champion saved: {dst.name}")
            
            del X_train, y_train_bin, X_val, y_val_bin
            gc.collect()

if __name__ == "__main__":
    main()