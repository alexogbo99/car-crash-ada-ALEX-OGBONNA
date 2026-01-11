import sys; from pathlib import Path; import json; import pickle; import gc; import traceback; import numpy as np; import pandas as pd; import torch; import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance

ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "C1_Direct_Light" 
MODELS_DIR = RESULTS_DIR / "saved_models"
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))

from model_definitions import predict_with_torch, RobustMLP, RobustGRU
from features_scripts.feature_tiers_C1 import TIERS, HISTORY_VARIANTS

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================
def load_or_fit_train_scaler(
    train_parquet: Path,
    feature_cols: list,
    scaler_path: Path,
    *,
    batch_rows: int = 200_000,
    max_batches: int | None = None,
):
    if scaler_path.exists():
        return joblib.load(scaler_path)

    scaler = StandardScaler()
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(train_parquet))
        for i, batch in enumerate(pf.iter_batches(columns=feature_cols, batch_size=batch_rows)):
            Xb = batch.to_pandas().fillna(0).to_numpy(dtype=np.float32, copy=False)
            scaler.partial_fit(Xb)
            if max_batches is not None and (i + 1) >= max_batches: break
    except Exception as e:
        print(f"Streaming scaler fit failed ({e}). Falling back to full read.")
        df = pd.read_parquet(train_parquet, columns=feature_cols)
        X = df.fillna(0).to_numpy(dtype=np.float32, copy=False)
        scaler.fit(X)

    joblib.dump(scaler, scaler_path)
    return scaler

def compute_metrics(y_true, y_pred):
    y_pred_nn = np.maximum(y_pred, 1e-6)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "poisson_dev": float(mean_poisson_deviance(y_true, y_pred_nn))
    }

def infer_mlp_structure(state_dict):
    layers = []
    idx = 0
    while True:
        key = f"net.{idx}.weight"
        if key in state_dict:
            out_dim = state_dict[key].shape[0]
            if out_dim == 1: break 
            layers.append(out_dim)
            idx += 4 
        else:
            break
    return layers

def infer_gru_structure(state_dict):
    if 'gru.weight_ih_l0' in state_dict:
        hidden_dim = state_dict['gru.weight_ih_l0'].shape[0] // 3
    else:
        hidden_dim = 16 
    num_layers = 0
    while True:
        if f'gru.weight_ih_l{num_layers}' in state_dict:
            num_layers += 1
        else:
            break
    return hidden_dim, num_layers

def print_model_details(algo, model):
    print(f"Model Specs for {algo}:")
    try:
        if algo in ["MLP", "GRU"]:
            if algo == "MLP":
                print(f"      Architecture: {model}")
            else:
                print(f"      Architecture: GRU(hidden={model.gru.hidden_size}, layers={model.gru.num_layers})")
        else:
            params = model.get_params()
            if algo == "RF":
                print(f"      Trees: {params.get('n_estimators')}, MaxDepth: {params.get('max_depth')}")
            elif algo == "HGB":
                print(f"      LR: {params.get('learning_rate')}, MaxLeaf: {params.get('max_leaf_nodes')}")
            elif algo == "GLM":
                print(f"      Alpha: {params.get('alpha')}")
    except:
        print("      (Could not extract details)")

# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================
def main():
    print("=== FINAL TEST EVALUATION (C1 | Direct | Light | DUAL METRIC) ===")
    with joblib.parallel_backend('threading', n_jobs=-1):
        
        results_file = RESULTS_DIR / "train_val_results_C1_Direct.json"
        if not results_file.exists(): 
            print(f"No results found at: {results_file}"); return
            
        df_res = pd.DataFrame(json.load(open(results_file)))
        
        TARGETS = ["y"]
        TIER_LIST = ["Tier1_Island", "Tier2_Weather", "Tier3_Full"]

        # === THE DUAL METRIC LOOP ===
        METRIC_CONFIGS = [
            ("rmse", "final_test_metrics_rmse.json"),
            ("poisson_dev", "final_test_metrics_dev_poisson.json")
        ]

        for sort_metric, out_filename in METRIC_CONFIGS:
            print(f"\n\nSTARTING PASS: Selecting Champions by {sort_metric.upper()}...")
            
            final_report = {}

            for target in TARGETS:
                final_report[target] = {}
                target_res = df_res[df_res["target"] == target]
                if target_res.empty: continue

                for tier in TIER_LIST:
                    try:
                        # 1. Identify Champion
                        tier_res = target_res[target_res["tier"] == tier]
                        if tier_res.empty: continue
                        
                        # --- KEY CHANGE: Sort by current metric ---
                        best_row = tier_res.sort_values(sort_metric, ascending=True).iloc[0]
                        # ------------------------------------------

                        best_algo, best_hist = best_row["algo"], best_row.get("history", "WithLag1")
                        print(f"\nChampion ({sort_metric}): {tier} -> {best_algo} ({best_hist}) [Val {sort_metric}: {best_row[sort_metric]:.4f}]")

                        # 2. Load Test Data
                        features = TIERS[tier] + HISTORY_VARIANTS[best_hist]
                        req_cols = list(set(features + [target, "cell_id", "date", "time_bin"]))
                        test_df = pd.read_parquet(DATA_PROCESSED / "test_dataset.parquet", columns=req_cols)
                        
                        fcols = test_df.select_dtypes(include=[np.float64]).columns
                        test_df[fcols] = test_df[fcols].astype(np.float32)
                        test_df.sort_values(["cell_id", "date", "time_bin"], inplace=True)
                        
                        X_test = test_df[features].fillna(0).values
                        y_test = test_df[target].fillna(0).values
                        del test_df; gc.collect()

                        # 3. Load & Apply Scaler
                        model_name = f"regressor_direct_{best_algo}_{tier}_{target}"

                        if best_algo in ["MLP", "GRU", "GLM"]:
                            scaler_path = MODELS_DIR / f"{model_name}_scaler.joblib"
                            scaler = load_or_fit_train_scaler(
                                DATA_PROCESSED / "train_dataset.parquet",
                                features,
                                scaler_path,
                            )
                            X_test = scaler.transform(X_test).astype(np.float32)

                        # 4. Load Model
                        if best_algo in ["MLP", "GRU"]:
                            pt_path = MODELS_DIR / f"{model_name}.pt"
                            if not pt_path.exists(): print(f"{pt_path} missing"); continue
                            
                            ckpt = torch.load(pt_path, map_location="cpu")
                            state_dict = ckpt["state_dict"]
                            
                            if best_algo == "MLP":
                                inferred_layers = infer_mlp_structure(state_dict)
                                model = RobustMLP(len(features), hidden_dims=inferred_layers)
                            else:
                                h_dim, n_lay = infer_gru_structure(state_dict)
                                model = RobustGRU(len(features), hidden_dim=h_dim, num_layers=n_lay)
                                
                            model.load_state_dict(state_dict)
                            print_model_details(best_algo, model)
                            
                            y_pred = predict_with_torch(model, X_test, is_gru=(best_algo=="GRU"))
                            
                        else:
                            pkl_path = MODELS_DIR / f"{model_name}.pkl"
                            if not pkl_path.exists(): print(f"{pkl_path} missing"); continue
                            with open(pkl_path, "rb") as f: model = pickle.load(f)
                            
                            print_model_details(best_algo, model)
                            y_pred = model.predict(X_test)

                        # 5. Metrics
                        metrics = compute_metrics(y_test, y_pred)
                        print(f"     Test Results: RMSE={metrics['rmse']:.4f} | Dev={metrics['poisson_dev']:.4f}")
                        
                        final_report[target][tier] = {
                            **metrics,
                            "algo": best_algo,
                            "history": best_hist,
                            "model_name": model_name,
                            "selection_metric": sort_metric
                        }

                        del model, X_test, y_test; gc.collect()

                    except Exception as e:
                        print(f"Error {tier}: {e}"); traceback.print_exc()

            with open(RESULTS_DIR / out_filename, "w") as f: json.dump(final_report, f, indent=2)
            print(f"Saved {out_filename}")

if __name__ == "__main__": main()