import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import re

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "results" / "plots" / "02_time_series_comparator"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_definitions import RobustMLP, RobustGRU, predict_with_torch
from features_scripts.feature_tiers_C7 import TIERS as TIERS_C7, HISTORY_VARIANTS as HIST_C7
from features_scripts.feature_tiers_C1 import TIERS as TIERS_C1, HISTORY_VARIANTS as HIST_C1

# ==============================================================================
# HELPERS: Torch shape inference (ROBUST & VERBOSE)
# ==============================================================================
def infer_mlp_structure(state_dict):
    """
    Robustly infers MLP hidden dimensions from state_dict keys.
    """
    print("      [DEBUG] Scanning state_dict for MLP structure...")
    
    # 1. Filter for Linear Layer weights (net.X.weight)
    keys = [k for k in state_dict.keys() if k.startswith("net.") and k.endswith(".weight")]
    
    # 2. Sort by layer index
    def get_idx(k):
        m = re.search(r"net\.(\d+)\.weight", k)
        return int(m.group(1)) if m else 999
    keys.sort(key=get_idx)
    
    dims = []
    for k in keys:
        w = state_dict[k]
        # 3. Only count 2D tensors (Linear Layers)
        if w.ndim == 2:
            out_dim, in_dim = w.shape
            dims.append(out_dim)
            
    # dims contains [Hidden1, Hidden2, ..., Output]
    if len(dims) > 1:
        inferred = dims[:-1]
        print(f"      [DEBUG] Inferred hidden_dims: {inferred}")
        return inferred
        
    print("      [DEBUG] Failed to infer. Fallback to [64, 32, 16]")
    return [64, 32, 16]

def infer_gru_structure(state_dict):
    if 'gru.weight_ih_l0' in state_dict:
        w_shape = state_dict['gru.weight_ih_l0'].shape
        hidden_dim = w_shape[0] // 3
        num_layers = 0
        while f'gru.weight_ih_l{num_layers}' in state_dict:
            num_layers += 1
        return hidden_dim, num_layers
    return 16, 2

# ==============================================================================
# MODELS CONFIG (UPDATED WITH CHAMPIONS)
# ==============================================================================
MODEL_CONFIGS = [
    # --- C1 CHAMPIONS ---
    {
        "family": "C1", "target": "y",
        "method": "Direct", "mode": "Heavy", "tier": "Tier3_Full", "algo": "GRU",
        "label": "Direct (Tier3 - GRU)"
    },
    {
        "family": "C1", "target": "y",
        "method": "Hurdle", "mode": "Heavy", "tier": "Tier3_Full", "algo": "MLP",
        "label": "Hurdle (Tier3 - MLP)"
    },
    # --- C7 CHAMPIONS ---
    {
        "family": "C7", "target": "y_7d_sum",
        "method": "Direct", "mode": "Heavy", "tier": "Tier3_Full", "algo": "RF",
        "label": "Direct (Tier3 - RF)"
    },
    {
        "family": "C7", "target": "y_7d_sum",
        "method": "Hurdle", "mode": "Heavy", "tier": "Tier3_Full", "algo": "MLP",
        "label": "Hurdle (Tier3 - MLP)"
    }
]

# ==============================================================================
# PREDICTION ENGINE
# ==============================================================================
def get_features(family, tier):
    if family == "C1":
        return TIERS_C1[tier] + HIST_C1["WithLag1"]
    else:
        return TIERS_C7[tier] + HIST_C7["WithLag1"]

def load_scaler(folder, algo, tier, target, method):
    candidates = [
        folder / f"scaler_{algo}_{tier}_{target}.pkl",
        folder / f"regressor_{method.lower()}_{algo}_{tier}_{target}_scaler.joblib",
        folder / f"scaler_{algo}_{tier}_{target}.joblib"
    ]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    return None

def predict_for_config(df_test, cfg):
    fam = cfg["family"]
    method = cfg["method"]
    mode = cfg["mode"]
    tier = cfg["tier"]
    algo = cfg["algo"]
    target = cfg["target"]
    
    print(f"   -> Predicting: {fam} {method} {tier} {algo}...")

    feats = get_features(fam, tier)
    X = df_test[feats].fillna(0).values
    
    folder_name = f"{fam}_{method}_{mode}"
    model_dir = RESULTS_DIR / folder_name / "saved_models"
    
    scaler = load_scaler(model_dir, algo, tier, target, method)
    if scaler:
        X = scaler.transform(X).astype(np.float32)
    elif algo in ["MLP", "GRU", "GLM"]:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X).astype(np.float32)

    y_pred = None
    
    # 1. Gatekeeper (if Hurdle)
    prob_gate = None
    if method == "Hurdle":
        gk_path = RESULTS_DIR / folder_name / f"champion_gatekeeper_{tier}_{target}.pkl"
        if not gk_path.exists():
             print(f"Gatekeeper missing at {gk_path}")
             return np.zeros(len(X))
        with open(gk_path, "rb") as f: gatekeeper = pickle.load(f)
        prob_gate = gatekeeper.predict_proba(X)[:, 1]

    # 2. Regressor
    base_name = f"regressor_{method.lower()}_{algo}_{tier}_{target}"
    pt_path = model_dir / f"{base_name}.pt"
    
    if pt_path.exists():
        ckpt = torch.load(pt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        input_dim = X.shape[1]
        
        if algo == "MLP":
            hidden_dims = infer_mlp_structure(state_dict)
            model = RobustMLP(input_dim, hidden_dims=hidden_dims)
            
        elif algo == "GRU":
            h_dim, n_lay = infer_gru_structure(state_dict)
            model = RobustGRU(input_dim, hidden_dim=h_dim, num_layers=n_lay)
            
        try:
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            y_raw = predict_with_torch(model, X, is_gru=(algo=="GRU"))
        except Exception as e:
            print(f"Torch load error: {e}")
            return np.zeros(len(X))
            
    else:
        pkl_path = model_dir / f"{base_name}.pkl"
        if not pkl_path.exists():
            print(f"Model missing: {pkl_path}")
            return np.zeros(len(X))
        with open(pkl_path, "rb") as f: model = pickle.load(f)
        y_raw = model.predict(X)

    # 3. Combine
    if method == "Hurdle":
        return prob_gate * y_raw
    else:
        return y_raw

# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_panel_by_timebin(df_agg, family, configs):
    time_bins = [0, 1, 2, 3]
    bin_labels = {0: "Night (00-06)", 1: "Morning (06-12)", 2: "Midday (12-18)", 3: "Evening (18-00)"}
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    target_col = "y" if family == "C1" else "y_7d_sum"
    
    fam_configs = [c for c in configs if c["family"] == family]
    
    for i, tb in enumerate(time_bins):
        ax = axes[i]
        subset = df_agg[df_agg["time_bin"] == tb]
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue
            
        # Plot Reality
        ax.plot(subset["date"], subset[target_col], color="black", linewidth=1.5, label="Actual", alpha=0.7)
        
        # Plot Models (Blue for Direct, Red for Hurdle)
        colors = ["#1f77b4", "#d62728"] # Blue, Red
        for idx, cfg in enumerate(fam_configs):
            col_name = f"pred_{cfg['method']}_{cfg['algo']}"
            if col_name in subset.columns:
                # Use solid line for Direct, Dashed for Hurdle for accessibility/clarity?
                # Or just colors. Let's stick to colors + legend.
                color = "#1f77b4" if cfg["method"] == "Direct" else "#d62728"
                ax.plot(subset["date"], subset[col_name], label=cfg["label"], color=color, linewidth=2, alpha=0.9)
        
        ax.set_title(bin_labels[tb], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if i >= 2: ax.set_xlabel("Date")
        if i % 2 == 0: ax.set_ylabel("Total Crashes")

    # Legend de-duplication
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12)
    
    plt.tight_layout()
    out_path = PLOTS_DIR / f"comparator_{family}_Direct_vs_Hurdle_by_timebin.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=== 02 Time Series Comparator (FINAL CHAMPIONS) ===")
    
    test_path = DATA_PROCESSED / "test_dataset.parquet"
    if not test_path.exists():
        print("Test dataset missing.")
        return
    
    df_test = pd.read_parquet(test_path)
    df_test["date"] = pd.to_datetime(df_test["date"])
    df_test = df_test.sort_values(["date", "time_bin"])
    
    # 1. Run Predictions
    for cfg in MODEL_CONFIGS:
        y_pred = predict_for_config(df_test, cfg)
        col_name = f"pred_{cfg['method']}_{cfg['algo']}"
        prefix = "C1" if cfg["family"] == "C1" else "C7"
        df_test[f"{prefix}_{col_name}"] = y_pred
            
    # 2. Aggregation
    agg_cols = ["y", "y_7d_sum"] + [c for c in df_test.columns if c.startswith("C1_") or c.startswith("C7_")]
    df_agg = df_test.groupby(["date", "time_bin"])[agg_cols].sum().reset_index()
    
    # 3. Plot C1
    c1_configs = [c for c in MODEL_CONFIGS if c["family"] == "C1"]
    df_c1 = df_agg.copy()
    for c in c1_configs:
        # Standardize column name for the plotting function
        df_c1[f"pred_{c['method']}_{c['algo']}"] = df_c1[f"C1_pred_{c['method']}_{c['algo']}"]
    plot_panel_by_timebin(df_c1, "C1", c1_configs)

    # 4. Plot C7
    c7_configs = [c for c in MODEL_CONFIGS if c["family"] == "C7"]
    df_c7 = df_agg.copy()
    for c in c7_configs:
        df_c7[f"pred_{c['method']}_{c['algo']}"] = df_c7[f"C7_pred_{c['method']}_{c['algo']}"]
    plot_panel_by_timebin(df_c7, "C7", c7_configs)
    
    print("\nDone with 02_time_series_comparator.")

if __name__ == "__main__":
    main()