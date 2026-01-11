import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Try importing tqdm
try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' module not found. Run: pip install tqdm")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Plot output directory
PLOTS_DIR = ROOT / "results" / "plots" / "01_features_analysis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))

from features_scripts.feature_tiers_C7 import TIERS as TIERS_C7, HISTORY_VARIANTS as HIST_C7
from features_scripts.feature_tiers_C1 import TIERS as TIERS_C1, HISTORY_VARIANTS as HIST_C1
from model_definitions import RobustMLP, RobustGRU, predict_with_torch

# ==============================================================================
# 1. HELPERS
# ==============================================================================
def get_feature_names(family, tier, history_var="WithLag1"):
    if family == "C7": return TIERS_C7[tier] + HIST_C7[history_var]
    else: return TIERS_C1[tier] + HIST_C1[history_var]

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
        else: break
    return layers

def infer_gru_structure(state_dict):
    if 'gru.weight_ih_l0' in state_dict:
        hidden_dim = state_dict['gru.weight_ih_l0'].shape[0] // 3
    else: hidden_dim = 16 
    num_layers = 0
    while True:
        if f'gru.weight_ih_l{num_layers}' in state_dict: num_layers += 1
        else: break
    return hidden_dim, num_layers

def load_champion_model(row):
    family, method, tier, algo, target = row["Family"], row["Method"], row["Tier"], row["Algo"], row["Target"]
    mode = row["Mode"]
    
    folder_name = f"{family}_{method}_{mode}" 
    base_name = f"regressor_{method.lower()}_{algo}_{tier}_{target}"
    
    folder = RESULTS_DIR / folder_name / "saved_models"
    model_pt = folder / f"{base_name}.pt"
    model_pkl = folder / f"{base_name}.pkl"
    
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return None, False, folder
    
    model = None
    is_torch = False
    
    try:
        if model_pt.exists():
            ckpt = torch.load(model_pt, map_location="cpu")
            state_dict = ckpt["state_dict"]
            if algo == "MLP":
                model = RobustMLP(state_dict["net.0.weight"].shape[1], hidden_dims=infer_mlp_structure(state_dict))
            elif algo == "GRU":
                h, n = infer_gru_structure(state_dict)
                model = RobustGRU(state_dict["gru.weight_ih_l0"].shape[1], hidden_dim=h, num_layers=n)
            model.load_state_dict(state_dict)
            model.eval()
            is_torch = True
        elif model_pkl.exists():
            with open(model_pkl, "rb") as f: model = joblib.load(f)
            is_torch = False
        else:
            print(f"Model file missing: {base_name}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        
    return model, is_torch, folder

# ==============================================================================
# 2. ANALYSIS METHODS
# ==============================================================================
def analyze_method_a_native(model, is_torch, feature_names, X_sample, algo):
    """Method A: Native / Sensitivity (Returns Zeros for GRU)"""
    importance = np.zeros(len(feature_names))
    method_name = "Unknown"
    
    try:
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            method_name = "Impurity (Gini)"
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
            method_name = "Coeff Magnitude"
        elif is_torch:
            X_tensor = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True)
            if algo == "GRU":
                return np.zeros(len(feature_names)), "Gradient (Skipped for GRU)"
            else:
                pred = model(X_tensor)
                pred.sum().backward()
                grads = X_tensor.grad.abs().mean(dim=0).detach().numpy()
                importance = grads
                method_name = "Input Sensitivity (Grads)"
    except Exception as e:
        print(f"Method A failed: {e}")
        
    return importance, method_name

def analyze_method_b_permutation(model, is_torch, X_sample, y_sample, algo):
    """Method B: Permutation Importance"""
    if is_torch:
        with torch.no_grad():
            pred_base = predict_with_torch(model, X_sample, is_gru=(algo=="GRU")).flatten()
    else:
        pred_base = model.predict(X_sample).flatten()
        
    baseline_rmse = np.sqrt(np.mean((y_sample - pred_base)**2))
    importances = []
    
    for col_idx in range(X_sample.shape[1]):
        X_perm = X_sample.copy()
        np.random.shuffle(X_perm[:, col_idx])
        
        if is_torch:
            with torch.no_grad():
                pred_perm = predict_with_torch(model, X_perm, is_gru=(algo=="GRU")).flatten()
        else:
            pred_perm = model.predict(X_perm).flatten()
            
        perm_rmse = np.sqrt(np.mean((y_sample - pred_perm)**2))
        imp = perm_rmse - baseline_rmse
        importances.append(imp)
        
    return np.array(importances), "Permutation (RMSE Drop)"

# ==============================================================================
# 3. PLOTTING (SEPARATE PLOTS)
# ==============================================================================
def plot_feature_ranking(df, importance_col, method_name, title, color_palette, output_path):
    """
    Plots a single ranking chart.
    Sorts the DataFrame by the specific importance column first.
    """
    # Sort descending by the specific metric
    df_sorted = df.sort_values(importance_col, ascending=False).head(20).copy()
    
    # Normalize for visualization (Optional, but makes bars readable)
    if df_sorted[importance_col].max() > 0:
        df_sorted["Relative Importance"] = df_sorted[importance_col] / df_sorted[importance_col].max()
    else:
        df_sorted["Relative Importance"] = 0
        
    plt.figure(figsize=(8, 10))
    sns.barplot(data=df_sorted, x="Relative Importance", y="Feature", palette=color_palette)
    
    plt.title(f"{title}\n{method_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    print("=== CHAMPION FEATURE ANALYSIS (SEPARATE PLOTS) ===")
    
    METRIC_PASSES = [
        ("rmse", "master_championship_results_rmse.csv"),
        ("poisson_dev", "master_championship_results_poisson_dev.csv")
    ]
    
    for metric_name, csv_filename in METRIC_PASSES:
        csv_path = ROOT / "results" / "plots" / "00_championship_ladder" / csv_filename
        
        print(f"\nPASS: {metric_name.upper()}")
        
        if not csv_path.exists():
            print(f"CSV missing: {csv_path}")
            continue
            
        df_results = pd.read_csv(csv_path)
        champions = df_results[df_results["Mode"] == "Heavy"].copy()
        
        if champions.empty: continue
        
        # Data Load Check
        try:
             pd.read_parquet(DATA_PROCESSED / "test_dataset.parquet", columns=["cell_id"])
        except:
             print("Test data missing."); continue

        for i, row in tqdm(champions.iterrows(), total=len(champions), desc=f"Analyzing {metric_name}"):
            fam, tier, algo, method = row["Family"], row["Tier"], row["Algo"], row["Method"]
            
            # 1. Load Model
            model, is_torch, folder = load_champion_model(row)
            if model is None: continue
                
            # 2. Prepare Data
            feats = get_feature_names(fam, tier)
            target = row["Target"]
            try:
                df_test = pd.read_parquet(DATA_PROCESSED / "test_dataset.parquet", columns=feats + [target]).sample(n=2000, random_state=42)
                X = df_test[feats].fillna(0).values
                y = df_test[target].values
                
                scaler_path = folder / f"scaler_{algo}_{tier}_{target}.pkl"
                if scaler_path.exists():
                    with open(scaler_path, "rb") as f: scaler = joblib.load(f)
                    X = scaler.transform(X).astype(np.float32)
                elif algo in ["MLP", "GRU", "GLM"]:
                    X = StandardScaler().fit_transform(X).astype(np.float32)
            except: continue

            # 3. Analyze & Save CSV (Save raw data in one file)
            imp_a, name_a = analyze_method_a_native(model, is_torch, feats, X, algo)
            imp_b, name_b = analyze_method_b_permutation(model, is_torch, X, y, algo)
            
            df_imp = pd.DataFrame({
                "Feature": feats,
                "Importance_A": imp_a,
                "Importance_B": imp_b
            })
            
            csv_out = PLOTS_DIR / f"importance_{fam}_{method}_{tier}_{algo}_{metric_name}.csv"
            df_imp.to_csv(csv_out, index=False)

            # 4. PLOTTING LOGIC
            base_title = f"{fam} {method} {tier} ({algo})"
            
            # Plot Method B (Permutation) -> ALWAYS DO THIS (Red)
            plot_feature_ranking(
                df_imp, "Importance_B", name_b, 
                base_title, "Reds_r", 
                PLOTS_DIR / f"importance_{fam}_{method}_{tier}_{metric_name}_method_b.png"
            )
            
            # Plot Method A (Native) -> ONLY IF NOT GRU (Blue)
            if algo != "GRU":
                plot_feature_ranking(
                    df_imp, "Importance_A", name_a, 
                    base_title, "Blues_r", 
                    PLOTS_DIR / f"importance_{fam}_{method}_{tier}_{metric_name}_method_a.png"
                )

    print("\nDone.")

if __name__ == "__main__":
    main()