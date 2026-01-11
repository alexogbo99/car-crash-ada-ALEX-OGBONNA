import sys
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from shapely.geometry import box
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INTER = ROOT / "data" / "intermediate"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "results" / "plots" / "04_spatial_residuals"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Grid Configuration (MATCH static_01_build_active_grid.py)
GRID_RES_DEG = 0.005
NYC_BOUNDS = {
    "minx": -74.257,
    "miny": 40.495,
    "maxx": -73.699,
    "maxy": 40.916,
}

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))

from model_definitions import RobustMLP, RobustGRU, predict_with_torch
from features_scripts.feature_tiers_C7 import TIERS as TIERS_C7, HISTORY_VARIANTS as HIST_C7
from features_scripts.feature_tiers_C1 import TIERS as TIERS_C1, HISTORY_VARIANTS as HIST_C1

# ==============================================================================
# 1. MODEL LOADING FUNCTIONS
# ==============================================================================

def load_saved_scaler(folder: Path, base_name: str):
    """
    Loads the cached train-fitted scaler saved by run_test_evaluation_* scripts.
    """
    sp = folder / "saved_models" / f"{base_name}_scaler.joblib"
    return joblib.load(sp) if sp.exists() else None

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

def load_specific_model(model_config):
    family = model_config["family"]
    method = model_config["method"]
    target = model_config["target"]
    tier = model_config["tier"]
    regressor_path = model_config["regressor_path"]
    gatekeeper_path = model_config.get("gatekeeper_path")
    
    print(f"\n🔍 Loading Specific Model: {family} {method} - {target}...")
    
    algo = model_config.get("algo", "Unknown")
    
    # Load regressor model
    model = None
    if str(regressor_path).endswith(".pt"):
        ckpt = torch.load(regressor_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        
        if algo == "MLP":
            if "net.0.weight" in state_dict:
                in_dim = state_dict["net.0.weight"].shape[1]
            else:
                for key in state_dict:
                    if "weight" in key and key.endswith(".0.weight"):
                        in_dim = state_dict[key].shape[1]; break
                else: raise ValueError("Cannot infer input dimension")
            
            layers = infer_mlp_structure(state_dict)
            model = RobustMLP(in_dim, hidden_dims=layers)
        
        elif algo == "GRU":
            if "gru.weight_ih_l0" in state_dict:
                in_dim = state_dict["gru.weight_ih_l0"].shape[1]
                h_dim, n_lay = infer_gru_structure(state_dict)
                model = RobustGRU(in_dim, hidden_dim=h_dim, num_layers=n_lay)
            else: raise ValueError("Cannot infer GRU structure")
        
        model.load_state_dict(state_dict)
        model.eval()
    
    elif str(regressor_path).endswith(".pkl"):
        with open(regressor_path, "rb") as f: model = pickle.load(f)
    
    # Load gatekeeper
    gatekeeper = None
    if method == "Hurdle" and gatekeeper_path and gatekeeper_path.exists():
        with open(gatekeeper_path, "rb") as f: gatekeeper = pickle.load(f)
    
    folder = regressor_path.parent.parent
    info = {
        "family": family,
        "method": method,
        "target": target,
        "tier": tier,
        "algo": algo,
        "folder": folder,
    }
    
    # Get history variant
    history_var = "WithLag1"
    metrics_file = folder / "final_test_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f: metrics_data = json.load(f)
            if target in metrics_data and tier in metrics_data[target]:
                history_var = metrics_data[target][tier].get("history", "WithLag1")
        except: pass
    
    return model, gatekeeper, info, history_var

# ==============================================================================
# 2. PREDICTION & GEOMETRY
# ==============================================================================

def get_spatial_predictions_specific(model_config):
    """Get predictions grouped by CELL + TIME_BIN"""
    model, gatekeeper, info, history_var = load_specific_model(model_config)
    if model is None: return None, None
    
    family = info["family"]
    tier = info["tier"]
    target = info["target"]
    algo = info["algo"]
    
    if family == "C7": features = TIERS_C7[tier] + HIST_C7[history_var]
    else: features = TIERS_C1[tier] + HIST_C1[history_var]
    
    # Load Test Data
    print("Loading Test Parquet...")
    req_cols = list(set(features + [target, "cell_id", "date", "time_bin"]))
    df = pd.read_parquet(DATA_PROCESSED / "test_dataset.parquet", columns=req_cols)
    
    X = df[features].fillna(0).values
    y_true = df[target].values
    
    # Scale
    if info["method"] == "Direct": base_name = f"regressor_direct_{algo}_{tier}_{target}"
    else: base_name = f"regressor_hurdle_{algo}_{tier}_{target}"
    
    if algo in ["MLP", "GRU"]:
        scaler = load_saved_scaler(info["folder"], base_name)
        if scaler: X_in = scaler.transform(X).astype(np.float32)
        else: X_in = X.astype(np.float32)
    else: X_in = X
    
    # Predict
    print(f"Predicting with {algo}...")
    if algo in ["MLP", "GRU"]:
        pred_reg = predict_with_torch(model, X_in, is_gru=(algo=="GRU"))
    else:
        pred_reg = model.predict(X_in)
    
    if info["method"] == "Hurdle" and gatekeeper is not None:
        prob_crash = gatekeeper.predict_proba(X)[:, 1]
        final_pred = prob_crash * pred_reg
    else:
        final_pred = pred_reg
    
    df["y_true"] = y_true
    df["y_pred"] = final_pred
    df["residual"] = df["y_true"] - df["y_pred"]
    
    # --- Group by CELL_ID AND TIME_BIN ---
    spatial_df = df.groupby(["cell_id", "time_bin"])[["y_true", "y_pred", "residual"]].sum().reset_index()
    return spatial_df, info

def reconstruct_nyc_grid():
    print("Reconstructing Grid Geometry...")
    minx, miny = NYC_BOUNDS["minx"], NYC_BOUNDS["miny"]
    maxx, maxy = NYC_BOUNDS["maxx"], NYC_BOUNDS["maxy"]
    res = GRID_RES_DEG
    x_coords = np.arange(minx, maxx, res)
    y_coords = np.arange(miny, maxy, res)
    polys = []
    for x in x_coords:
        for y in y_coords:
            polys.append(box(x, y, x + res, y + res))
    gdf = gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:4326")
    gdf["cell_id"] = gdf.index.astype(int)
    return gdf

def load_or_reconstruct_grid():
    grid_path = DATA_INTER / "static_grid.parquet"
    if grid_path.exists(): return gpd.read_parquet(grid_path)
    grid_path_geo = DATA_PROCESSED / "nyc_grid_500m.geojson"
    if grid_path_geo.exists(): return gpd.read_file(grid_path_geo)
    return reconstruct_nyc_grid()

# ==============================================================================
# 3. PLOTTING (PANEL VERSION)
# ==============================================================================

def plot_residual_panel(spatial_df, info, grid_gdf, suffix=""):
    """
    Creates a 2x2 Panel Plot (Night, Morning, Midday, Evening)
    """
    if grid_gdf is None: return

    # Merge Geometry
    spatial_df["cell_id"] = spatial_df["cell_id"].astype(int)
    grid_gdf["cell_id"] = grid_gdf["cell_id"].astype(int)
    
    # Merge LEFT to keep all grid cells if needed, or INNER to show only active
    # Using INNER here to match data availability
    gdf_viz_all = grid_gdf.merge(spatial_df, on="cell_id", how="inner")
    
    if gdf_viz_all.empty:
        print("Merge resulted in empty map.")
        return

    # Projection
    if gdf_viz_all.crs is None: gdf_viz_all.set_crs(epsg=4326, inplace=True)
    gdf_viz_all = gdf_viz_all.to_crs(epsg=3857)

    # Determine Global Scale for this Model (Symmetric)
    limit = max(abs(gdf_viz_all["residual"].min()), abs(gdf_viz_all["residual"].max()))
    
    # Setup Figure (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    time_labels = {
        0: "Night (00-06)", 
        1: "Morning (06-12)", 
        2: "Midday (12-18)", 
        3: "Evening (18-00)"
    }

    for tb in range(4):
        ax = axes[tb]
        data_bin = gdf_viz_all[gdf_viz_all["time_bin"] == tb]
        
        if data_bin.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            ax.axis("off")
            continue

        data_bin.plot(
            column="residual",
            ax=ax,
            cmap="RdBu_r", 
            alpha=0.8,
            vmin=-limit/2, 
            vmax=limit/2, 
            legend=True,
            legend_kwds={'label': "Residual", 'shrink': 0.5}
        )
        
        try: ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except: pass
        
        ax.set_title(time_labels[tb], fontsize=14, fontweight='bold')
        ax.axis("off")

    # Global Title
    family = info["family"]
    method = info["method"]
    algo = info["algo"]
    plt.suptitle(f"Spatial Bias by Time of Day: {family} {method} ({algo})", fontsize=20, fontweight='bold', y=0.95)
    
    out_path = PLOTS_DIR / f"spatial_residuals_panel_{family}_{method}_{algo}{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Panel: {out_path}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=== SPATIAL RESIDUAL ANALYSIS (PANEL BY TIMEBIN) ===")
    
    # 1. Geometry
    grid_gdf = load_or_reconstruct_grid()
    
    # 2. Configs (The 4 Champions)
    model_configs = [
        {
            "name": "C7_Direct",
            "family": "C7", "method": "Direct", "target": "y_7d_sum", "tier": "Tier3_Full",
            "regressor_path": RESULTS_DIR / "C7_Direct_Heavy" / "saved_models" / "regressor_direct_RF_Tier3_Full_y_7d_sum.pkl",
            "algo": "RF"
        },
        {
            "name": "C7_Hurdle",
            "family": "C7", "method": "Hurdle", "target": "y_7d_sum", "tier": "Tier3_Full",
            "regressor_path": RESULTS_DIR / "C7_Hurdle_Heavy" / "saved_models" / "regressor_hurdle_MLP_Tier3_Full_y_7d_sum.pt",
            "gatekeeper_path": RESULTS_DIR / "C7_Hurdle_Heavy" / "champion_gatekeeper_Tier3_Full_y_7d_sum.pkl",
            "algo": "MLP"
        },
        {
            "name": "C1_Direct",
            "family": "C1", "method": "Direct", "target": "y", "tier": "Tier3_Full",
            "regressor_path": RESULTS_DIR / "C1_Direct_Heavy" / "saved_models" / "regressor_direct_GRU_Tier3_Full_y.pt",
            "algo": "GRU"
        },
        {
            "name": "C1_Hurdle",
            "family": "C1", "method": "Hurdle", "target": "y", "tier": "Tier3_Full",
            "regressor_path": RESULTS_DIR / "C1_Hurdle_Heavy" / "saved_models" / "regressor_hurdle_MLP_Tier3_Full_y.pt",
            "gatekeeper_path": RESULTS_DIR / "C1_Hurdle_Heavy" / "champion_gatekeeper_Tier3_Full_y.pkl",
            "algo": "MLP"
        }
    ]
    
    for config in model_configs:
        print(f"\nProcessing: {config['name']}...")
        try:
            df_result, info_result = get_spatial_predictions_specific(config)
            if df_result is not None:
                # Plot Panel
                plot_residual_panel(df_result, info_result, grid_gdf, suffix=f"_{config['name']}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
            
    print("\nDone.")