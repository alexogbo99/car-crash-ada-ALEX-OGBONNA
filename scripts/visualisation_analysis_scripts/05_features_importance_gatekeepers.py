import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "results" / "plots" / "06_feature_importance_gatekeepers"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))

# Import Feature Definitions to reconstruct names
from features_scripts.feature_tiers_C7 import TIERS as TIERS_C7, HISTORY_VARIANTS as HIST_C7
from features_scripts.feature_tiers_C1 import TIERS as TIERS_C1, HISTORY_VARIANTS as HIST_C1

# ==============================================================================
# HELPERS
# ==============================================================================

import joblib  # add this

def load_saved_scaler(folder: Path, base_name: str):
    """
    Loads the cached train-fitted scaler saved by run_test_evaluation_* scripts.
    Expected location: <results/<EXP>/saved_models/<base_name>_scaler.joblib>
    """
    sp = folder / "saved_models" / f"{base_name}_scaler.joblib"
    return joblib.load(sp) if sp.exists() else None

def load_gatekeeper_anywhere(results_dir: Path, folder: Path, tier: str, target: str):
    """
    Gatekeeper might be saved in:
      A) <results/<EXP>/champion_gatekeeper_...pkl>
      B) <results/champion_gatekeeper_...pkl>   (global)
    """
    candidates = [
        folder / f"champion_gatekeeper_{tier}_{target}.pkl",
        results_dir / f"champion_gatekeeper_{tier}_{target}.pkl",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    return None

def get_feature_names(family, tier, history_var="WithLag1"):
    """Reconstructs the exact list of feature names used during training."""
    if family == "C7":
        return TIERS_C7[tier] + HIST_C7[history_var]
    else:
        return TIERS_C1[tier] + HIST_C1[history_var]

def plot_importance_bar(importance_df, title, filename, color_palette="Blues_r"):
    """Generic plotter for top 20 features with CSV export"""
    plt.figure(figsize=(10, 8))
    
    # Take top 20
    top_df = importance_df.head(20)
    
    sns.barplot(
        data=top_df,
        x="importance",
        y="feature",
        palette=color_palette,
        edgecolor="black"
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Relative Importance / Coefficient Magnitude", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save plot
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot: {out_path}")
    plt.close()
    
    # Save CSV with all features (not just top 20)
    csv_filename = filename.replace('.png', '.csv')
    csv_path = PLOTS_DIR / csv_filename
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")


# ==============================================================================
# 1. C7 ANALYSIS: What drives CRASH SEVERITY/VOLUME?
# ==============================================================================
def analyze_c7_importance(tier="Tier3_Full"):
    print(f"\nAnalyzing C7 Feature Importance ({tier})...")
    
    # Strategy: Find the best TREE model (RF or HGB) because they interpret well.
    # We scan C7_Direct_Heavy for the best RF/HGB.
    
    path = RESULTS_DIR / "C7_Direct_Heavy" / "train_val_results.json"
    if not path.exists():
        print("C7 Results file missing.")
        return

    try:
        results = json.load(open(path))
        # Filter for Tier 3 and Tree models
        candidates = [
            r for r in results 
            if r["tier"] == tier 
            and r["algo"] in ["RF", "HGB"]
        ]
        
        if not candidates:
            print("No Tree-based models (RF/HGB) found for C7. Skipping.")
            return

        # Pick best one
        best = sorted(candidates, key=lambda x: x["rmse"])[0]
        algo = best["algo"]
        target = best["target"]
        hist = best.get("history", "WithLag1")
        
        print(f"Best Tree Model: {algo} (RMSE: {best['rmse']:.4f})")
        
        # Load Model
        model_name = f"regressor_direct_{algo}_{tier}_{target}.pkl"
        model_path = RESULTS_DIR / "C7_Direct_Heavy" / "saved_models" / model_name
        
        if not model_path.exists():
            print(f"Model file missing: {model_path}")
            return
            
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        # Extract Importances
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
        else:
            print("Model has no feature_importances_ attribute.")
            return
            
        # Map to Names
        feat_names = get_feature_names("C7", tier, hist)
        
        if len(imps) != len(feat_names):
            print(f"Mismatch: Model has {len(imps)} features, but we generated {len(feat_names)} names.")
            # Fallback: create dummy names
            feat_names = [f"Feature_{i}" for i in range(len(imps))]
            
        df_imp = pd.DataFrame({"feature": feat_names, "importance": imps})
        df_imp = df_imp.sort_values("importance", ascending=False)
        
        # Plot
        plot_importance_bar(
            df_imp, 
            f"C7 Feature Importance ({algo})\nDrivers of Crash Volume/Severity", 
            "importance_C7_drivers.png",
            color_palette="Blues_r"
        )
        
    except Exception as e:
        print(f"Error in C7 analysis: {e}")

# ==============================================================================
# 2. C1 ANALYSIS: What drives CRASH PROBABILITY (Gatekeeper)?
# ==============================================================================
def analyze_c1_gatekeeper(tier="Tier3_Full"):
    print(f"\nAnalyzing C1 Gatekeeper (Crash Probability Drivers)...")
    
    # We look for the Gatekeeper pickle in C1_Hurdle_Heavy
    # This file is usually named "champion_gatekeeper_Tier3_Full_y.pkl"
    
    gk_path = RESULTS_DIR / "C1_Hurdle_Heavy" / f"champion_gatekeeper_{tier}_y.pkl"
    
    if not gk_path.exists():
        print(f"Gatekeeper file missing: {gk_path}")
        return
        
    try:
        with open(gk_path, "rb") as f:
            gatekeeper = pickle.load(f)
            
        print(f"   loaded: {type(gatekeeper)}")
        
        # Check if it's a Pipeline (StandardScaler + Model)
        model = gatekeeper
        if hasattr(gatekeeper, "steps"):
            model = gatekeeper.named_steps.get("logisticregression", None)
            if model is None:
                # Try getting the last step if name is unknown
                model = gatekeeper[-1]
                
        # Get Feature Names (C1 Logic)
        feat_names = get_feature_names("C1", tier, "WithLag1") # Defaulting to WithLag1
        
        df_imp = pd.DataFrame()

        # A. Logistic Regression (Coefficients)
        if hasattr(model, "coef_"):
            print("Extracting Logistic Regression Coefficients...")
            coefs = model.coef_[0] # Binary class, take first array

            
            df_imp = pd.DataFrame({"feature": feat_names, "importance": np.abs(coefs), "sign": np.sign(coefs)})
            
        # B. Tree Classifier (Feature Importances)
        elif hasattr(model, "feature_importances_"):
            print("Extracting Tree Importances...")
            imps = model.feature_importances_
            df_imp = pd.DataFrame({"feature": feat_names, "importance": imps})
            
        else:
            print("Unknown model type for interpretation.")
            return
            
        # Sort and Plot
        if not df_imp.empty:
            df_imp = df_imp.sort_values("importance", ascending=False)
            
            plot_importance_bar(
                df_imp, 
                f"C1 Gatekeeper Importance\nWhat triggers a crash probability?", 
                "importance_C1_gatekeeper.png",
                color_palette="Blues_r"
            )
            
    except Exception as e:
        print(f"Error in C1 analysis: {e}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # 1. C7 (Volume) Drivers
    analyze_c7_importance("Tier3_Full")
    
    # 2. C1 (Probability) Drivers
    analyze_c1_gatekeeper("Tier3_Full")
    
    print("\nDone.")