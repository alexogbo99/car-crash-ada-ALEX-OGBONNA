import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score, roc_auc_score, precision_recall_curve

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "results" / "plots" / "06_gatekeeper_analysis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODE = "Heavy"  # We analyze the Champion "Heavy" models

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from features_scripts.feature_tiers_C7 import TIERS as TIERS_C7, HISTORY_VARIANTS as HIST_C7
from features_scripts.feature_tiers_C1 import TIERS as TIERS_C1, HISTORY_VARIANTS as HIST_C1

# ==============================================================================
# HELPERS
# ==============================================================================
def get_features(family, tier):
    if family == "C1":
        return TIERS_C1[tier] + HIST_C1["WithLag1"]
    else:
        return TIERS_C7[tier] + HIST_C7["WithLag1"]

def compute_lift_deciles(y_true, y_prob):
    """
    Calculates the Lift for each decile (10% buckets).
    Lift = (Positive Rate in Decile) / (Global Positive Rate)
    """
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    df = df.sort_values('p', ascending=False)
    
    # Split into 10 bins
    df['decile'] = pd.qcut(df.index, 10, labels=False) 
    # Note: qcut on index after sort splits into equal chunks
    # Better: explicit array split
    chunks = np.array_split(df, 10)
    
    global_rate = y_true.mean()
    lifts = []
    
    for i, chunk in enumerate(chunks):
        decile_rate = chunk['y'].mean()
        lift = decile_rate / global_rate if global_rate > 0 else 0
        lifts.append(lift)
        
    return lifts # Returns [Lift_Top10%, ..., Lift_Bottom10%]

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================
def plot_calibration_family(results_list, family):
    """Plots Reliability Diagram for all Tiers in a Family"""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    for res in results_list:
        if res['Family'] != family: continue
        plt.plot(res['cal_prob'], res['cal_true'], marker='o', linewidth=2, label=f"{res['Tier']} (Brier={res['Brier']:.4f})")
    
    plt.title(f"Calibration Curve: {family} Gatekeepers")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Actual Risk)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out = PLOTS_DIR / f"calibration_curve_{family}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out.name}")
    plt.close()

def plot_lift_family(results_list, family):
    """Plots Lift by Decile"""
    plt.figure(figsize=(10, 6))
    
    # Deciles 1 to 10
    x = range(1, 11)
    
    for res in results_list:
        if res['Family'] != family: continue
        plt.plot(x, res['lift_curve'], marker='s', linewidth=2, label=res['Tier'])
        
    plt.axhline(1.0, color='k', linestyle='--', label="Random Guessing (Lift=1)")
    plt.title(f"Lift Chart: {family} Gatekeepers")
    plt.xlabel("Decile (1 = Highest Probability, 10 = Lowest)")
    plt.ylabel("Lift (Multiple of Average Crash Rate)")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out = PLOTS_DIR / f"lift_chart_{family}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out.name}")
    plt.close()

def plot_pr_family(results_list, family):
    """Plots Precision-Recall Curve"""
    plt.figure(figsize=(8, 6))
    
    for res in results_list:
        if res['Family'] != family: continue
        plt.plot(res['pr_recall'], res['pr_precision'], linewidth=2, label=f"{res['Tier']} (AP={res['AP']:.3f})")
        
    plt.title(f"Precision-Recall Curve: {family}")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out = PLOTS_DIR / f"pr_curve_{family}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out.name}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=== 04 Gatekeeper Analysis: Probabilistic Metrics ===")
    
    # 1. Load Data
    test_path = DATA_PROCESSED / "test_dataset.parquet"
    if not test_path.exists():
        print(f"Missing: {test_path}")
        return
    print("-> Loading Test Data...")
    df_test = pd.read_parquet(test_path)
    
    champions = [
        {"family": "C1", "tier": "Tier1_Island",  "target": "y"},
        {"family": "C1", "tier": "Tier2_Weather", "target": "y"},
        {"family": "C1", "tier": "Tier3_Full",    "target": "y"},
        {"family": "C7", "tier": "Tier1_Island",  "target": "y_7d_sum"},
        {"family": "C7", "tier": "Tier2_Weather", "target": "y_7d_sum"},
        {"family": "C7", "tier": "Tier3_Full",    "target": "y_7d_sum"},
    ]

    results_data = []

    # 2. Analyze Models
    for champ in champions:
        fam = champ["family"]
        tier = champ["tier"]
        target_col = champ["target"]
        
        print(f"\n🔎 Processing {fam} | {tier}...")
        
        folder = RESULTS_DIR / f"{fam}_Hurdle_{MODE}"
        model_path = folder / f"champion_gatekeeper_{tier}_{target_col}.pkl"
        
        if not model_path.exists():
            print(f"Model missing: {model_path}")
            continue
            
        with open(model_path, "rb") as f: clf = pickle.load(f)
            
        feats = get_features(fam, tier)
        X = df_test[feats].fillna(0).values
        # Binary target: 1 if crashes > 0
        y_true = (df_test[target_col] > 0).astype(int).values
        
        # Get Probabilities
        y_prob = clf.predict_proba(X)[:, 1]
        
        # --- A. Brier Score (MSE of Probabilities) ---
        # Lower is better. Checks calibration + refinement.
        brier = brier_score_loss(y_true, y_prob)
        
        # --- B. Log Loss (Confidence) ---
        ll = log_loss(y_true, y_prob)
        
        # --- C. Average Precision (Area under PR Curve) ---
        ap = average_precision_score(y_true, y_prob)
        
        # --- D. Calibration Curve Data ---
        # We use 10 bins to check alignment
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        
        # --- E. Lift Deciles ---
        lifts = compute_lift_deciles(y_true, y_prob)
        
        # --- F. PR Curve Data ---
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        # Downsample for plotting if too large
        if len(prec) > 1000:
            indices = np.linspace(0, len(prec)-1, 1000).astype(int)
            prec, rec = prec[indices], rec[indices]

        print(f"Brier: {brier:.5f} | LogLoss: {ll:.4f} | AP: {ap:.4f}")
        
        results_data.append({
            "Family": fam,
            "Tier": tier,
            "Brier": brier,
            "LogLoss": ll,
            "AP": ap,
            "cal_true": prob_true,
            "cal_prob": prob_pred,
            "lift_curve": lifts,
            "pr_precision": prec,
            "pr_recall": rec
        })

    # 3. Generate Plots
    if results_data:
        print("\nGenerating Plots...")
        plot_calibration_family(results_data, "C1")
        plot_calibration_family(results_data, "C7")
        
        plot_lift_family(results_data, "C1")
        plot_lift_family(results_data, "C7")
        
        plot_pr_family(results_data, "C1")
        plot_pr_family(results_data, "C7")
        
        # Save Summary CSV
        summary_df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['cal_true', 'cal_prob', 'lift_curve', 'pr_precision', 'pr_recall']} for r in results_data])
        csv_path = PLOTS_DIR / "gatekeeper_probabilistic_metrics.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSummary CSV saved: {csv_path}")

if __name__ == "__main__":
    main()