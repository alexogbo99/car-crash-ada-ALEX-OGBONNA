# scripts/feature_selection_01_screening.py
#
# Lightweight feature screening for C1 (crashes, 1-day horizon).
# - Defines logical feature groups (time, history, static traffic, road, infra, weather, forecasts).
# - Loads data/processed/train_dataset.parquet.
# - Samples a subset of rows for RAM safety.
# - Computes:
#     * Spearman corr with y_t1  (count target)
#     * Pearson corr with Z_crash_1d = 1{y_t1 > 0}  (occurrence target)
#     * Mutual information (reg + clf)
#     * HistGradientBoostingRegressor feature importances
# - Saves results to results_feature_screening_C1.csv

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
TRAIN_PATH = DATA_PROCESSED / "train_dataset.parquet"

# --- results structure: results/<target>/<approach>/ ---
TARGET_NAME = "C1"                 
APPROACH_NAME = "feature_screening"

RESULTS_ROOT = ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / TARGET_NAME / APPROACH_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / f"results_feature_screening_{TARGET_NAME}.csv"



# sample size for screening (adjust for RAM)
SAMPLE_SIZE = 250_000  # e.g. 250k rows; if train has fewer, uses full train

TARGET_REG = "y_t1"  # crashes tomorrow (C1)
# Classification target will be Z_crash_1d = 1{y_t1 > 0}


# ============================================================
# FEATURE POOLS (LOGIC-BASED)
# You can tweak these lists if column names differ.
# ============================================================

# Time
FEATURES_TIME = [
    "doy",
    "dow",
    "month",
    "year",
    "time_bin",
]

# History
FEATURES_HISTORY = [
    "y_lag1",
    "y_roll7",
    
]

# Static traffic & neighbor traffic
FEATURES_STATIC_TRAFFIC = [
    "vol_static",
    "max_traffic_volume",
    "local_traffic_uncertainty",
    "max_traffic_volume_tb",
    "vol_static_tb",
    "tod_share",
    # neighbor & deltas (traffic)
    "neighbor_vol_static",
    "neighbor_max_traffic_volume",
    "neighbor_local_traffic_uncertainty",
    "delta_vol_static_vs_neighbors",
    "delta_max_traffic_volume_vs_neighbors",
    "delta_local_traffic_uncertainty_vs_neighbors",
    
]

# Road network
FEATURES_ROAD = [
    "road_segment_count",
    "road_length_m",
    "road_length_density",
    "maxspeed_avg_kph",
    "maxspeed_max_kph",
    "node_count",
    "node_degree_mean",
    "intersection_count",
    "intersection_complexity",
    "intersection_density",
    
]

# Infrastructure
FEATURES_INFRA = [
    "bus_density",
    "subway_density",
    "vz_density",
    "dist_to_center_m",
]

# Realized weather (current bin)
FEATURES_WEATHER_REALIZED = [
    "w_temp_mean",
    "w_wind_mean",
    "w_precip_sum",
    "w_snow_sum",
]

# Weather climatology (DOY x bin)
FEATURES_WEATHER_CLIM = [
    "temp_mean",
    "temp_sigma",
    "wind_mean",
    "wind_sigma",
    "precip_mean",
    "precip_sigma",
    "snow_mean",
    "snow_sigma",
]

# Weather pseudo-forecasts (we include h=1 and h=7 for now)
def make_weather_fc_vars(horizons=(1, 7)):
    vars_ = []
    for h in horizons:
        suffix = f"_h{h}d"
        for var in ["temp", "wind", "precip", "snow"]:
            vars_.append(f"fc_{var}_mean{suffix}")
            vars_.append(f"fc_{var}_q05{suffix}")
            vars_.append(f"fc_{var}_q95{suffix}")
    return vars_


FEATURES_WEATHER_FC = make_weather_fc_vars(horizons=(1, 7))

# Traffic pseudo-forecasts (we exclude q05 because it's always 0)
def make_traffic_fc_vars(horizons=(1, 7)):
    vars_ = []
    for h in horizons:
        suffix = f"_h{h}d"
        vars_.append(f"fc_traffic_mean{suffix}")
        vars_.append(f"fc_traffic_q95{suffix}")
        # fc_traffic_q05_* exists but is all-zero -> we drop it from candidates
    return vars_


FEATURES_TRAFFIC_FC = make_traffic_fc_vars(horizons=(1, 7))

# Extra "other" candidates if you want (example: severity)
FEATURES_OTHER = [
    "local_severity_uncertainty",
    "n_persons_injured",
    "n_persons_killed",
    
]


# Pools grouped in a dict for mapping & reporting
FEATURE_POOLS = {
    "time": FEATURES_TIME,
    "history": FEATURES_HISTORY,
    "static_traffic": FEATURES_STATIC_TRAFFIC,
    "road": FEATURES_ROAD,
    "infra": FEATURES_INFRA,
    "weather_realized": FEATURES_WEATHER_REALIZED,
    "weather_clim": FEATURES_WEATHER_CLIM,
    "weather_fc": FEATURES_WEATHER_FC,
    "traffic_fc": FEATURES_TRAFFIC_FC,
    "other": FEATURES_OTHER,
}


# ============================================================
# HELPER: map feature name → logical group
# ============================================================

def group_of_feature(feature_name: str) -> str:
    for group, cols in FEATURE_POOLS.items():
        if feature_name in cols:
            return group
    return "unassigned"

def build_report(candidate_features,
                 corr_reg=None,
                 corr_clf=None,
                 mi_reg=None,
                 mi_clf=None,
                 hgb_importance=None):
    """
    Build a DataFrame with one row per feature and whatever metrics are available.
    Missing metrics are filled with NaN.
    """
    records = []
    for j, col in enumerate(candidate_features):
        records.append({
            "feature": col,
            "group": group_of_feature(col),
            "corr_reg_y_t1_spearman": float(corr_reg.get(col, np.nan)) if corr_reg is not None else np.nan,
            "corr_clf_Z_crash1d": float(corr_clf.get(col, np.nan)) if corr_clf is not None else np.nan,
            "mi_reg_y_t1": float(mi_reg[j]) if mi_reg is not None else np.nan,
            "mi_clf_Z_crash1d": float(mi_clf[j]) if mi_clf is not None else np.nan,
            "hgb_importance": float(hgb_importance[j]) if hgb_importance is not None else np.nan,
        })

    report = pd.DataFrame(records)
    # sort for readability
    report = report.sort_values(by=["group", "feature"])
    return report


def save_report(stage_name,
                candidate_features,
                corr_reg=None,
                corr_clf=None,
                mi_reg=None,
                mi_clf=None,
                hgb_importance=None):
    """
    Build and save the report to OUTPUT_CSV.
    Overwrites the same file each time, so it's always the latest stage.
    """
    report = build_report(candidate_features, corr_reg, corr_clf, mi_reg, mi_clf, hgb_importance)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved report after {stage_name} to: {OUTPUT_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== Feature Screening for C1 (y_t1) ===")
    print(f"ROOT:           {ROOT}")
    print(f"TRAIN_PATH:     {TRAIN_PATH}")
    print(f"OUTPUT_CSV:     {OUTPUT_CSV}")
    print(f"SAMPLE_SIZE:    {SAMPLE_SIZE}")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train parquet not found at {TRAIN_PATH}")

    print("\nLoading train dataset ...")
    df = pd.read_parquet(TRAIN_PATH)

    # Basic checks
    if TARGET_REG not in df.columns:
        raise ValueError(f"Target column '{TARGET_REG}' not found in train_dataset.parquet.")

    # Sample rows for screening
    n_total = len(df)
    if n_total > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE:,} rows from {n_total:,} ...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        print(f"Using full dataset ({n_total:,} rows).")

    # Build candidate feature list as union of all pools
    all_pool_features = []
    for cols in FEATURE_POOLS.values():
        all_pool_features.extend(cols)
    all_pool_features = list(dict.fromkeys(all_pool_features))  # de-dup while preserving order

    # Keep only those that actually exist and are numeric
    numeric_cols = df.select_dtypes(include=["number"]).columns
    candidate_features = [
        c for c in all_pool_features
        if c in df.columns and c in numeric_cols
    ]

    # Explicitly drop known-bad candidates (all zeros)
    drop_always = [
        "fc_traffic_q05_h1d", "fc_traffic_q05_h2d", "fc_traffic_q05_h3d",
        "fc_traffic_q05_h4d", "fc_traffic_q05_h5d", "fc_traffic_q05_h6d",
        "fc_traffic_q05_h7d",
        "neighbor_local_severity_uncertainty",
        "delta_local_severity_uncertainty_vs_neighbors",
    ]
    candidate_features = [c for c in candidate_features if c not in drop_always]

    print(f"\nNumber of candidate features (existing & numeric): {len(candidate_features)}")
    if len(candidate_features) == 0:
        raise ValueError("No candidate features found; check FEATURE_POOLS vs parquet columns.")

    # Summarize missing features (for your info)
    missing_features = [c for c in all_pool_features if c not in df.columns]
    if missing_features:
        print("\nSome configured features are not present in the parquet (just FYI):")
        for c in missing_features:
            print(f"  - {c}")

    # Targets
    y_reg = df[TARGET_REG].astype("float32")
    Z_crash_1d = (y_reg > 0).astype("int8")

    # Prepare X matrix
    X = df[candidate_features].astype("float32")
    # Fill NaNs with 0 for MI / trees (not perfect, but fine for screening)
    X_filled = X.fillna(0.0)

    # ========================================================
    # 1) Correlations
    # ========================================================
    print("\nComputing correlations ...")

    # Spearman for regression target
    corr_reg = X.corrwith(y_reg, method="spearman")

    # Pearson for binary target (point-biserial)
    corr_clf = X.corrwith(Z_crash_1d.astype("float32"))


    # Save what we have so far (correlations only)
    save_report(
        stage_name="correlations",
        candidate_features=candidate_features,
        corr_reg=corr_reg,
        corr_clf=corr_clf,
        mi_reg=None,
        mi_clf=None,
        hgb_importance=None,
    )




    # ========================================================
    # 2) Mutual Information
    # ========================================================
    print("Computing mutual information (regression + classification) ...")

    X_values = X_filled.values
    y_reg_values = y_reg.values
    y_clf_values = Z_crash_1d.values

    mi_reg = mi_clf = None
    try:
        mi_reg = mutual_info_regression(X_values, y_reg_values, random_state=42)
        mi_clf = mutual_info_classif(X_values, y_clf_values, random_state=42)
    except Exception as e:
        print(f"WARNING: Mutual information computation failed: {e}. "
            "Proceeding without MI (these columns will be NaN).")

    # Save correlations + MI (if available)
    save_report(
        stage_name="correlations_and_MI",
        candidate_features=candidate_features,
        corr_reg=corr_reg,
        corr_clf=corr_clf,
        mi_reg=mi_reg,
        mi_clf=mi_clf,
        hgb_importance=None,
    )

    
    # ========================================================
    # 3) HGB feature importance (robust to older sklearn)
    # ========================================================
    print("Fitting HistGradientBoostingRegressor for feature importances ...")

    # Default: NaN importances (so we don't crash if something fails)
    hgb_importance = np.full(len(candidate_features), np.nan, dtype=float)

    try:
        hgb = HistGradientBoostingRegressor(
            max_depth=6,
            max_iter=100,
            learning_rate=0.1,
            random_state=42,
        )
        hgb.fit(X_values, y_reg_values)

        if hasattr(hgb, "feature_importances_"):
            hgb_importance = np.asarray(hgb.feature_importances_, dtype=float)
        else:
            print(
                "WARNING: HistGradientBoostingRegressor has no feature_importances_ "
                "in this sklearn version. Leaving hgb_importance as NaN."
            )
    except Exception as e:
        print(
            "WARNING: Failed to compute HGB feature importances. "
            f"Reason: {e}. Leaving hgb_importance as NaN."
        )

    # Save full report (corr + MI + HGB importance if available)
    save_report(
        stage_name="correlations_MI_and_HGB",
        candidate_features=candidate_features,
        corr_reg=corr_reg,
        corr_clf=corr_clf,
        mi_reg=mi_reg,
        mi_clf=mi_clf,
        hgb_importance=hgb_importance,
    )



    # ========================================================
    # 4) Build report
    # ========================================================
    print("Building feature report ...")

    records = []
    for j, col in enumerate(candidate_features):
        records.append({
            "feature": col,
            "group": group_of_feature(col),
            "corr_reg_y_t1_spearman": float(corr_reg.get(col, np.nan)),
            "corr_clf_Z_crash1d": float(corr_clf.get(col, np.nan)),
            "mi_reg_y_t1": float(mi_reg[j]),
            "mi_clf_Z_crash1d": float(mi_clf[j]),
            "hgb_importance": float(hgb_importance[j]),
        })

    report = pd.DataFrame(records)
    report = report.sort_values(
        by=["group", "hgb_importance"],
        ascending=[True, False]
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_CSV, index=False)

    print("\n=== DONE ===")
    print(f"Saved feature screening results to: {OUTPUT_CSV}")
    print("Columns in report:")
    print(report.columns.tolist())
    print("\nYou can now inspect this CSV and use it to define Tier1/Tier2/Tier3 feature lists.")


if __name__ == "__main__":
    main()
