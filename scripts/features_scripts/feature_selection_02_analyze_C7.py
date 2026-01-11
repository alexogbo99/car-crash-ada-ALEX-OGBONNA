# scripts/feature_selection_02_analyze.py
#
# Analyze the output of feature_selection_01_screening.py:
#   - Load results_feature_screening_C7.csv
#   - Add absolute correlations
#   - Rank-normalize metrics to [0, 1]
#   - Compute a composite_score per feature
#   - Save:
#       * results_feature_screening_C7_with_scores.csv
#       * results_feature_screening_C7_top_by_group.csv
#   - Print summary tables for interpretation.
#
# This script does NOT assign tiers automatically; it helps YOU see which
# variables are strong within each logical group.

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

TARGET_NAME = "C7"
APPROACH_NAME = "feature_screening"

RESULTS_ROOT = ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / TARGET_NAME / APPROACH_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = RESULTS_DIR / f"results_feature_screening_{TARGET_NAME}.csv"
OUTPUT_DETAILED = RESULTS_DIR / f"results_feature_screening_{TARGET_NAME}_with_scores.csv"
OUTPUT_TOP = RESULTS_DIR / f"results_feature_screening_{TARGET_NAME}_top_by_group.csv"





# How many top features to show/save per group
TOP_K_PER_GROUP = 10

# Threshold for calling a feature "weak" in all metrics (purely informational)
WEAK_CORR_THRESHOLD = 0.01   # abs(corr) below this on both targets
WEAK_MI_THRESHOLD = 1e-4     # MI below this on both tasks
WEAK_IMPORTANCE_THRESHOLD = 1e-4  # importance below this


def rank_normalize(series: pd.Series) -> pd.Series:
    """
    Convert a numeric series into [0, 1] ranks:
      - 0 = lowest value, 1 = highest value.
    NaNs become 0.
    """
    s = series.astype(float)
    # Handle all-NaN or constant series
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    # If all values equal, return 0.5 for non-NaN (everything equal)
    if s.nunique(dropna=True) == 1:
        out = pd.Series(0.5, index=s.index)
        out[s.isna()] = 0.0
        return out

    ranks = s.rank(method="average", na_option="keep")
    min_r, max_r = ranks.min(), ranks.max()
    norm = (ranks - min_r) / (max_r - min_r)
    norm[s.isna()] = 0.0
    return norm


def main():
    print("=== Feature Screening Analysis for C7 ===")
    print(f"ROOT: {ROOT}")
    print(f"INPUT_CSV: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print(f"Loaded {len(df)} features from {INPUT_CSV}")

    required_cols = [
        "feature",
        "group",
        "corr_reg_y_t1_spearman",
        "corr_clf_Z_crash1d",
        "mi_reg_y_t1",
        "mi_clf_Z_crash1d",
        "hgb_importance",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # --------------------------------------------------------
    # 1) Add absolute correlations
    # --------------------------------------------------------
    df["abs_corr_reg_y_t1_spearman"] = df["corr_reg_y_t1_spearman"].abs()
    df["abs_corr_clf_Z_crash1d"] = df["corr_clf_Z_crash1d"].abs()

    # --------------------------------------------------------
    # 2) Rank-normalize metrics to [0, 1]
    # --------------------------------------------------------
    print("Rank-normalizing metrics to [0, 1] ...")

    df["score_corr_reg"] = rank_normalize(df["abs_corr_reg_y_t1_spearman"])
    df["score_corr_clf"] = rank_normalize(df["abs_corr_clf_Z_crash1d"])
    df["score_mi_reg"] = rank_normalize(df["mi_reg_y_t1"])
    df["score_mi_clf"] = rank_normalize(df["mi_clf_Z_crash1d"])
    df["score_importance"] = rank_normalize(df["hgb_importance"])

    # Composite score = average of available scores
    score_cols = [
        "score_corr_reg",
        "score_corr_clf",
        "score_mi_reg",
        "score_mi_clf",
        "score_importance",
    ]
    df["composite_score"] = df[score_cols].mean(axis=1)

    # --------------------------------------------------------
    # 3) Save full table with scores
    # --------------------------------------------------------
    df.to_csv(OUTPUT_DETAILED, index=False)
    print(f"Saved detailed table with scores to: {OUTPUT_DETAILED}")

    # --------------------------------------------------------
    # 4) Top K per group
    # --------------------------------------------------------
    print("\nTop features per group (by composite_score):")

    tops = []
    for grp, sub in df.groupby("group"):
        sub_sorted = sub.sort_values("composite_score", ascending=False)
        top_sub = sub_sorted.head(TOP_K_PER_GROUP).copy()
        tops.append(top_sub)

        print(f"\nGroup: {grp} (n = {len(sub)})")
        print(top_sub[["feature", "composite_score",
                       "abs_corr_reg_y_t1_spearman",
                       "abs_corr_clf_Z_crash1d",
                       "mi_reg_y_t1",
                       "mi_clf_Z_crash1d",
                       "hgb_importance"]].to_string(index=False))

    if tops:
        top_by_group = pd.concat(tops, ignore_index=True)
        top_by_group.to_csv(OUTPUT_TOP, index=False)
        print(f"\nSaved top {TOP_K_PER_GROUP} features per group to: {OUTPUT_TOP}")
    else:
        print("\nNo groups found in the input CSV.")


    # --------------------------------------------------------
    # 5) Weak features: all metrics tiny
    # --------------------------------------------------------
    print("\nIdentifying 'weak' features (all metrics very small) ...")
    weak_mask = (
        (df["abs_corr_reg_y_t1_spearman"] < WEAK_CORR_THRESHOLD) &
        (df["abs_corr_clf_Z_crash1d"] < WEAK_CORR_THRESHOLD) &
        (df["mi_reg_y_t1"] < WEAK_MI_THRESHOLD) &
        (df["mi_clf_Z_crash1d"] < WEAK_MI_THRESHOLD) &
        (df["hgb_importance"] < WEAK_IMPORTANCE_THRESHOLD)
    )

    weak_df = df[weak_mask].sort_values("composite_score")
    print(f"Number of weak features: {len(weak_df)}")

    if len(weak_df) > 0:
        print("\nSample of weak features (candidates to drop or move to Tier3):")
        print(weak_df[["feature", "group",
                       "abs_corr_reg_y_t1_spearman",
                       "abs_corr_clf_Z_crash1d",
                       "mi_reg_y_t1",
                       "mi_clf_Z_crash1d",
                       "hgb_importance"]].head(20).to_string(index=False))

    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
