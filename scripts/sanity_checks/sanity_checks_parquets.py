# scripts/final_02_sanity_check_data_prep_v2.py
#
# Sanity check for the final processed datasets produced by the v2 pipeline:
#   - data/processed/train_dataset.parquet
#   - data/processed/test_dataset.parquet
#
# It runs:
#   1) Structural checks (rows, columns, primary key uniqueness)
#   2) Dead-signal scan (constant / all-zero numeric columns)
#   3) Forecast interval consistency (q05 <= mean <= q95)
#   4) Column overview (name, dtype, missing %)
#   5) Basic target inspection (y, y_t1, y_7d_sum if present)
#   6) PIPELINE-SPECIFIC CHECKS:
#      - Expected feature groups present
#      - Static invariance over time (train)
#      - Train/test temporal split & PK overlap

import os
from pathlib import Path

import numpy as np
import pandas as pd


# =================================================
# CONFIGURATION
# =================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"

DATASETS = [
    ("TRAIN", DATA_PROCESSED / "train_dataset.parquet"),
    ("VALIDATION",  DATA_PROCESSED / "validation_dataset.parquet"),
    ("TEST",  DATA_PROCESSED / "test_dataset.parquet"),
]

# Columns used as primary key in the panel
PK_COLS = ["cell_id", "date", "time_bin"]

# Expected feature groups (pipeline-specific)
EXPECTED_GROUPS = {
    "target": [
        "y",
    ],
    "static_traffic": [
        "vol_static",
        "max_traffic_volume",
        "local_traffic_uncertainty",
        "neighbor_vol_static",
        "neighbor_max_traffic_volume",
        "neighbor_local_traffic_uncertainty",
    ],
    "road_network": [
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
        "neighbor_road_segment_count",
        "neighbor_road_length_density",
        "neighbor_intersection_count",
        "delta_road_segment_count_vs_neighbors",
        "delta_road_length_density_vs_neighbors",
        "delta_intersection_count_vs_neighbors",
    ],
    "infrastructure": [
        "bus_density",
        "subway_density",
        "vz_density",
        "dist_to_center_m",
    ],
    "weather_realized": [
        "w_temp_mean",
        "w_wind_mean",
        "w_precip_sum",
        "w_snow_sum",
    ],
    "weather_fc_example": [
        "fc_temp_mean_h1d",
        "fc_temp_q05_h1d",
        "fc_temp_q95_h1d",
        "fc_precip_mean_h1d",
        "fc_precip_q05_h1d",
        "fc_precip_q95_h1d",
    ],
    "traffic_fc_example": [
        "fc_traffic_mean_h1d",
        "fc_traffic_q05_h1d",
        "fc_traffic_q95_h1d",
    ],
    "lags": [
        "y_lag1",
        "y_roll7",
    ],
}

# Columns that should be static (per cell_id) over time
STATIC_COLS = [
    "vol_static",
    "max_traffic_volume",
    "local_traffic_uncertainty",
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
    "bus_density",
    "subway_density",
    "vz_density",
    "dist_to_center_m",
    "neighbor_vol_static",
    "neighbor_max_traffic_volume",
    "neighbor_local_traffic_uncertainty",
    "neighbor_road_segment_count",
    "neighbor_road_length_density",
    "neighbor_intersection_count",
    "delta_vol_static_vs_neighbors",
    "delta_max_traffic_volume_vs_neighbors",
    "delta_local_traffic_uncertainty_vs_neighbors",
    "delta_road_segment_count_vs_neighbors",
    "delta_road_length_density_vs_neighbors",
    "delta_intersection_count_vs_neighbors",
]

# Columns that must be non-negative
NON_NEGATIVE_COLS = [
    "y",
    "road_segment_count",
    "road_length_m",
    "road_length_density",
    "node_count",
    "intersection_count",
    "intersection_density",
    "bus_density",
    "subway_density",
    "vz_density",
    "w_precip_sum",
    "w_snow_sum",
]


# =================================================
# UTILS (existing)
# =================================================
def check_structural(df: pd.DataFrame, name: str) -> None:
    print("\n" + "=" * 60)
    print(f"[{name}] 1. STRUCTURAL CHECKS")
    print("=" * 60)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Primary key uniqueness
    missing_pk = [c for c in PK_COLS if c not in df.columns]
    if missing_pk:
        print(f"⚠️  Primary key columns missing: {missing_pk}")
    else:
        dup = df.duplicated(subset=PK_COLS).sum()
        if dup == 0:
            print(f"Primary Key ({' + '.join(PK_COLS)}) is UNIQUE.")
        else:
            print(f"WARNING: Found {dup} duplicate rows based on Primary Key!")


def scan_dead_signals(df: pd.DataFrame, name: str) -> None:
    print("\n" + "=" * 60)
    print(f"[{name}] 2. VARIATION & ZERO CHECKS (numeric columns)")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    zero_cols = []
    constant_cols = []
    good_cols_count = 0

    # skip obvious ID-like numeric columns
    skip_cols = {"cell_id", "time_bin"}

    for col in numeric_cols:
        if col in skip_cols:
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if pd.isna(c_min) and pd.isna(c_max):
            # completely NaN: treat as constant but note separately
            constant_cols.append(f"{col} (all NaN)")
        elif c_min == c_max:
            if c_max == 0:
                zero_cols.append(col)
            else:
                constant_cols.append(f"{col} (Value: {c_max})")
        else:
            good_cols_count += 1

    print(f"{good_cols_count} numeric columns have non-trivial variation (min != max).")

    if constant_cols:
        print(f"\n  {len(constant_cols)} columns are CONSTANT (no variation):")
        for c in constant_cols[:20]:
            print(f"   - {c}")
        if len(constant_cols) > 20:
            print(f"   ... and {len(constant_cols) - 20} more.")

    if zero_cols:
        print(f"\n {len(zero_cols)} columns are ALL ZEROS:")
        fc_zeros = [c for c in zero_cols if c.startswith("fc_")]
        other_zeros = [c for c in zero_cols if not c.startswith("fc_")]

        if fc_zeros:
            print("   -> FORECAST VARIABLES (merge/gen may have failed?):")
            for c in fc_zeros[:20]:
                print(f"      - {c}")
            if len(fc_zeros) > 20:
                print(f"      ... and {len(fc_zeros) - 20} more.")

        if other_zeros:
            print("   -> OTHER VARIABLES:")
            for c in other_zeros[:20]:
                print(f"      - {c}")
            if len(other_zeros) > 20:
                print(f"      ... and {len(other_zeros) - 20} more.")
    else:
        print("\nNo all-zero numeric columns found.")


def forecast_interval_checks(df: pd.DataFrame, name: str) -> None:
    """
    Check basic logic: fc_*_q05_h*d <= fc_*_mean_h*d <= fc_*_q95_h*d
    for a few key variables and horizons.
    """
    print("\n" + "=" * 60)
    print(f"[{name}] 3. FORECAST INTERVAL LOGIC CHECKS")
    print("=" * 60)

    vars_to_check = ["temp", "wind", "precip", "snow", "traffic"]
    horizons_to_check = [1, 7]

    any_checked = False
    for var in vars_to_check:
        for h in horizons_to_check:
            mean_col = f"fc_{var}_mean_h{h}d"
            q05_col = f"fc_{var}_q05_h{h}d"
            q95_col = f"fc_{var}_q95_h{h}d"

            if all(c in df.columns for c in [mean_col, q05_col, q95_col]):
                any_checked = True
                violations = df[
                    (df[q05_col] > df[mean_col]) | (df[mean_col] > df[q95_col])
                ]
                if len(violations) > 0:
                    print(
                        f"LOGIC ERROR for {var} h={h}d: "
                        f"{len(violations)} rows with q05 > mean or mean > q95"
                    )
                else:
                    print(f"{var} h={h}d intervals valid: {q05_col} <= {mean_col} <= {q95_col}")

                # Show a small sample
                sample = df[[mean_col, q05_col, q95_col]].head(3)
                print(f"   Sample ({var}, h={h}d):")
                print(sample.to_string(index=False))

    if not any_checked:
        print("No matching forecast columns found for interval checks (fc_*_mean/q05/q95_h*d).")


def column_overview(df: pd.DataFrame, name: str) -> None:
    """
    Print all columns with dtype and missing percentage.
    """
    print("\n" + "=" * 60)
    print(f"[{name}] 4. COLUMN OVERVIEW (name, dtype, missing%)")
    print("=" * 60)

    n = len(df)
    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        n_missing = df[col].isna().sum()
        miss_pct = (n_missing / n * 100.0) if n > 0 else 0.0
        rows.append((col, str(dtype), n_missing, miss_pct))

    overview_df = pd.DataFrame(
        rows, columns=["column", "dtype", "n_missing", "missing_pct"]
    ).sort_values("column")

    # Print summary
    print(overview_df.to_string(index=False, max_rows=200))


def target_inspection(df: pd.DataFrame, name: str) -> None:
    """
    Quick summary for potential target columns: y, y_t1, y_7d_sum, has_future, etc.
    """
    print("\n" + "=" * 60)
    print(f"[{name}] 5. TARGET & LABEL INSPECTION")
    print("=" * 60)

    candidate_targets = ["y", "y_t1", "y_7d_sum", "y_crash_count", "has_future"]

    for col in candidate_targets:
        if col in df.columns:
            series = df[col]
            print(
                f"- {col}: dtype={series.dtype}, "
                f"min={series.min()}, max={series.max()}, "
                f"mean={series.mean():.4f}, non-null={series.notna().sum():,}"
            )
        else:
            print(f"- {col}: (not present)")


# =================================================
# NEW PIPELINE-SPECIFIC CHECKS
# =================================================
def check_expected_columns(df: pd.DataFrame, name: str) -> None:
    print("\n" + "=" * 60)
    print(f"[{name}] 6. EXPECTED FEATURE GROUPS")
    print("=" * 60)

    for group, cols in EXPECTED_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        print(f"\nGroup '{group}': {len(present)} present, {len(missing)} missing")
        if present:
            print("   Present:", present)
        if missing:
            print("   Missing:", missing)


def check_static_invariance(df: pd.DataFrame) -> None:
    """
    For columns that should be static per cell_id, verify that they do not
    vary over time in TRAIN dataset.
    """
    print("\n" + "=" * 60)
    print("[TRAIN] 7. STATIC INVARIANCE (per cell_id)")
    print("=" * 60)

    if "cell_id" not in df.columns:
        print("No 'cell_id' column; skipping static invariance check.")
        return

    static_cols = [c for c in STATIC_COLS if c in df.columns]
    if not static_cols:
        print("No static columns found to check.")
        return

    for col in static_cols:
        nunq = df.groupby("cell_id")[col].nunique(dropna=False).max()
        if nunq > 1:
            print(f"Column '{col}' varies over time within at least one cell (max nunique={nunq}).")
        else:
            print(f"Column '{col}' is static per cell_id (as expected).")

    # Non-negativity check for key columns (train only)
    print("\n[TRAIN] 8. NON-NEGATIVITY CHECKS (selected columns)")
    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            min_val = df[col].min()
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                print(f"{col}: min={min_val}, {n_neg} negative values.")
            else:
                print(f"{col}: min={min_val}, no negative values.")
        else:
            print(f"{col}: not present in TRAIN; skipping.")


def check_train_test_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Joint checks on TRAIN+TEST:
      - Date ranges and ordering
      - PK overlap between train and test
    """
    print("\n" + "=" * 60)
    print("GLOBAL: 9. TRAIN/TEST SPLIT INTEGRITY")
    print("=" * 60)

    if "date" not in train_df.columns or "date" not in test_df.columns:
        print("'date' column missing in one of the datasets; skipping date range check.")
    else:
        train_min, train_max = train_df["date"].min(), train_df["date"].max()
        test_min, test_max = test_df["date"].min(), test_df["date"].max()
        print(f"TRAIN date range: {train_min} -> {train_max}")
        print(f"TEST  date range: {test_min} -> {test_max}")

        if test_min > train_max:
            print("TEST starts strictly after TRAIN ends (no temporal leakage).")
        elif test_min == train_max:
            print("TEST starts on the same day TRAIN ends; check your intended split.")
        else:
            print("TEST begins before TRAIN ends; temporal leakage risk!")

    # PK overlap
    missing_pk_train = [c for c in PK_COLS if c not in train_df.columns]
    missing_pk_test = [c for c in PK_COLS if c not in test_df.columns]
    if missing_pk_train or missing_pk_test:
        print(f"Cannot check PK overlap; missing PK cols in train={missing_pk_train}, test={missing_pk_test}")
    else:
        train_pk = train_df[PK_COLS]
        test_pk = test_df[PK_COLS]
        merged = train_pk.merge(test_pk, on=PK_COLS, how="inner")
        overlap = len(merged)
        if overlap == 0:
            print("No PK overlap between TRAIN and TEST.")
        else:
            print(f"Found {overlap} PK overlaps between TRAIN and TEST.")


# =================================================
# MAIN
# =================================================
def main():
    print("=== FINAL SANITY CHECK FOR DATA PREP (v2) ===")
    print(f"Root:           {ROOT}")
    print(f"DATA_PROCESSED: {DATA_PROCESSED}")

    loaded = {}

    for name, path in DATASETS:
        print("\n" + "#" * 60)
        print(f"### DATASET: {name} ({path})")
        print("#" * 60)

        if not path.exists():
            print(f"Error: {path} not found, skipping.")
            continue

        print(f"Loading {path} ...")
        df = pd.read_parquet(path)
        loaded[name] = df

        # Run existing checks
        check_structural(df, name)
        scan_dead_signals(df, name)
        forecast_interval_checks(df, name)
        column_overview(df, name)
        target_inspection(df, name)
        check_expected_columns(df, name)

        # Additional static invariance checks only on TRAIN
        if name == "TRAIN":
            check_static_invariance(df)

    # Joint checks if both TRAIN and TEST loaded
    if "TRAIN" in loaded and "TEST" in loaded:
        check_train_test_split(loaded["TRAIN"], loaded["TEST"])

    print("\n" + "=" * 60)
    print("SANITY CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
