# scripts/dynamic_14_build_targets.py

from pathlib import Path
import gc

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# PATHS & CONFIG
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_INTER = ROOT / "data" / "intermediate"
DATA_PROCESSED = ROOT / "data" / "processed"

PANEL_WEATHER_TRAFFIC_TRAIN_PARQUET = DATA_INTER / "panel_weather_traffic_train.parquet"
PANEL_WEATHER_TRAFFIC_VALIDATION_PARQUET = DATA_INTER / "panel_weather_traffic_validation.parquet"
PANEL_WEATHER_TRAFFIC_TEST_PARQUET = DATA_INTER / "panel_weather_traffic_test.parquet"

TRAIN_DATASET_PARQUET = DATA_PROCESSED / "train_dataset.parquet"
VALIDATION_DATASET_PARQUET = DATA_PROCESSED / "validation_dataset.parquet"
TEST_DATASET_PARQUET = DATA_PROCESSED / "test_dataset.parquet"

print(f"ROOT:                 {ROOT}")
print(f"DATA_INTER:           {DATA_INTER}")
print(f"PANEL_TRAIN:          {PANEL_WEATHER_TRAFFIC_TRAIN_PARQUET}")
print(f"PANEL_VALIDATION:     {PANEL_WEATHER_TRAFFIC_VALIDATION_PARQUET}")
print(f"PANEL_TEST:           {PANEL_WEATHER_TRAFFIC_TEST_PARQUET}")
print(f"TRAIN_DATASET:        {TRAIN_DATASET_PARQUET}")
print(f"VALIDATION_DATASET:   {VALIDATION_DATASET_PARQUET}")
print(f"TEST_DATASET:         {TEST_DATASET_PARQUET}")

# --------------------------------------------------------------------------------------
# BUILD TARGETS
# --------------------------------------------------------------------------------------
def build_final_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build final dataset with:
      - crash targets: y_t1, y_7d_sum
      - death targets: d_t1, d_7d_sum
      - time features: dow, month, year

    Assumes panel has columns:
      - cell_id, date, time_bin
      - y (crash count)
      - n_persons_killed (death count per bin)

    IMPORTANT: we assume panel is already sorted by (cell_id, time_bin, date)
    as created in dynamic_11; we do NOT copy or resort here to save memory.
    """
    # Ensure date is datetime (no copy, just overwrite)
    if not np.issubdtype(panel["date"].dtype, np.datetime64):
        panel["date"] = pd.to_datetime(panel["date"])

    # Time features (in-place)
    panel["dow"] = panel["date"].dt.dayofweek.astype("int8")   
    panel["month"] = panel["date"].dt.month.astype("int8")
    panel["year"] = panel["date"].dt.year.astype("int16")

    # Drop neighbors list if still present 
    if "neighbors" in panel.columns:
        panel = panel.drop(columns=["neighbors"])

    # Group by cell & time_bin for shifting; sort=False to avoid extra sorting
    g = panel.groupby(["cell_id", "time_bin"], sort=False, group_keys=False)

    # ------------------------------------------------------------------
    # Crash targets
    # ------------------------------------------------------------------
    # y_t1: crashes tomorrow in same cell & bin
    panel["y_t1"] = g["y"].shift(-1)

    # y_7d_sum: sum of crashes over next 7 days in same cell & bin
    y_7d_sum = None
    for k in range(1, 8):
        shifted = g["y"].shift(-k)
        if y_7d_sum is None:
            y_7d_sum = shifted
        else:
            y_7d_sum = y_7d_sum.add(shifted, fill_value=0)

    panel["y_7d_sum"] = y_7d_sum

    # ------------------------------------------------------------------
    # Death targets (persons killed)
    # ------------------------------------------------------------------
    if "n_persons_killed" in panel.columns:
        # d_t1: deaths tomorrow in same cell & bin
        panel["d_t1"] = g["n_persons_killed"].shift(-1)

        # d_7d_sum: sum of deaths over next 7 days in same cell & bin
        d_7d_sum = None
        for k in range(1, 8):
            shifted_d = g["n_persons_killed"].shift(-k)
            if d_7d_sum is None:
                d_7d_sum = shifted_d
            else:
                d_7d_sum = d_7d_sum.add(shifted_d, fill_value=0)

        panel["d_7d_sum"] = d_7d_sum
    else:
        # Fallback: if somehow not present, set to zero
        panel["d_t1"] = 0.0
        panel["d_7d_sum"] = 0.0

    # ------------------------------------------------------------------
    # Clean up NaNs & dtypes (these allocations are small compared to a full copy)
    # ------------------------------------------------------------------
    # After computing y_t1, y_7d_sum, d_t1, d_7d_sum

    mask = (
        panel["y_t1"].notna()
        & panel["y_7d_sum"].notna()
        & panel["d_t1"].notna()
        & panel["d_7d_sum"].notna()
    )

    panel = panel.loc[mask].copy()

    for col in ["y_t1", "y_7d_sum", "d_t1", "d_7d_sum"]:
        panel[col] = panel[col].astype("float32")


    return panel

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Dynamic Step 14: Build Final Train/Test Datasets with Crash & Death Targets ===")
    print(f"ROOT:          {ROOT}")
    print(f"DATA_INTER:    {DATA_INTER}")
    print(f"DATA_PROCESSED:{DATA_PROCESSED}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. TRAIN
    print("\n- Loading TRAIN panel with weather & traffic")
    train_panel = pd.read_parquet(PANEL_WEATHER_TRAFFIC_TRAIN_PARQUET)
    print(f"  -> {len(train_panel)} rows")

    print("- Building final TRAIN dataset (with targets)")
    train_dataset = build_final_dataset(train_panel)
    del train_panel
    gc.collect()
    print(f"  -> Final train rows (with targets): {len(train_dataset)}")

    train_dataset.to_parquet(TRAIN_DATASET_PARQUET, index=False)
    print(f"Saved train_dataset to {TRAIN_DATASET_PARQUET}")

    # 2. VALIDATION
    print("\n- Loading VALIDATION panel with weather & traffic")
    validation_panel = pd.read_parquet(PANEL_WEATHER_TRAFFIC_VALIDATION_PARQUET)
    print(f"  -> {len(validation_panel)} rows")

    print("- Building final VALIDATION dataset (with targets)")
    validation_dataset = build_final_dataset(validation_panel)
    del validation_panel
    gc.collect()
    print(f"  -> Final validation rows (with targets): {len(validation_dataset)}")

    validation_dataset.to_parquet(VALIDATION_DATASET_PARQUET, index=False)
    print(f"Saved validation_dataset to {VALIDATION_DATASET_PARQUET}")

    # 3. TEST
    print("\n- Loading TEST panel with weather & traffic")
    test_panel = pd.read_parquet(PANEL_WEATHER_TRAFFIC_TEST_PARQUET)
    print(f"  -> {len(test_panel)} rows")

    print("- Building final TEST dataset (with targets)")
    test_dataset = build_final_dataset(test_panel)
    del test_panel
    gc.collect()
    print(f"  -> Final test rows (with targets): {len(test_dataset)}")

    test_dataset.to_parquet(TEST_DATASET_PARQUET, index=False)
    print(f"Saved test_dataset  to {TEST_DATASET_PARQUET}")


if __name__ == "__main__":
    main()
