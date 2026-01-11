# scripts/dynamic_11_build_base_panel.py

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# --------------------------------------------------------------------------------------
# PATHS & CONFIG (self-contained)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

CRASHES_PARQUET = DATA_RAW / "crashes.parquet"
STATIC_GRID_PARQUET = DATA_INTER / "static_grid.parquet"
STATIC_FEATURES_PARQUET = DATA_INTER / "static_features.parquet"
WEATHER_SIGMA_SEASON_BINS_PARQUET = DATA_INTER / "weather_sigma_season_bins.parquet"


PANEL_BASE_TRAIN_PARQUET = DATA_INTER / "panel_base_train.parquet"
PANEL_BASE_VALIDATION_PARQUET = DATA_INTER / "panel_base_validation.parquet"
PANEL_BASE_TEST_PARQUET = DATA_INTER / "panel_base_test.parquet"


# Time ranges
TRAIN_START = "2021-01-01"
TRAIN_END   = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

# Time bins: 4 per day
#    0: 00:00–05:59
#    1: 06:00–11:59
#    2: 12:00–17:59
#    3: 18:00–23:59
TIME_BINS = np.array([0, 1, 2, 3], dtype="int8")

# Weight of a fatal crash vs non-fatal in severity score
SEVERITY_DEATH_WEIGHT = 5.0

# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def assign_time_bin(hour: int) -> int:
    """Map hour 0..23 to a 6-hour time bin."""
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    else:
        return 3


def add_lags_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    For one (cell_id, time_bin) series sorted by date, add:
      - y_lag1: y at previous day
      - y_roll7: sum(y over previous 7 days)
    """
    df = df.sort_values("date")
    df["y_lag1"] = df["y"].shift(1)
    df["y_roll7"] = (
        df["y"].rolling(window=7, min_periods=1).sum().shift(1)
    )
    return df

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic numeric downcast: int64->int32/16/8, float64->float32.
    Uses pandas' built-in downcast per column.
    """
    df = df.copy()

    int_cols = df.select_dtypes(include=["int", "int64", "int32"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def cast_panel_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply *explicit* dtypes for key columns,
    then downcast remaining numerics.
    """
    df = df.copy()

    # IDs / keys
    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype("int32")
    if "time_bin" in df.columns:
        df["time_bin"] = df["time_bin"].astype("int8")

    # target & lags
    if "y" in df.columns:
        df["y"] = df["y"].astype("int16")
    if "y_lag1" in df.columns:
        df["y_lag1"] = df["y_lag1"].astype("float32")
    if "y_roll7" in df.columns:
        df["y_roll7"] = df["y_roll7"].astype("float32")

    # crash-severity counts
    if "n_persons_injured" in df.columns:
        df["n_persons_injured"] = df["n_persons_injured"].astype("int16")
    if "n_persons_killed" in df.columns:
        df["n_persons_killed"] = df["n_persons_killed"].astype("int16")

    # DOY
    if "doy" in df.columns:
        df["doy"] = df["doy"].astype("int16")

    # static traffic
    for col in ["vol_static", "max_traffic_volume", "local_traffic_uncertainty"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # severity-based uncertainty (per cell + neighbors)
    for col in [
        "local_severity_uncertainty",
        "neighbor_local_severity_uncertainty",
        "delta_local_severity_uncertainty_vs_neighbors",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # climatology + forecast weather (all temp_*, wind_*, precip_*, snow_*)
    weather_prefixes = ("temp_", "wind_", "precip_", "snow_", "fc_temp_", "fc_wind_", "fc_precip_", "fc_snow_")
    for col in df.columns:
        if col.startswith(weather_prefixes):
            df[col] = df[col].astype("float32")

    # Finally, a generic numeric downcast for anything left
    df = downcast_numeric(df)

    return df

def compute_cell_severity_uncertainty(crash_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cell severity-based uncertainty from crash aggregates.

    We first collapse crash_agg (cell_id, date, time_bin) -> (cell_id, date)
    to get daily counts, then build a simple severity score per day:

        severity_d = crashes_d + SEVERITY_DEATH_WEIGHT * deaths_d

    and finally:

        local_severity_uncertainty = std(severity_d) / (1 + mean(severity_d))

    This uses only days with >=1 crash in the history of the cell. Cells with
    too few crash days end up with 0 uncertainty.
    """
    if crash_agg.empty:
        return pd.DataFrame({"cell_id": [], "local_severity_uncertainty": []})

    daily = (
        crash_agg.groupby(["cell_id", "date"])
        .agg(
            crashes_d=("y", "sum"),
            deaths_d=("n_persons_killed", "sum"),
        )
        .reset_index()
    )

    # severity score per day
    daily["severity_d"] = daily["crashes_d"] + SEVERITY_DEATH_WEIGHT * daily["deaths_d"]

    stats = (
        daily.groupby("cell_id")["severity_d"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "severity_mean", "std": "severity_std"})
    )

    stats["local_severity_uncertainty"] = stats["severity_std"] / (1.0 + stats["severity_mean"])

    # handle NaNs (cells with a single crash day or constant severity)
    stats["local_severity_uncertainty"] = stats["local_severity_uncertainty"].fillna(0.0)

    return stats[["cell_id", "local_severity_uncertainty"]]

def compute_neighbor_severity_features(static_features: pd.DataFrame) -> pd.DataFrame:
    """
    Given static_features with columns:
        - cell_id
        - neighbors (list of neighbor cell_ids)
        - local_severity_uncertainty

    compute:
        - neighbor_local_severity_uncertainty
        - delta_local_severity_uncertainty_vs_neighbors

    Returned DataFrame has one row per cell_id with these new columns.
    """
    if "neighbors" not in static_features.columns or "local_severity_uncertainty" not in static_features.columns:
        # Nothing to do
        return pd.DataFrame(
            {
                "cell_id": static_features.get("cell_id", pd.Series([], dtype="int32")),
                "neighbor_local_severity_uncertainty": [],
                "delta_local_severity_uncertainty_vs_neighbors": [],
            }
        )

    df = static_features[["cell_id", "neighbors", "local_severity_uncertainty"]].copy()

    # Map cell_id -> local_severity_uncertainty for quick lookup
    sev_map = df.set_index("cell_id")["local_severity_uncertainty"].to_dict()

    def neighbor_mean(row):
        neighs = row["neighbors"]
        if neighs is None or (isinstance(neighs, float) and pd.isna(neighs)):
            return np.nan
        if not isinstance(neighs, (list, tuple)):
            return np.nan

        vals = [sev_map.get(n) for n in neighs if n in sev_map]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if not vals:
            return np.nan
        return float(np.mean(vals))

    df["neighbor_local_severity_uncertainty"] = df.apply(neighbor_mean, axis=1)
    df["delta_local_severity_uncertainty_vs_neighbors"] = (
        df["local_severity_uncertainty"] - df["neighbor_local_severity_uncertainty"]
    )

    # Fill NaNs with 0.0 (cells with no valid neighbors)
    df["neighbor_local_severity_uncertainty"] = df["neighbor_local_severity_uncertainty"].fillna(0.0)
    df["delta_local_severity_uncertainty_vs_neighbors"] = df[
        "delta_local_severity_uncertainty_vs_neighbors"
    ].fillna(0.0)

    # Downcast
    df["neighbor_local_severity_uncertainty"] = df["neighbor_local_severity_uncertainty"].astype("float32")
    df["delta_local_severity_uncertainty_vs_neighbors"] = df[
        "delta_local_severity_uncertainty_vs_neighbors"
    ].astype("float32")

    return df[["cell_id", "neighbor_local_severity_uncertainty", "delta_local_severity_uncertainty_vs_neighbors"]]


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Dynamic Step 11: Build Base Panel (crashes + static + climatology + lags) ===")
    print(f"Project root: {ROOT}")
    print(f"DATA_RAW:     {DATA_RAW}")
    print(f"DATA_INTER:   {DATA_INTER}")

    DATA_INTER.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load static grid & static features
    # ------------------------------------------------------------------
    grid = gpd.read_parquet(STATIC_GRID_PARQUET)
    static_features = pd.read_parquet(STATIC_FEATURES_PARQUET)
    print(f"- Loaded static_grid: {len(grid)} cells")
    print(f"- Loaded static_features: {len(static_features)} rows")

    cell_ids = static_features["cell_id"].unique()
    n_cells = len(cell_ids)
    print(f"- Unique static cells: {n_cells}")

    # ------------------------------------------------------------------
    # 2. Build full date x time_bin panel skeleton for 2021-2024
    # ------------------------------------------------------------------
    all_dates = pd.date_range(TRAIN_START, TEST_END, freq="D")
    print(f"- Total days from {TRAIN_START} to {TEST_END}: {len(all_dates)}")

    panel_index = pd.MultiIndex.from_product(
        [cell_ids, all_dates, TIME_BINS],
        names=["cell_id", "date", "time_bin"],
    )
    panel = panel_index.to_frame(index=False)
    print(f"- Base panel skeleton rows: {len(panel)}")

    # ------------------------------------------------------------------
    # 3. Prepare crashes: map to cells & time bins, aggregate counts
    #    (STEP A: injuries & deaths per cell/date/time_bin)
    # ------------------------------------------------------------------
    crashes = pd.read_parquet(CRASHES_PARQUET)
    print(f"- Loaded crashes: {len(crashes)} rows total")

    # Filter by datetime range
    crashes["crash_datetime"] = pd.to_datetime(crashes["crash_datetime"], errors="coerce")
    mask_period = (crashes["crash_datetime"] >= TRAIN_START) & (
        crashes["crash_datetime"] <= TEST_END
    )
    crashes = crashes.loc[mask_period].copy()
    print(f"- Crashes in [2021-2024]: {len(crashes)}")

    # Ensure injury/death counts are numeric
    for col in ["n_persons_injured", "n_persons_killed"]:
        if col in crashes.columns:
            crashes[col] = pd.to_numeric(crashes[col], errors="coerce").fillna(0).astype("int16")
        else:
            crashes[col] = 0

    # Date & time_bin
    crashes["date"] = crashes["crash_datetime"].dt.floor("D")
    crashes["hour"] = crashes["crash_datetime"].dt.hour
    crashes["time_bin"] = crashes["hour"].apply(assign_time_bin).astype("int8")

    # Map crashes to cells: spatial join with static grid
    crash_gdf = gpd.GeoDataFrame(
        crashes,
        geometry=gpd.points_from_xy(crashes["longitude"], crashes["latitude"]),
        crs="EPSG:4326",
    )
    crash_gdf = crash_gdf.to_crs(grid.crs)

    joined = gpd.sjoin(
        crash_gdf,
        grid[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    print(f"- Crashes with a matched cell: {len(joined)}")

    # Aggregate per (cell_id, date, time_bin):
    #   y = crash_count
    #   n_persons_injured, n_persons_killed
    crash_agg = (
        joined.groupby(["cell_id", "date", "time_bin"])
        .agg(
            y=("collision_id", "count"),
            n_persons_injured=("n_persons_injured", "sum"),
            n_persons_killed=("n_persons_killed", "sum"),
        )
        .reset_index()
    )
    crash_agg["date"] = pd.to_datetime(crash_agg["date"])
    print(f"- crash_agg rows: {len(crash_agg)}")

    # ------------------------------------------------------------------
    # 4. Merge crash counts into panel
    # ------------------------------------------------------------------
    panel = panel.merge(
        crash_agg,
        on=["cell_id", "date", "time_bin"],
        how="left",
    )

    # fill missing crash-related columns with 0
    for col in ["y", "n_persons_injured", "n_persons_killed"]:
        panel[col] = panel[col].fillna(0)

    panel["y"] = panel["y"].astype("int16")
    panel["n_persons_injured"] = panel["n_persons_injured"].astype("int16")
    panel["n_persons_killed"] = panel["n_persons_killed"].astype("int16")


    # ------------------------------------------------------------------
    # 5. Add DOY and merge climatological weather (from weather_sigma_season)
    # ------------------------------------------------------------------
    # 5. Add DOY and merge climatological weather (BIN-level: doy + time_bin)
    panel["doy"] = panel["date"].dt.dayofyear.astype("int16")

    weather_season_bins = pd.read_parquet(WEATHER_SIGMA_SEASON_BINS_PARQUET)
    print(f"- Loaded weather_sigma_season_bins: {len(weather_season_bins)} rows")

    panel = panel.merge(
        weather_season_bins,
        on=["doy", "time_bin"],
        how="left",
    )

    # ------------------------------------------------------------------
    # 6. STEP B: compute severity-based uncertainty per cell and merge
    # ------------------------------------------------------------------
    severity_stats = compute_cell_severity_uncertainty(crash_agg)
    print(f"- Computed local_severity_uncertainty for {len(severity_stats)} cells")

    static_features = static_features.merge(severity_stats, on="cell_id", how="left")
    static_features["local_severity_uncertainty"] = (
        static_features["local_severity_uncertainty"].fillna(0.0).astype("float32")
    )

    # ------------------------------------------------------------------
    # 7. STEP C: neighbor severity features (using static neighbor list)
    # ------------------------------------------------------------------
    neighbor_sev = compute_neighbor_severity_features(static_features)
    print(
        "- Computed neighbor severity features for "
        f"{len(neighbor_sev)} cells (neighbor_local_severity_uncertainty, delta_...)"
    )

    static_features = static_features.merge(neighbor_sev, on="cell_id", how="left")

    # ensure no NaNs and proper dtype
    for col in [
        "neighbor_local_severity_uncertainty",
        "delta_local_severity_uncertainty_vs_neighbors",
    ]:
        if col in static_features.columns:
            static_features[col] = static_features[col].fillna(0.0).astype("float32")

    # ------------------------------------------------------------------
    # 8. Merge static features (incl. severity + neighbor severity) into panel
    # ------------------------------------------------------------------
    panel = panel.merge(static_features, on="cell_id", how="left")



    # ------------------------------------------------------------------
    # 7. Add lags of y by (cell_id, time_bin)
    # ------------------------------------------------------------------
    print("- Adding lags y_lag1 and y_roll7 ...")
    panel = (
        panel.sort_values(["cell_id", "time_bin", "date"])
        .groupby(["cell_id", "time_bin"], group_keys=False)
        .apply(add_lags_by_group)
    )

    panel["y_lag1"] = panel["y_lag1"].fillna(0).astype("float32")
    panel["y_roll7"] = panel["y_roll7"].fillna(0).astype("float32")

    print("- Panel with lags built.")

    # ------------------------------------------------------------------
    # 8. Split into train and test and save
    # ------------------------------------------------------------------
    mask_train = (panel["date"] >= TRAIN_START) & (panel["date"] <= TRAIN_END)
    mask_validation = (panel["date"] >= VALIDATION_START) & (panel["date"] <= VALIDATION_END)
    mask_test = (panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)

    panel_train = panel.loc[mask_train].reset_index(drop=True)
    panel_validation = panel.loc[mask_validation].reset_index(drop=True)
    panel_test = panel.loc[mask_test].reset_index(drop=True)

    print(f"- panel_train rows: {len(panel_train)}")
    print(f"- panel_validation rows: {len(panel_validation)}")
    print(f"- panel_test rows:  {len(panel_test)}")

    # --- shrink dtypes before saving ---
    panel_train = cast_panel_dtypes(panel_train)
    panel_validation = cast_panel_dtypes(panel_validation)
    panel_test = cast_panel_dtypes(panel_test)

    PANEL_BASE_TRAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    panel_train.to_parquet(PANEL_BASE_TRAIN_PARQUET, index=False)
    panel_validation.to_parquet(PANEL_BASE_VALIDATION_PARQUET, index=False)
    panel_test.to_parquet(PANEL_BASE_TEST_PARQUET, index=False)

    print(f"Saved base train panel to {PANEL_BASE_TRAIN_PARQUET}")
    print(f"Saved base train panel to {PANEL_BASE_VALIDATION_PARQUET}")
    print(f"Saved base test panel to  {PANEL_BASE_TEST_PARQUET}")



if __name__ == "__main__":
    main()
