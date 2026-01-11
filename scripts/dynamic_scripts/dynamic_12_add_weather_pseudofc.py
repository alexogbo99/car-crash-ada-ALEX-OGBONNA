# scripts/dynamic_12_add_weather_pseudofc.py

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# PATHS & CONFIG
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

PANEL_BASE_TRAIN_PARQUET = DATA_INTER / "panel_base_train.parquet"
PANEL_BASE_VALIDATION_PARQUET = DATA_INTER / "panel_base_validation.parquet"  
PANEL_BASE_TEST_PARQUET = DATA_INTER / "panel_base_test.parquet"

WEATHER_SIGMA_SEASON_BINS_PARQUET = DATA_INTER / "weather_sigma_season_bins.parquet"
WEATHER_HOURLY_PARQUET = DATA_RAW / "weather_hourly.parquet"

PANEL_WEATHER_TRAIN_PARQUET = DATA_INTER / "panel_weather_train.parquet"
PANEL_WEATHER_VALIDATION_PARQUET = DATA_INTER / "panel_weather_validation.parquet"
PANEL_WEATHER_TEST_PARQUET = DATA_INTER / "panel_weather_test.parquet"

HORIZONS = list(range(1, 8))  

# how much to scale sigma as horizon increases
SIGMA_SCALE = {
    1: 0.3,
    2: 0.4,
    3: 0.5,
    4: 0.6,
    5: 0.7,
    6: 0.8,
    7: 0.9,
}

Z_05 = 1.6448536269514722  # Normal quantile for 5% / 95%

# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def assign_time_bin(hour: int) -> int:
    """Map hour 0..23 to a 6-hour time bin (0..3), same as in dynamic_11."""
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    else:
        return 3

def build_realized_weather_bins(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly weather to (date, time_bin):

        w_temp_mean   = mean(temp)
        w_wind_mean   = mean(wind)
        w_precip_sum  = sum(precip)
        w_snow_sum    = sum(snow)

    Returns:
        DataFrame with columns: date, time_bin, w_temp_mean, w_wind_mean,
        w_precip_sum, w_snow_sum
    """
    df = weather_df.copy()

    # Ensure timestamp is proper datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Date & hour
    df["date"] = df["timestamp"].dt.floor("D")
    df["hour"] = df["timestamp"].dt.hour

    # Map hour -> time_bin (0..3)
    df["time_bin"] = df["hour"].apply(assign_time_bin).astype("int8")

    # Aggregate by (date, time_bin)
    agg = (
        df.groupby(["date", "time_bin"])
        .agg(
            w_temp_mean=("temp", "mean"),
            w_wind_mean=("wind", "mean"),
            # precip: sum over the 6h bin
            w_precip_sum=("precip", "sum"),
            w_snow_sum=("snow", "sum"),
        )
        .reset_index()
    )

    # Make sure date is datetime64[ns]
    agg["date"] = pd.to_datetime(agg["date"])

    # Cast to float32 to keep things light
    for col in ["w_temp_mean", "w_wind_mean", "w_precip_sum", "w_snow_sum"]:
        agg[col] = agg[col].astype("float32")

    return agg


# --------------------------------------------------------------------------------------
# CORE LOGIC
# --------------------------------------------------------------------------------------
def add_weather_forecasts(panel: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """
    Given a base panel (with 'date' and 'time_bin' columns) and a climatology
    table 'clim' with columns:
        doy, time_bin,
        temp_mean, temp_sigma,
        wind_mean, wind_sigma,
        precip_mean, precip_sigma,
        snow_mean, snow_sigma

    add forecast features:
        fc_<var>_mean_h{h}d, fc_<var>_q05_h{h}d, fc_<var>_q95_h{h}d
    for var in {temp, wind, precip, snow}, h = 1..7.
    """
    panel = panel.copy()

    if "time_bin" not in panel.columns:
        raise ValueError("Panel must contain 'time_bin' for bin-specific climatology.")

    # Indexed by (doy, time_bin)
    clim_idx = clim.set_index(["doy", "time_bin"])

    dates = panel["date"]
    time_bins = panel["time_bin"].astype("int16").values

    for h in HORIZONS:
        print(f"  - Adding weather forecasts for h={h} day(s) ahead")

        future_dates = dates + pd.to_timedelta(h, unit="D")
        future_doy = future_dates.dt.dayofyear.astype("int16").values

        # Build a MultiIndex of lookup keys for each row
        key = pd.MultiIndex.from_arrays(
            [future_doy, time_bins],
            names=["doy", "time_bin"],
        )

        # Lookup corresponding climatology rows
        clim_lookup = clim_idx.loc[key]

        t_mean = clim_lookup["temp_mean"].to_numpy(dtype="float32")
        t_sigma = clim_lookup["temp_sigma"].to_numpy(dtype="float32") * SIGMA_SCALE[h]

        w_mean = clim_lookup["wind_mean"].to_numpy(dtype="float32")
        w_sigma = clim_lookup["wind_sigma"].to_numpy(dtype="float32") * SIGMA_SCALE[h]

        p_mean = clim_lookup["precip_mean"].to_numpy(dtype="float32")
        p_sigma = clim_lookup["precip_sigma"].to_numpy(dtype="float32") * SIGMA_SCALE[h]

        s_mean = clim_lookup["snow_mean"].to_numpy(dtype="float32")
        s_sigma = clim_lookup["snow_sigma"].to_numpy(dtype="float32") * SIGMA_SCALE[h]

        # temp
        panel[f"fc_temp_mean_h{h}d"] = t_mean
        panel[f"fc_temp_q05_h{h}d"] = t_mean - Z_05 * t_sigma
        panel[f"fc_temp_q95_h{h}d"] = t_mean + Z_05 * t_sigma

        # wind
        panel[f"fc_wind_mean_h{h}d"] = w_mean
        panel[f"fc_wind_q05_h{h}d"] = w_mean - Z_05 * w_sigma
        panel[f"fc_wind_q95_h{h}d"] = w_mean + Z_05 * w_sigma

        # precip
        panel[f"fc_precip_mean_h{h}d"] = p_mean
        panel[f"fc_precip_q05_h{h}d"] = p_mean - Z_05 * p_sigma
        panel[f"fc_precip_q95_h{h}d"] = p_mean + Z_05 * p_sigma

        # snow
        panel[f"fc_snow_mean_h{h}d"] = s_mean
        panel[f"fc_snow_q05_h{h}d"] = s_mean - Z_05 * s_sigma
        panel[f"fc_snow_q95_h{h}d"] = s_mean + Z_05 * s_sigma

    return panel


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
    Light-weight dtype cleanup:
    - enforce compact ints for IDs
    - enforce float32 for lags
    - DO NOT make a full copy or global downcast (too heavy for 18M rows).
    """
    # operate in-place to avoid huge copies
    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype("int32")
    if "time_bin" in df.columns:
        df["time_bin"] = df["time_bin"].astype("int8")

    if "y" in df.columns:
        df["y"] = df["y"].astype("int16")
    if "y_lag1" in df.columns:
        df["y_lag1"] = df["y_lag1"].astype("float32")
    if "y_roll7" in df.columns:
        df["y_roll7"] = df["y_roll7"].astype("float32")

    if "doy" in df.columns:
        df["doy"] = df["doy"].astype("int16")

    # Static traffic if present (in dynamic_12 we might not need this, but harmless)
    for col in ["vol_static", "max_traffic_volume", "local_traffic_uncertainty"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # Weather climatology & forecasts: all created as float32 already,
    
    weather_prefixes = (
        "temp_", "wind_", "precip_", "snow_",
        "fc_temp_", "fc_wind_", "fc_precip_", "fc_snow_"
    )
    for col in df.columns:
        if col.startswith(weather_prefixes):
            df[col] = df[col].astype("float32")

        # Realized per-bin weather (added in this script)
    for col in ["w_temp_mean", "w_wind_mean", "w_precip_sum", "w_snow_sum"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    return df

    return df

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Dynamic Step 12: Add Realized Weather + Weather Pseudo-Forecasts (h=1..7) ===")
    print(f"ROOT:        {ROOT}")
    print(f"DATA_RAW:    {DATA_RAW}")
    print(f"DATA_INTER:  {DATA_INTER}")

    # 1. Load base panels (already have crashes + static + climatology + lags)
    panel_train = pd.read_parquet(PANEL_BASE_TRAIN_PARQUET)
    panel_validation = pd.read_parquet(PANEL_BASE_VALIDATION_PARQUET)
    panel_test = pd.read_parquet(PANEL_BASE_TEST_PARQUET)
    print(f"- Loaded base train panel: {len(panel_train)} rows")
    print(f"- Loaded base train panel: {len(panel_validation)} rows")
    print(f"- Loaded base test panel:  {len(panel_test)} rows")

    # 2. Load hourly weather and build realized per-bin aggregates
    weather_hourly = pd.read_parquet(WEATHER_HOURLY_PARQUET)
    print(f"- Loaded weather_hourly: {len(weather_hourly)} rows")

    weather_bin = build_realized_weather_bins(weather_hourly)
    print(f"- Built realized weather bins: {len(weather_bin)} rows (date x time_bin)")

    # 3. Merge realized weather into base panels (no spatial variation)
    print("- Merging realized weather into TRAIN panel")
    panel_train = panel_train.merge(weather_bin, on=["date", "time_bin"], how="left")

    print("- Merging realized weather into VALIDATION panel")
    panel_validation = panel_validation.merge(weather_bin, on=["date", "time_bin"], how="left")

    print("- Merging realized weather into TEST panel")
    panel_test = panel_test.merge(weather_bin, on=["date", "time_bin"], how="left")

    # 4. Load bin-level climatology
    clim = pd.read_parquet(WEATHER_SIGMA_SEASON_BINS_PARQUET)
    print(f"- Loaded weather_sigma_season_bins: {len(clim)} rows")


    print("- Adding forecasts to TRAIN panel")
    panel_train_fc = add_weather_forecasts(panel_train, clim)

    print("- Adding forecasts to VALIDATION panel")
    panel_validation_fc = add_weather_forecasts(panel_validation, clim)

    print("- Adding forecasts to TEST panel")
    panel_test_fc = add_weather_forecasts(panel_test, clim)

    # 5. Shrink dtypes & save
    #panel_train_fc = cast_panel_dtypes(panel_train_fc)
    #panel_test_fc = cast_panel_dtypes(panel_test_fc)

    panel_train_fc.to_parquet(PANEL_WEATHER_TRAIN_PARQUET, index=False)
    panel_validation_fc.to_parquet(PANEL_WEATHER_VALIDATION_PARQUET, index=False)
    panel_test_fc.to_parquet(PANEL_WEATHER_TEST_PARQUET, index=False)

    print(f"Saved panel_weather_train to {PANEL_WEATHER_TRAIN_PARQUET}")
    print(f"Saved panel_weather_validation to {PANEL_WEATHER_VALIDATION_PARQUET}")
    print(f"Saved panel_weather_test  to {PANEL_WEATHER_TEST_PARQUET}")




if __name__ == "__main__":
    main()
