# scripts/dynamic_13_add_traffic_pseudofc.py

from pathlib import Path
import gc
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# PATHS & CONFIG
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

PANEL_WEATHER_TRAIN_PARQUET = DATA_INTER / "panel_weather_train.parquet"
PANEL_WEATHER_VALIDATION_PARQUET = DATA_INTER / "panel_weather_validation.parquet"
PANEL_WEATHER_TEST_PARQUET = DATA_INTER / "panel_weather_test.parquet"

#basic sigma (DOY) + sigma_factor
TRAFFIC_SIGMA_SEASON_PARQUET = DATA_INTER / "traffic_sigma_season.parquet"

# bin-level sigma (DOY + time_bin) + sigma_factor
TRAFFIC_SIGMA_SEASON_BINS_PARQUET = DATA_INTER / "traffic_sigma_season_bins.parquet"

# time-of-day profile file (already used)
TRAFFIC_TOD_PROFILE_PARQUET = DATA_INTER / "traffic_tod_profile.parquet"
TRAFFIC_TIMESERIES_PARQUET = DATA_RAW / "traffic_timeseries.parquet"

PANEL_WEATHER_TRAFFIC_TRAIN_PARQUET = DATA_INTER / "panel_weather_traffic_train.parquet"
PANEL_WEATHER_TRAFFIC_VALIDATION_PARQUET = DATA_INTER / "panel_weather_traffic_validation.parquet"
PANEL_WEATHER_TRAFFIC_TEST_PARQUET = DATA_INTER / "panel_weather_traffic_test.parquet"

HORIZONS = list(range(1, 8))
Z_05 = 1.6448536269514722  # Normal quantile for 5% / 95%

TRAFFIC_SIGMA_SCALE = {
    1: 0.3,
    2: 0.45,
    3: 0.55,
    4: 0.65,
    5: 0.72,
    6: 0.78,
    7: 0.82,
}

# clip range for multiplier
SIGMA_FACTOR_CLIP_LOW = 0.5
SIGMA_FACTOR_CLIP_HIGH = 2.0


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def assign_time_bin(hour: int) -> int:
    """
    Map hour 0..23 to a 6-hour time bin, consistent with other scripts:

        0: 00:00–05:59
        1: 06:00–11:59
        2: 12:00–17:59
        3: 18:00–23:59
    """
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    else:
        return 3


def load_tod_profile() -> pd.DataFrame:
    """
    Load global time-of-day traffic profile (share of daily volume per time_bin).

    If traffic_tod_profile.parquet exists in DATA_INTER, use it.
    Otherwise compute from traffic_timeseries.parquet and save it.
    Returns a DataFrame with columns: time_bin (int8), tod_share (float32).
    """
    if TRAFFIC_TOD_PROFILE_PARQUET.exists():
        tod = pd.read_parquet(TRAFFIC_TOD_PROFILE_PARQUET)
        print(f"- Loaded traffic_tod_profile: {len(tod)} bins")
        # Ensure expected dtypes
        if "time_bin" in tod.columns:
            tod["time_bin"] = tod["time_bin"].astype("int8")
        if "tod_share" in tod.columns:
            tod["tod_share"] = tod["tod_share"].astype("float32")
        return tod

    print("- traffic_tod_profile.parquet not found; computing global time-of-day profile from traffic_timeseries")

    traffic_ts = pd.read_parquet(TRAFFIC_TIMESERIES_PARQUET)
    print(f"  -> Loaded traffic_timeseries: {len(traffic_ts)} rows")

    # Basic cleaning
    if "volume" not in traffic_ts.columns:
        raise ValueError("traffic_timeseries.parquet must contain a 'volume' column.")
    if "timestamp" not in traffic_ts.columns:
        raise ValueError("traffic_timeseries.parquet must contain a 'timestamp' column.")

    traffic_ts = traffic_ts.dropna(subset=["timestamp", "volume"]).copy()
    traffic_ts["timestamp"] = pd.to_datetime(traffic_ts["timestamp"], errors="coerce")
    traffic_ts = traffic_ts.dropna(subset=["timestamp"]).copy()

    traffic_ts["date"] = traffic_ts["timestamp"].dt.floor("D")
    traffic_ts["hour"] = traffic_ts["timestamp"].dt.hour.astype("int16")
    traffic_ts["time_bin"] = traffic_ts["hour"].apply(assign_time_bin).astype("int8")

    # Aggregate to sensor x date x time_bin
    daily_bin = (
        traffic_ts.groupby(["sensor_id", "date", "time_bin"], as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "bin_volume"})
    )

    # Daily totals per sensor x date
    daily_total = (
        daily_bin.groupby(["sensor_id", "date"], as_index=False)["bin_volume"]
        .sum()
        .rename(columns={"bin_volume": "daily_total"})
    )

    daily = daily_bin.merge(daily_total, on=["sensor_id", "date"], how="left")
    daily = daily[daily["daily_total"] > 0].copy()
    daily["share"] = daily["bin_volume"] / daily["daily_total"]

    # Global mean share per time_bin across all sensors and days
    tod = (
        daily.groupby("time_bin")["share"]
        .mean()
        .reset_index()
        .rename(columns={"share": "tod_share"})
    )

    total_share = tod["tod_share"].sum()
    if total_share <= 0 or np.isnan(total_share):
        # Fallback: uniform split if something went wrong
        print("  ! Warning: tod_share sum <= 0; falling back to uniform shares")
        n_bins = len(tod)
        tod["tod_share"] = 1.0 / max(n_bins, 1)
    else:
        tod["tod_share"] = tod["tod_share"] / total_share

    tod["time_bin"] = tod["time_bin"].astype("int8")
    tod["tod_share"] = tod["tod_share"].astype("float32")

    # Save for reuse
    TRAFFIC_TOD_PROFILE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tod.to_parquet(TRAFFIC_TOD_PROFILE_PARQUET, index=False)
    print(f"  -> Saved traffic_tod_profile to {TRAFFIC_TOD_PROFILE_PARQUET}")

    return tod


def add_traffic_forecasts(
    panel: pd.DataFrame,
    traffic_season_bins: pd.DataFrame,
    tod_profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add time-bin-specific traffic pseudo-forecasts:

        fc_traffic_mean_h{h}d, fc_traffic_q05_h{h}d, fc_traffic_q95_h{h}d
        for h in 1..7.

    Option 1 logic (stable scaling):

    - Mean in bin:
        mean_h = vol_static_tb = vol_static * tod_share

    - Base uncertainty at the correct scale (cell-level):
        local_uncertainty (already interpolated to cells)

    - Seasonal multiplier from city climatology (dimensionless):
        sigma_factor(doy, time_bin)

    - For each horizon h:
        future_doy = (date + h days).dayofyear
        factor     = sigma_factor(future_doy, time_bin)

        sigma_h    = local_uncertainty * factor * TRAFFIC_SIGMA_SCALE[h]

        q05        = mean_h - Z_05 * sigma_h
        q95        = mean_h + Z_05 * sigma_h
    """
    panel = panel.copy()

    required_cols = ["vol_static", "local_traffic_uncertainty", "time_bin", "date"]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel must contain columns {required_cols}, missing: {missing}")

    # Map time_bin -> tod_share
    tod_map = tod_profile.set_index("time_bin")["tod_share"]
    panel["tod_share"] = panel["time_bin"].map(tod_map).astype("float32")

    # Fallback: if some time_bins are missing in tod_profile, use uniform share
    if panel["tod_share"].isna().any():
        n_bins = tod_profile["time_bin"].nunique()
        fallback_share = np.float32(1.0 / max(n_bins, 1))
        panel["tod_share"] = panel["tod_share"].fillna(fallback_share)

    # Time-bin traffic baselines
    panel["vol_static"] = panel["vol_static"].astype("float32")
    panel["vol_static_tb"] = (panel["vol_static"] * panel["tod_share"]).astype("float32")

    if "max_traffic_volume" in panel.columns:
        panel["max_traffic_volume"] = panel["max_traffic_volume"].astype("float32")
        panel["max_traffic_volume_tb"] = (
            panel["max_traffic_volume"] * panel["tod_share"]
        ).astype("float32")

    # Ensure traffic_season_bins has sigma_factor; compute if missing (backward-compatible)
    if "sigma_factor" not in traffic_season_bins.columns:
        if "sigma_traffic_season_tb" not in traffic_season_bins.columns:
            raise ValueError(
                "traffic_sigma_season_bins.parquet must contain 'sigma_traffic_season_tb' "
                "(and ideally 'sigma_factor')."
            )

        baseline = traffic_season_bins.groupby("time_bin")["sigma_traffic_season_tb"].median()
        traffic_season_bins = traffic_season_bins.copy()
        traffic_season_bins["sigma_factor"] = (
            traffic_season_bins["sigma_traffic_season_tb"] / traffic_season_bins["time_bin"].map(baseline)
        ).astype("float32")
        traffic_season_bins["sigma_factor"] = traffic_season_bins["sigma_factor"].clip(
            SIGMA_FACTOR_CLIP_LOW, SIGMA_FACTOR_CLIP_HIGH
        ).astype("float32")

    # Build index: (doy, time_bin) -> sigma_factor
    factor_idx = traffic_season_bins.set_index(["doy", "time_bin"])["sigma_factor"]

    base_mean = panel["vol_static_tb"].astype("float32")
    local_unc = panel["local_traffic_uncertainty"].astype("float32")
    dates = pd.to_datetime(panel["date"], errors="coerce")
    time_bins = panel["time_bin"].astype("int8")

    for h in HORIZONS:
        print(f"  - Adding traffic forecasts for h={h} day(s) ahead")

        future_dates = dates + pd.to_timedelta(h, unit="D")
        future_doy = future_dates.dt.dayofyear.astype("int16")

        # MultiIndex lookup key
        key = pd.MultiIndex.from_arrays(
            [future_doy.values, time_bins.values],
            names=["doy", "time_bin"],
        )

        factor = factor_idx.reindex(key).to_numpy(dtype="float32")

        # Missing -> neutral multiplier
        factor = np.nan_to_num(factor, nan=1.0)

        # Horizon-specific sigma (multiplicative; stays on cell scale)
        sigma_h = (local_unc * factor) * np.float32(TRAFFIC_SIGMA_SCALE[h])

        mean_h = base_mean  # same mean across horizons

        q05_h = mean_h - np.float32(Z_05) * sigma_h
        q95_h = mean_h + np.float32(Z_05) * sigma_h

        # clip to feasible non-negative volumes
        q05_h = np.clip(q05_h, 0.0, None).astype("float32")
        q95_h = np.clip(q95_h, 0.0, None).astype("float32")

        panel[f"fc_traffic_mean_h{h}d"] = mean_h.astype("float32")
        panel[f"fc_traffic_q05_h{h}d"] = q05_h
        panel[f"fc_traffic_q95_h{h}d"] = q95_h

    return panel


def cast_panel_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast dtypes WITHOUT copying the entire dataframe (avoids pandas consolidation OOM).
    Operates column-wise in place.
    """
    # time_bin should be tiny
    if "time_bin" in df.columns and df["time_bin"].dtype != "int8":
        df["time_bin"] = df["time_bin"].astype("int8", copy=False)

    # Make sure key traffic cols are float32
    for col in ["vol_static", "local_traffic_uncertainty", "tod_share", "vol_static_tb", "max_traffic_volume_tb"]:
        if col in df.columns and df[col].dtype != "float32":
            df[col] = df[col].astype("float32", copy=False)

    # Forecast cols: cast only those that exist, column-wise
    prefixes = (
        "temp_", "wind_", "precip_", "snow_",
        "fc_temp_", "fc_wind_", "fc_precip_", "fc_snow_",
        "fc_traffic_",
    )

    for col in df.columns:
        if col.startswith(prefixes) and df[col].dtype != "float32":
            df[col] = df[col].astype("float32", copy=False)

    return df



# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Dynamic Step 13: Add Traffic Pseudo-Forecasts (h=1..7, sigma_factor multiplier) ===")
    print(f"ROOT:       {ROOT}")
    print(f"DATA_RAW:   {DATA_RAW}")
    print(f"DATA_INTER: {DATA_INTER}")

    # 1. Load panels with weather
    panel_train = pd.read_parquet(PANEL_WEATHER_TRAIN_PARQUET)
    panel_validation = pd.read_parquet(PANEL_WEATHER_VALIDATION_PARQUET)
    panel_test = pd.read_parquet(PANEL_WEATHER_TEST_PARQUET)
    print(f"- Loaded panel_weather_train: {len(panel_train)} rows")
    print(f"- Loaded panel_weather_validation: {len(panel_validation)} rows")
    print(f"- Loaded panel_weather_test:  {len(panel_test)} rows")

    # 2. Load bin-level traffic sigma (includes sigma_factor after updated static_03)
    traffic_season_bins = pd.read_parquet(TRAFFIC_SIGMA_SEASON_BINS_PARQUET)
    print(f"- Loaded traffic_sigma_season_bins: {len(traffic_season_bins)} rows")

    # 3. Load / compute global time-of-day traffic profile
    tod_profile = load_tod_profile()

    # 4. Add traffic forecasts
    print("- Adding traffic forecasts to TRAIN panel")
    panel_train_fc = add_traffic_forecasts(panel_train, traffic_season_bins, tod_profile)

    print("- Adding traffic forecasts to VALIDATION panel")
    panel_validation_fc = add_traffic_forecasts(panel_validation, traffic_season_bins, tod_profile)

    print("- Adding traffic forecasts to TEST panel")
    panel_test_fc = add_traffic_forecasts(panel_test, traffic_season_bins, tod_profile)

    # 5. TRAIN: cast, save, free
    print("- Casting dtypes & saving TRAIN panel")
    panel_train_fc = cast_panel_dtypes(panel_train_fc)
    panel_train_fc.to_parquet(PANEL_WEATHER_TRAFFIC_TRAIN_PARQUET, index=False)

    del panel_train_fc, panel_train
    gc.collect()

    # 6. VALIDATION: cast, save, free
    print("- Casting dtypes & saving VALIDATION panel")
    panel_validation_fc = cast_panel_dtypes(panel_validation_fc)
    panel_validation_fc.to_parquet(PANEL_WEATHER_TRAFFIC_VALIDATION_PARQUET, index=False)

    del panel_validation_fc, panel_validation
    gc.collect()

    # 7. TEST: cast, save, free
    print("- Casting dtypes & saving TEST panel")
    panel_test_fc = cast_panel_dtypes(panel_test_fc)
    panel_test_fc.to_parquet(PANEL_WEATHER_TRAFFIC_TEST_PARQUET, index=False)

    del panel_test_fc, panel_test
    gc.collect()

    print("Done. Saved traffic-forecast-enriched panels.")


if __name__ == "__main__":
    main()
