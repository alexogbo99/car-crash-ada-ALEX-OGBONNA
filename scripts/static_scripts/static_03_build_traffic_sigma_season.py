# scripts/static_03_build_traffic_sigma_season.py

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# PATHS (self-contained)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

TRAFFIC_TIMESERIES_PARQUET = DATA_RAW / "traffic_timeseries.parquet"
TRAFFIC_SIGMA_SEASON_PARQUET = DATA_INTER / "traffic_sigma_season.parquet"
TRAFFIC_SIGMA_SEASON_BINS_PARQUET = DATA_INTER / "traffic_sigma_season_bins.parquet" 


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def assign_time_bin(hour: int) -> int:
    """
    Map hour 0–23 to the 4 x 6-hour bins used everywhere else:

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


def build_traffic_sigma_season(traffic_ts_df: pd.DataFrame, window_days: int = 15) -> pd.DataFrame:
    """
    Build DOY-based seasonal volatility for traffic (city-wide, daily).

    Steps:
      - Aggregate all sensors to daily_total_volume.
      - Remove weekly pattern (subtract mean per weekday).
      - For each DOY (1..366), compute std of residuals within +/- window_days on the circle.

    Returns:
      DataFrame with columns: doy, sigma_traffic_season
    """
    df = traffic_ts_df.copy()

    # ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "volume"])

    # aggregate to daily total volume across ALL sensors
    df["date"] = df["timestamp"].dt.date
    daily = (
        df.groupby("date")["volume"]
        .sum()
        .reset_index()
        .rename(columns={"volume": "daily_total_volume"})
    )

    # add day-of-week and DOY
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dow"] = daily["date"].dt.dayofweek  # 0=Mon
    daily["doy"] = daily["date"].dt.dayofyear

    # remove weekly pattern: residual = daily_total - mean_by_dow
    mean_by_dow = daily.groupby("dow")["daily_total_volume"].mean()
    daily["residual"] = daily.apply(
        lambda row: row["daily_total_volume"] - mean_by_dow[row["dow"]], axis=1
    )

    # Now build DOY-based std with +/- window_days smoothing on circular year
    all_doys = np.arange(1, 367)  
    sigma_list = []

    global_std = float(daily["residual"].std(ddof=1)) if len(daily) > 1 else 0.0

    for doy in all_doys:
        # circular distance in DOY space (wrap-around)
        dist = np.minimum(
            np.abs(daily["doy"] - doy),
            366 - np.abs(daily["doy"] - doy),
        )

        in_window = dist <= window_days
        residuals = daily.loc[in_window, "residual"].values

        if len(residuals) < 5:
            # Not enough data in window; fallback to global std of residuals
            sigma = global_std
        else:
            sigma = float(np.std(residuals, ddof=1))

        sigma_list.append({"doy": int(doy), "sigma_traffic_season": sigma})

    sigma_df = pd.DataFrame(sigma_list)
    return sigma_df


def build_traffic_sigma_season_bins(traffic_ts_df: pd.DataFrame, window_days: int = 15) -> pd.DataFrame:
    """
    Build DOY + time_bin seasonal volatility for traffic.

    Steps:
      - Map each timestamp to (date, time_bin).
      - Aggregate across ALL sensors to bin_total_volume per (date, time_bin).
      - Within each time_bin:
          * remove weekly pattern (mean per weekday in that bin),
          * compute residuals,
          * for each DOY, compute std of residuals within +/- window_days (circular),
            with fallback to global std in that bin if needed.

    Returns:
      DataFrame with columns: doy, time_bin, sigma_traffic_season_tb
    """
    df = traffic_ts_df.copy()

    # ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "volume"])

    # map to date + time_bin
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["time_bin"] = df["hour"].apply(assign_time_bin).astype("int8")

    # aggregate to total volume per (date, time_bin) across ALL sensors
    daily_bin = (
        df.groupby(["date", "time_bin"])["volume"]
        .sum()
        .reset_index()
        .rename(columns={"volume": "bin_total_volume"})
    )

    daily_bin["date"] = pd.to_datetime(daily_bin["date"])
    daily_bin["dow"] = daily_bin["date"].dt.dayofweek  
    daily_bin["doy"] = daily_bin["date"].dt.dayofyear

    all_doys = np.arange(1, 367)  
    all_bins = sorted(daily_bin["time_bin"].unique())
    rows = []

    for tb in all_bins:
        sub = daily_bin[daily_bin["time_bin"] == tb].copy()
        if sub.empty:
            continue

        # weekly pattern within this time_bin
        mean_by_dow_tb = sub.groupby("dow")["bin_total_volume"].mean()
        sub["residual"] = sub.apply(
            lambda row: row["bin_total_volume"] - mean_by_dow_tb[row["dow"]],
            axis=1,
        )

        global_std_tb = float(sub["residual"].std(ddof=1)) if len(sub) > 1 else 0.0

        for doy in all_doys:
            # circular distance in DOY space
            dist = np.minimum(
                np.abs(sub["doy"] - doy),
                366 - np.abs(sub["doy"] - doy),
            )
            in_window = dist <= window_days
            residuals = sub.loc[in_window, "residual"].values

            if len(residuals) < 5:
                sigma = global_std_tb
            else:
                sigma = float(np.std(residuals, ddof=1))

            rows.append(
                {
                    "doy": int(doy),
                    "time_bin": int(tb),
                    "sigma_traffic_season_tb": sigma,
                }
            )

    out = pd.DataFrame(rows).sort_values(["doy", "time_bin"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Static Step 03: Build Traffic Sigma Season (DOY + time_bin) ===")
    print(f"Project root: {ROOT}")
    print(f"DATA_RAW:     {DATA_RAW}")
    print(f"DATA_INTER:   {DATA_INTER}")

    DATA_INTER.mkdir(parents=True, exist_ok=True)

    # 1. Load traffic timeseries
    traffic_ts = pd.read_parquet(TRAFFIC_TIMESERIES_PARQUET)
    print(f"- Loaded traffic_timeseries: {len(traffic_ts)} rows")

    # 2a. DOY-only sigma (backward compatible)
    sigma_df = build_traffic_sigma_season(traffic_ts, window_days=15)
    sigma_df.to_parquet(TRAFFIC_SIGMA_SEASON_PARQUET, index=False)
    print(f"Saved DOY-based traffic_sigma_season to {TRAFFIC_SIGMA_SEASON_PARQUET}")

    # 2b. DOY + time_bin sigma (NEW)
    sigma_bins_df = build_traffic_sigma_season_bins(traffic_ts, window_days=15)

    # ----------------------------
    #addition: sigma_factor (dimensionless multiplier)
    # ----------------------------
    baseline = sigma_bins_df.groupby("time_bin")["sigma_traffic_season_tb"].median()
    sigma_bins_df["sigma_factor"] = (
        sigma_bins_df["sigma_traffic_season_tb"] / sigma_bins_df["time_bin"].map(baseline)
    ).astype("float32")

    # Safety clip: avoids extreme multipliers
    sigma_bins_df["sigma_factor"] = sigma_bins_df["sigma_factor"].clip(0.5, 2.0).astype("float32")

    sigma_bins_df.to_parquet(TRAFFIC_SIGMA_SEASON_BINS_PARQUET, index=False)
    print(f"Saved bin-level traffic_sigma_season_bins to {TRAFFIC_SIGMA_SEASON_BINS_PARQUET}")
    print("   (includes column: sigma_factor)")

if __name__ == "__main__":
    main()
