# scripts/static_04_build_weather_sigma_season.py

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# PATHS (self-contained)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

WEATHER_HOURLY_PARQUET = DATA_RAW / "weather_hourly.parquet"
WEATHER_SIGMA_SEASON_PARQUET = DATA_INTER / "weather_sigma_season.parquet"
WEATHER_SIGMA_SEASON_BINS_PARQUET = DATA_INTER / "weather_sigma_season_bins.parquet"  


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def assign_time_bin(hour: int) -> int:
    """Map hour 0–23 to the 4 x 6-hour bins used everywhere else."""
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    else:
        return 3


def build_weather_sigma_season(
    weather_df,
    vars_mean=("temp", "wind", "precip", "snow"),
    window_days=15,
):
    """
    Build DOY-only climatology (mean) and volatility (std) for weather variables.

    Assumes:
      - weather_df has columns: timestamp + variables in vars_mean

    Steps:
      - Aggregate hourly -> daily means for each variable.
      - For each DOY, compute mean and std over a +/- window_days
        window on circular year, with a fallback to global stats if
        there isn't enough data in the local window.

    Returns:
      DataFrame with columns:
        doy,
        <var>_mean, <var>_sigma for each var in vars_mean
    """
    df = weather_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    cols_needed = ["timestamp"] + list(vars_mean)
    df = df[cols_needed].copy()

    # Aggregate hourly -> daily
    df["date"] = df["timestamp"].dt.date
    # Aggregate hourly -> daily
    # - temp/wind: mean
    # - precip/snow: sum (daily accumulation)
    agg_daily = {var: ("sum" if var in ("precip", "snow") else "mean") for var in vars_mean}

    daily = (
        df.groupby("date")
        .agg(agg_daily)
        .reset_index()
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily["doy"] = daily["date"].dt.dayofyear

    all_doys = np.arange(1, 367) 
    rows = []

    for doy in all_doys:
        # circular distance in DOY space
        dist = np.minimum(
            np.abs(daily["doy"] - doy),
            366 - np.abs(daily["doy"] - doy),
        )
        in_window = dist <= window_days
        window_sub = daily.loc[in_window]

        row = {"doy": int(doy)}

        for var in vars_mean:
            vals_window = window_sub[var].dropna().values
            vals_all = daily[var].dropna().values

            if len(vals_window) >= 5:
                vals = vals_window
            elif len(vals_all) > 0:
                vals = vals_all
            else:
                row[f"{var}_mean"] = np.nan
                row[f"{var}_sigma"] = np.nan
                continue

            row[f"{var}_mean"] = float(np.mean(vals))
            if len(vals) > 1:
                row[f"{var}_sigma"] = float(np.std(vals, ddof=1))
            else:
                row[f"{var}_sigma"] = 0.0

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("doy").reset_index(drop=True)
    return out


def build_weather_sigma_season_bins(
    weather_df,
    vars_mean=("temp", "wind", "precip", "snow"),
    window_days=15,
):
    """
    Build DOY + time_bin climatology (mean) and volatility (std) for weather variables.

    Assumes:
      - weather_df has columns: timestamp + variables in vars_mean
    Steps:
      - Map each hourly obs to (date, time_bin).
      - Aggregate hourly -> per (date, time_bin) means/sums.
      - For each (time_bin, DOY), compute mean and std using +/- window_days
        on circular year with a fallback to global stats within that bin.
    Returns:
      DataFrame with columns:
        doy, time_bin,
        <var>_mean, <var>_sigma  for each var in vars_mean
    """
    df = weather_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    cols_needed = ["timestamp"] + list(vars_mean)
    df = df[cols_needed].copy()

    # Map to (date, time_bin)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["time_bin"] = df["hour"].apply(assign_time_bin).astype("int8")

    # Aggregate hourly -> (date, time_bin)
    # - temp/wind: mean over the bin
    # - precip/snow: sum over the bin (bin accumulation)
    agg_bin = {var: ("sum" if var in ("precip", "snow") else "mean") for var in vars_mean}

    daily_bin = (
        df.groupby(["date", "time_bin"])
        .agg(agg_bin)
        .reset_index()
    )

    daily_bin["date"] = pd.to_datetime(daily_bin["date"])
    daily_bin["doy"] = daily_bin["date"].dt.dayofyear

    all_doys = np.arange(1, 367)  # 1..366
    all_bins = sorted(daily_bin["time_bin"].unique())
    rows = []

    for tb in all_bins:
        sub_bin = daily_bin[daily_bin["time_bin"] == tb].copy()
        if sub_bin.empty:
            continue

        for doy in all_doys:
            # circular distance in DOY space
            dist = np.minimum(
                np.abs(sub_bin["doy"] - doy),
                366 - np.abs(sub_bin["doy"] - doy),
            )
            in_window = dist <= window_days
            window_sub = sub_bin.loc[in_window]

            row = {"doy": int(doy), "time_bin": int(tb)}

            for var in vars_mean:
                # values in the window
                vals_window = window_sub[var].dropna().values
                # global within this time_bin as fallback
                vals_all = sub_bin[var].dropna().values

                if len(vals_window) >= 5:
                    vals = vals_window
                elif len(vals_all) > 0:
                    # not enough in window: fallback to global per-bin stats
                    vals = vals_all
                else:
                    # absolutely no data for this var in this bin
                    row[f"{var}_mean"] = np.nan
                    row[f"{var}_sigma"] = np.nan
                    continue

                row[f"{var}_mean"] = float(np.mean(vals))
                # std with ddof=1 but handle len(vals)==1
                if len(vals) > 1:
                    row[f"{var}_sigma"] = float(np.std(vals, ddof=1))
                else:
                    row[f"{var}_sigma"] = 0.0

            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["doy", "time_bin"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Static Step 04: Build Weather Sigma Season (DOY + time_bin) ===")
    print(f"Project root: {ROOT}")
    print(f"DATA_RAW:     {DATA_RAW}")
    print(f"DATA_INTER:   {DATA_INTER}")

    DATA_INTER.mkdir(parents=True, exist_ok=True)

    # 1. Load weather_hourly
    weather = pd.read_parquet(WEATHER_HOURLY_PARQUET)
    print(f"- Loaded weather_hourly: {len(weather)} rows")

    # 2a. DOY-only climatology & sigma (backward compatible)
    sigma_df = build_weather_sigma_season(
        weather_df=weather,
        vars_mean=("temp", "wind", "precip", "snow"),
        window_days=15,
    )
    sigma_df.to_parquet(WEATHER_SIGMA_SEASON_PARQUET, index=False)
    print(f"Saved DOY-based weather_sigma_season to {WEATHER_SIGMA_SEASON_PARQUET}")

    # 2b. DOY + time_bin climatology & sigma (NEW)
    sigma_bins_df = build_weather_sigma_season_bins(
        weather_df=weather,
        vars_mean=("temp", "wind", "precip", "snow"),
        window_days=15,
    )
    sigma_bins_df.to_parquet(WEATHER_SIGMA_SEASON_BINS_PARQUET, index=False)
    print(f"Saved bin-level weather_sigma_season_bins to {WEATHER_SIGMA_SEASON_BINS_PARQUET}")


if __name__ == "__main__":
    main()
