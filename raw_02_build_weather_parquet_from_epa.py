# scripts/raw_02_build_weather_parquet_from_epa.py

from pathlib import Path
from datetime import date
import requests
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------------------
# PATHS (self-contained)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
WEATHER_PARQUET = DATA_RAW / "weather_hourly.parquet"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# --------------------------------------------------------------------------------------
# FETCH HOURLY WEATHER FROM OPEN-METEO
# --------------------------------------------------------------------------------------
def fetch_open_meteo_hourly(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo Historical Weather API.

    Output schema:
        timestamp (datetime, naive local),
        temp      (°C),
        wind      (m/s),
        precip    (mm/hour),
        snow      (mm of snowfall, approx)

    Docs: https://open-meteo.com/en/docs/historical-weather-api
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,  # "yyyy-mm-dd"
        "end_date": end_date,      # "yyyy-mm-dd"
        "hourly": ",".join(
            [
                "temperature_2m",
                "windspeed_10m",
                "precipitation",  
                "snowfall",       
            ]
        ),
        "timezone": timezone,
        "timeformat": "iso8601",
        
    }

    print(f"- Requesting Open-Meteo from {start_date} to {end_date} "
          f"at lat={latitude}, lon={longitude}")

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly")
    if not hourly:
        raise RuntimeError("Open-Meteo response has no 'hourly' field")

    df = pd.DataFrame(hourly)
    if df.empty:
        raise RuntimeError("Open-Meteo returned empty hourly DataFrame")

    # Parse time to timestamp
    df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # If timezone-aware, drop tz to keep naive local timestamps like before
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    # Rename to project schema
    df.rename(
        columns={
            "temperature_2m": "temp",
            "windspeed_10m": "wind",
            "precipitation": "precip",
            "snowfall": "snow_raw",
        },
        inplace=True,
    )

    # Roughly convert snowfall to mm to stay compatible with older EPA logic.
    # Docs: snowfall is depth of new snow; often given in cm. We convert to mm.
    # If it's already mm water equivalent, this just rescales, which is OK for ML.
    df["snow"] = df["snow_raw"].astype(float) * 10.0

    # Keep the expected columns
    df = df[["timestamp", "temp", "wind", "precip", "snow"]].copy()

    # Downcast for size
    df["temp"] = pd.to_numeric(df["temp"], downcast="float")
    df["wind"] = pd.to_numeric(df["wind"], downcast="float")
    df["precip"] = pd.to_numeric(df["precip"], downcast="float")
    df["snow"] = pd.to_numeric(df["snow"], downcast="float")

    print(f"- Received {len(df)} hourly records from Open-Meteo")
    return df


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Raw Step 02: Build weather_hourly.parquet from Open-Meteo ===")
    print(f"Project root: {ROOT}")
    print(f"DATA_RAW:     {DATA_RAW}")

    # You can adjust the range if you want
    # Here: match your crash / panel period (2021-01-01 .. 2024-12-31)
    start_date = "2021-01-01"
    end_date = "2024-12-31"

    # NYC / Central Park-ish coordinates
    latitude = 40.7833
    longitude = -73.9667

    weather = fetch_open_meteo_hourly(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        timezone="America/New_York",
    )

    weather = weather.sort_values("timestamp").reset_index(drop=True)

    # Drop rows where everything except timestamp is NaN (should be rare)
    mask_all_nan = weather[["temp", "wind", "precip", "snow"]].isna().all(axis=1)
    weather = weather[~mask_all_nan].reset_index(drop=True)

    print(f"- Final hourly weather rows: {len(weather)}")

    WEATHER_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(WEATHER_PARQUET, index=False)
    print(f"Saved weather_hourly.parquet: {WEATHER_PARQUET}")


if __name__ == "__main__":
    main()
