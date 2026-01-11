# scripts/raw_01_build_core_parquets.py

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkt as wkt


# --------------------------------------------------------------------------------------
# PATHS - all relative to this script
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]        
DATA_RAW = ROOT / "data" / "raw"

CRASHES_CSV = DATA_RAW / "Motor_Vehicle_Collisions_-_Crashes_20251129.csv"
TRAFFIC_CSV = DATA_RAW / "Automated_Traffic_Volume_Counts_20251201.csv"

CRASHES_PARQUET = DATA_RAW / "crashes.parquet"
TRAFFIC_SENSORS_PARQUET = DATA_RAW / "traffic_sensors.parquet"
TRAFFIC_TIMESERIES_PARQUET = DATA_RAW / "traffic_timeseries.parquet"


# --------------------------------------------------------------------------------------
# CRASHES
# --------------------------------------------------------------------------------------
def build_crashes_parquet():
    print("=== Building crashes.parquet from CSV ===")
    print(f"- Reading: {CRASHES_CSV}")

    df = pd.read_csv(CRASHES_CSV, low_memory=False)

    # Keep only rows with coordinates
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()

    # Parse datetime (CRASH DATE is like '09/11/2021', CRASH TIME like '2:39')
    df["CRASH_DATETIME"] = pd.to_datetime(
        df["CRASH DATE"] + " " + df["CRASH TIME"],
        errors="coerce",
        format="%m/%d/%Y %H:%M",
    )
    df = df.dropna(subset=["CRASH_DATETIME"]).copy()

    # ------------------------------------------------------------------
    # Build a robust 'n_persons_killed' from components + total
    # ------------------------------------------------------------------
    # Components are already int64 in your CSV, but we coerce to be safe.
    df["n_ped_killed"] = pd.to_numeric(
        df["NUMBER OF PEDESTRIANS KILLED"], errors="coerce"
    ).fillna(0).astype("int16")

    df["n_cyclist_killed"] = pd.to_numeric(
        df["NUMBER OF CYCLIST KILLED"], errors="coerce"
    ).fillna(0).astype("int16")

    df["n_motorist_killed"] = pd.to_numeric(
        df["NUMBER OF MOTORIST KILLED"], errors="coerce"
    ).fillna(0).astype("int16")

    df["sum_sub_killed"] = (
        df["n_ped_killed"] + df["n_cyclist_killed"] + df["n_motorist_killed"]
    )

    # Raw total from the CSV
    df["persons_killed_raw"] = pd.to_numeric(
        df["NUMBER OF PERSONS KILLED"], errors="coerce"
    ).fillna(0).astype("int16")

    # Our canonical death count:
    # - if they filled the subtype fields, use the sum
    # - otherwise fall back to the total
    df["n_persons_killed"] = np.where(
        df["sum_sub_killed"] > 0,
        df["sum_sub_killed"],
        df["persons_killed_raw"],
    ).astype("int16")

    # Injured: we keep the original total as-is (could also be rebuilt from components later)
    df["n_persons_injured"] = pd.to_numeric(
        df["NUMBER OF PERSONS INJURED"], errors="coerce"
    ).fillna(0).astype("int16")

    # ------------------------------------------------------------------
    # Final crashes DataFrame for the pipeline
    # ------------------------------------------------------------------
    crashes = df[
        [
            "CRASH_DATETIME",
            "LATITUDE",
            "LONGITUDE",
            "BOROUGH",
            "COLLISION_ID",
            "n_persons_injured",
            "n_persons_killed",
        ]
    ].copy()

    crashes = crashes.rename(
        columns={
            "CRASH_DATETIME": "crash_datetime",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "BOROUGH": "borough",
            "COLLISION_ID": "collision_id",
        }
    )

    
    crashes["collision_id"] = crashes["collision_id"].astype("int32")
    crashes["latitude"] = crashes["latitude"].astype("float32")
    crashes["longitude"] = crashes["longitude"].astype("float32")

    crashes.to_parquet(CRASHES_PARQUET, index=False)
    print(f"Saved crashes.parquet: {CRASHES_PARQUET} (rows: {len(crashes)})")


# --------------------------------------------------------------------------------------
# TRAFFIC
# --------------------------------------------------------------------------------------
def build_traffic_parquets():
    print("=== Building traffic_sensors.parquet and traffic_timeseries.parquet from CSV ===")
    print(f"- Reading: {TRAFFIC_CSV}")

    df = pd.read_csv(TRAFFIC_CSV, low_memory=False)

    # Convert Vol to numeric
    df["Vol"] = pd.to_numeric(df["Vol"], errors="coerce")
    df = df.dropna(subset=["Vol", "WktGeom", "Yr", "M", "D", "HH", "MM"]).copy()

    # Timestamp from Yr/M/D HH:MM
    df["timestamp"] = pd.to_datetime(
        dict(year=df["Yr"], month=df["M"], day=df["D"], hour=df["HH"], minute=df["MM"]),
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"]).copy()

    # Geometry from WKT
    df["geometry"] = df["WktGeom"].apply(wkt.loads)

    # IMPORTANT: Automated Traffic Volume Counts WktGeom is in a local projected CRS
    # For NYC, it's typically EPSG:2263 (NAD83 / New York Long Island ft).
    # So we:
    #   1) Create GeoDataFrame in EPSG:2263
    #   2) Reproject to WGS84 (EPSG:4326) for storage

    gdf_local = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2263")
    gdf = gdf_local.to_crs("EPSG:4326")


    # Define a sensor_id: SegmentID + Direction
    gdf["sensor_id"] = gdf["SegmentID"].astype(str) + "_" + gdf["Direction"].astype(str)

    # Timeseries
    traffic_ts = gdf[["sensor_id", "timestamp", "Vol"]].copy()
    traffic_ts = traffic_ts.rename(columns={"Vol": "volume"})

    # Sensors: one geometry per sensor_id
    sensors = (
        gdf[["sensor_id", "geometry"]]
        .drop_duplicates(subset=["sensor_id"])
        .reset_index(drop=True)
    )

    sensors.to_parquet(TRAFFIC_SENSORS_PARQUET, index=False)
    traffic_ts.to_parquet(TRAFFIC_TIMESERIES_PARQUET, index=False)

    print(f"Saved traffic_sensors.parquet: {TRAFFIC_SENSORS_PARQUET} (rows: {len(sensors)})")
    print(f"Saved traffic_timeseries.parquet: {TRAFFIC_TIMESERIES_PARQUET} (rows: {len(traffic_ts)})")


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print(f"Project root: {ROOT}")
    print(f"DATA_RAW:     {DATA_RAW}")
    build_crashes_parquet()
    build_traffic_parquets()


if __name__ == "__main__":
    main()
