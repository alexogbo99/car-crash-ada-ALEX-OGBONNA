# scripts/static_02_build_static_traffic_and_neighbors.py

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree 


# --------------------------------------------------------------------------------------
# PATHS (relative)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

STATIC_GRID_PARQUET = DATA_INTER / "static_grid.parquet"
STATIC_FEATURES_PARQUET = DATA_INTER / "static_features.parquet"

TRAFFIC_TIMESERIES_PARQUET = DATA_RAW / "traffic_timeseries.parquet"
TRAFFIC_SENSORS_PARQUET = DATA_RAW / "traffic_sensors.parquet"


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def compute_sensor_traffic_stats(traffic_ts_df, volume_col="volume", sensor_id_col="sensor_id"):
    """
    Compute mean, max, std volume per sensor over the whole history.

    Returns DataFrame with columns:
        sensor_id, mean_vol, max_vol, std_vol
    """
    stats = (
        traffic_ts_df
        .groupby(sensor_id_col)[volume_col]
        .agg(["mean", "max", "std"])
        .reset_index()
    )
    stats = stats.rename(
        columns={
            sensor_id_col: "sensor_id",
            "mean": "mean_vol",
            "max": "max_vol",
            "std": "std_vol",
        }
    )
    # Fallback: if std is NaN (single reading), use 0.2 * mean
    stats["std_vol"] = stats["std_vol"].fillna(stats["mean_vol"] * 0.2)
    return stats


def interpolate_static_traffic_to_grid(grid_gdf, sensors_gdf, sensor_stats_df, k=3):
    """
    Interpolate static traffic stats from sensors to grid cells with KNN (k=3)
    in the same CRS as grid_gdf (metric).

    Expects:
        - grid_gdf: GeoDataFrame with cell_id, geometry (CRS metric)
        - sensors_gdf: GeoDataFrame with (sensor_id, geometry) in any CRS
        - sensor_stats_df: DataFrame with sensor_id, mean_vol, max_vol, std_vol

    Returns DataFrame with:
        cell_id, avg_traffic_volume, max_traffic_volume, local_traffic_uncertainty
    """
    # Project sensors to grid CRS
    sensors_proj = sensors_gdf.to_crs(grid_gdf.crs)

    # Merge stats onto sensors
    sensors_merged = sensors_proj.merge(sensor_stats_df, on="sensor_id", how="inner")

    # --- CLEAN BAD GEOMETRIES ---
    sensors_merged = sensors_merged[sensors_merged.geometry.notna()].copy()
    sensors_merged = sensors_merged[~sensors_merged.geometry.is_empty].copy()

    # Extract coordinates
    sensor_x = sensors_merged.geometry.x.to_numpy()
    sensor_y = sensors_merged.geometry.y.to_numpy()

    # Remove NaN / inf coordinates
    finite_mask = np.isfinite(sensor_x) & np.isfinite(sensor_y)
    sensors_merged = sensors_merged[finite_mask].copy()
    sensor_x = sensor_x[finite_mask]
    sensor_y = sensor_y[finite_mask]

    if len(sensors_merged) == 0:
        raise ValueError("No valid sensor geometries left after cleaning.")

    # Build KD-tree in x,y
    sensor_xy = np.vstack([sensor_x, sensor_y]).T

    # If there are fewer sensors than k, reduce k
    k_eff = min(k, sensor_xy.shape[0])

    tree = cKDTree(sensor_xy)

    # Grid centroids
    grid_centroids = grid_gdf.geometry.centroid
    grid_x = grid_centroids.x.to_numpy()
    grid_y = grid_centroids.y.to_numpy()
    grid_xy = np.vstack([grid_x, grid_y]).T

    dists, idxs = tree.query(grid_xy, k=k_eff)
    if k_eff == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    # Avoid divide-by-zero
    dists = np.where(dists == 0, 1e-6, dists)
    weights = 1.0 / dists
    weights_norm = weights / weights.sum(axis=1, keepdims=True)

    mean_vals = sensors_merged["mean_vol"].values[idxs]
    max_vals = sensors_merged["max_vol"].values[idxs]
    std_vals = sensors_merged["std_vol"].values[idxs]

    avg_traffic = np.sum(mean_vals * weights_norm, axis=1)
    max_traffic = np.sum(max_vals * weights_norm, axis=1)
    local_unc = np.sum(std_vals * weights_norm, axis=1)

    df = pd.DataFrame(
        {
            "cell_id": grid_gdf["cell_id"].values,
            "avg_traffic_volume": avg_traffic.astype("float32"),
            "max_traffic_volume": max_traffic.astype("float32"),
            "local_traffic_uncertainty": local_unc.astype("float32"),
        }
    )
    return df


def build_neighbor_map(grid_gdf, buffer_meters=200):
    """
    Build neighbor list per cell_id using a spatial buffer.

    Returns DataFrame:
        cell_id, neighbors   (neighbors as list of ints)
    """
    grid_buf = grid_gdf.copy()
    grid_buf["geometry"] = grid_buf.geometry.buffer(buffer_meters)

    joined = gpd.sjoin(
        grid_buf[["cell_id", "geometry"]],
        grid_gdf[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="_left",
        rsuffix="_right",
    )

    # Figure out which columns are the "left" and "right" cell IDs.
    cols = joined.columns.tolist()

    # Heuristics for different geopandas versions / suffix behavior
    if "cell_id_left" in cols and "cell_id_right" in cols:
        left_col = "cell_id_left"
        right_col = "cell_id_right"
    elif "cell_id" in cols and "cell_id_right" in cols:
        # left kept as "cell_id", right got suffixed
        left_col = "cell_id"
        right_col = "cell_id_right"
    elif "cell_id_left" in cols and "cell_id" in cols:
        # left got suffixed, right stayed as "cell_id"
        left_col = "cell_id_left"
        right_col = "cell_id"
    else:
        # fallback: pick any two columns that contain "cell_id"
        cell_cols = [c for c in cols if "cell_id" in c]
        if len(cell_cols) < 2:
            raise ValueError(f"Could not identify cell_id columns in sjoin result. Columns: {cols}")
        left_col, right_col = cell_cols[0], cell_cols[1]

    # Exclude self-neighbors
    joined = joined[joined[left_col] != joined[right_col]]

    neighbors_series = (
        joined.groupby(left_col)[right_col]
        .apply(lambda x: list(sorted(set(x))))
        .rename("neighbors")
        .reset_index()
        .rename(columns={left_col: "cell_id"})
    )

    return neighbors_series


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Static Step 02: Build Static Traffic & Neighbors ===")
    print(f"Project root: {ROOT}")
    print(f"DATA_RAW:     {DATA_RAW}")
    print(f"DATA_INTER:   {DATA_INTER}")

    DATA_INTER.mkdir(parents=True, exist_ok=True)

    # 1. Load active grid
    grid = gpd.read_parquet(STATIC_GRID_PARQUET)
    print(f"- Loaded active grid: {len(grid)} cells")

    # 2. Load traffic timeseries & sensors
    traffic_ts = pd.read_parquet(TRAFFIC_TIMESERIES_PARQUET)
    sensors = gpd.read_parquet(TRAFFIC_SENSORS_PARQUET)
    print(f"- Traffic records: {len(traffic_ts)}")
    print(f"- Sensors: {len(sensors)}")

    # 3. Compute per-sensor traffic stats
    sensor_stats = compute_sensor_traffic_stats(traffic_ts)
    print(f"- Sensor stats computed for {len(sensor_stats)} sensors")

    # 4. Interpolate static traffic to grid
    traffic_static = interpolate_static_traffic_to_grid(
        grid_gdf=grid,
        sensors_gdf=sensors,
        sensor_stats_df=sensor_stats,
        k=3,
    )
    print("- Interpolated static traffic to grid")

    # 5. Build neighbor map
    neighbors_df = build_neighbor_map(grid, buffer_meters=200)
    print("- Neighbor map built")

    # 6. Assemble static_features
    static_features = grid[["cell_id"]].merge(traffic_static, on="cell_id", how="left")
    static_features = static_features.merge(neighbors_df, on="cell_id", how="left")

    # vol_static instead of avg_traffic_volume
    static_features = static_features.rename(
        columns={"avg_traffic_volume": "vol_static"}
    )

    # Fill missing
    for col in ["vol_static", "max_traffic_volume", "local_traffic_uncertainty"]:
        static_features[col] = static_features[col].fillna(0.0).astype("float32")

    static_features.to_parquet(STATIC_FEATURES_PARQUET, index=False)
    print(f"Saved static_features to {STATIC_FEATURES_PARQUET} (rows: {len(static_features)})")


if __name__ == "__main__":
    main()
