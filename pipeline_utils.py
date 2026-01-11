# pipeline_utils.py
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import box, Point
from scipy.spatial import cKDTree  

from config import NYC_BOUNDS, GRID_RES_DEG

def build_full_grid(nyc_bounds=None, grid_res_deg=None, out_crs="EPSG:32618"):
    """
    Build a rectangular grid over NYC bounds with resolution grid_res_deg (in degrees),
    return a GeoDataFrame in a projected CRS (default: EPSG:32618).

    Columns:
        - cell_id (int)
        - geometry (Polygon) in out_crs
    """
    if nyc_bounds is None:
        nyc_bounds = NYC_BOUNDS
    if grid_res_deg is None:
        grid_res_deg = GRID_RES_DEG

    minx = nyc_bounds["minx"]
    maxx = nyc_bounds["maxx"]
    miny = nyc_bounds["miny"]
    maxy = nyc_bounds["maxy"]

    xs = np.arange(minx, maxx, grid_res_deg)
    ys = np.arange(miny, maxy, grid_res_deg)

    polys = []
    ids = []
    cid = 0
    for x in xs:
        for y in ys:
            polys.append(box(x, y, x + grid_res_deg, y + grid_res_deg))
            ids.append(cid)
            cid += 1

    grid = gpd.GeoDataFrame(
        {"cell_id": np.array(ids, dtype="int32"), "geometry": polys},
        crs="EPSG:4326",
    )

    # Reproject to metric CRS for distances etc.
    grid_m = grid.to_crs(out_crs)
    return grid_m


def get_active_cell_ids_from_crashes(crashes_df, grid_gdf,
                                     lon_col="longitude",
                                     lat_col="latitude",
                                     crashes_crs="EPSG:4326"):
    """
    Spatially join crashes to grid cells, return unique active cell_ids.

    Assumes crashes_df has longitude & latitude columns in crashes_crs.
    """
    crashes_df = crashes_df.dropna(subset=[lon_col, lat_col]).copy()

    gdf_crash = gpd.GeoDataFrame(
        crashes_df,
        geometry=gpd.points_from_xy(crashes_df[lon_col], crashes_df[lat_col]),
        crs=crashes_crs,
    )

    gdf_crash = gdf_crash.to_crs(grid_gdf.crs)

    joined = gpd.sjoin(
        gdf_crash,
        grid_gdf[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    active_ids = joined["cell_id"].unique()
    return set(active_ids)


def get_active_cell_ids_from_sensors(sensors_gdf, grid_gdf):
    """
    Spatially join traffic sensors to grid cells, return unique active cell_ids.

    Assumes sensors_gdf is a GeoDataFrame with a valid CRS.
    """
    sensors_proj = sensors_gdf.to_crs(grid_gdf.crs)
    joined = gpd.sjoin(
        sensors_proj,
        grid_gdf[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    active_ids = joined["cell_id"].unique()
    return set(active_ids)


def expand_active_cells_with_neighbors(grid_gdf, active_cell_ids, buffer_meters=200):
    """
    Given a grid and a set of active_cell_ids, expand to include neighbor cells
    whose polygons fall within buffer_meters around ANY active cell.

    Returns a set of cell_ids (active ∪ neighbors).
    """
    active_cells = grid_gdf[grid_gdf["cell_id"].isin(active_cell_ids)].copy()
    if active_cells.empty:
        return set()

    active_buffer = active_cells.copy()
    active_buffer["geometry"] = active_buffer.geometry.buffer(buffer_meters)

    joined = gpd.sjoin(
        grid_gdf[["cell_id", "geometry"]],
        active_buffer[["geometry"]],
        how="inner",
        predicate="intersects",
    )

    neighbor_ids = joined["cell_id"].unique()
    return set(neighbor_ids)



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


def interpolate_static_traffic_to_grid(
    grid_gdf,
    sensors_gdf,
    sensor_stats_df,
    k=3,
):
    """
    Interpolate static traffic stats from sensors to grid cells with KNN (k=3)
    in the same CRS as grid_gdf (metric).

    Expects:
        - grid_gdf: GeoDataFrame with cell_id, geometry (CRS metric)
        - sensors_gdf: GeoDataFrame with (sensor_id, geometry)
        - sensor_stats_df: DataFrame with sensor_id, mean_vol, max_vol, std_vol

    Returns DataFrame with:
        cell_id, avg_traffic_volume, max_traffic_volume, local_traffic_uncertainty
    """
    sensors_proj = sensors_gdf.to_crs(grid_gdf.crs)

    sensors_merged = sensors_proj.merge(sensor_stats_df, on="sensor_id", how="inner")

    sensor_xy = np.vstack(
        [sensors_merged.geometry.x.values, sensors_merged.geometry.y.values]
    ).T
    tree = cKDTree(sensor_xy)

    grid_centroids = grid_gdf.geometry.centroid
    grid_xy = np.vstack([grid_centroids.x.values, grid_centroids.y.values]).T

    dists, idxs = tree.query(grid_xy, k=k)
    if k == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

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

    joined = joined[joined["cell_id_left"] != joined["cell_id_right"]]

    neighbors_series = (
        joined.groupby("cell_id_left")["cell_id_right"]
        .apply(lambda x: list(sorted(set(x))))
        .rename("neighbors")
        .reset_index()
        .rename(columns={"cell_id_left": "cell_id"})
    )

    return neighbors_series
