# scripts/static_01_build_active_grid.py

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


# --------------------------------------------------------------------------------------
# PATHS & CONSTANTS (self-contained, no config.py)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]         
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

CRASHES_PARQUET = DATA_RAW / "crashes.parquet"
TRAFFIC_SENSORS_PARQUET = DATA_RAW / "traffic_sensors.parquet"

STATIC_GRID_PARQUET = DATA_INTER / "static_grid.parquet"

# Grid config (same as in your config.py)
GRID_RES_DEG = 0.005
NYC_BOUNDS = {
    "minx": -74.257,
    "miny": 40.495,
    "maxx": -73.699,
    "maxy": 40.916,
}


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
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

    # Reproject to metric CRS for distances.
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

    Assumes sensors_gdf is a GeoDataFrame with a valid CRS (we used EPSG:4326).
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


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Static Step 01: Build Active Grid ===")
    print(f"Project root: {ROOT}")
    print(f"DATA_RAW:     {DATA_RAW}")
    print(f"DATA_INTER:   {DATA_INTER}")

    DATA_INTER.mkdir(parents=True, exist_ok=True)

    # 1. Build full grid
    grid = build_full_grid()
    print(f"- Full grid cells: {len(grid)}")

    # 2. Load crashes
    crashes = pd.read_parquet(CRASHES_PARQUET)
    active_from_crashes = get_active_cell_ids_from_crashes(
        crashes_df=crashes,
        grid_gdf=grid,
        lon_col="longitude",
        lat_col="latitude",
    )
    print(f"- Active cells from crashes: {len(active_from_crashes)}")

    # 3. Load sensors
    sensors_gdf = gpd.read_parquet(TRAFFIC_SENSORS_PARQUET)  # EPSG:4326
    active_from_sensors = get_active_cell_ids_from_sensors(sensors_gdf, grid)
    print(f"- Active cells from sensors: {len(active_from_sensors)}")

    # 4. Combine & expand with neighbors
    active_ids = active_from_crashes.union(active_from_sensors)
    print(f"- Active cells before neighbors: {len(active_ids)}")

    active_plus_neighbors = expand_active_cells_with_neighbors(
        grid_gdf=grid,
        active_cell_ids=active_ids,
        buffer_meters=200,
    )
    print(f"- Active cells after neighbors: {len(active_plus_neighbors)}")

    # 5. Filter grid & save
    grid_active = grid[grid["cell_id"].isin(active_plus_neighbors)].copy()
    print(f"- Final grid size: {len(grid_active)} cells")

    grid_active.to_parquet(STATIC_GRID_PARQUET, index=False)
    print(f"Saved static grid to {STATIC_GRID_PARQUET}")


if __name__ == "__main__":
    main()
