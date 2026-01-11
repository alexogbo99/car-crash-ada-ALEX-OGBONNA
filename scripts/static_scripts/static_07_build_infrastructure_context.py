# scripts/static_07_build_infrastructure_context.py

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

STATIC_GRID_PARQUET = DATA_INTER / "static_grid.parquet"
STATIC_FEATURES_PARQUET = DATA_INTER / "static_features.parquet"


# -------------------------------------------------------------------
# Helper to find raw files by pattern
# -------------------------------------------------------------------
def find_raw_file(patterns):
    """
    Try a list of glob patterns under data/raw and return the first match.
    Raise FileNotFoundError if nothing is found.
    """
    for pat in patterns:
        matches = list(DATA_RAW.glob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find any file in {DATA_RAW} matching patterns: {patterns}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("=== Static Step 07: Build Infrastructure Context Features ===")
    print(f"ROOT:          {ROOT}")
    print(f"DATA_RAW:      {DATA_RAW}")
    print(f"DATA_INTER:    {DATA_INTER}")
    print(f"GRID:          {STATIC_GRID_PARQUET}")
    print(f"STATIC FEATS:  {STATIC_FEATURES_PARQUET}")
    print()

    if not STATIC_GRID_PARQUET.exists():
        raise FileNotFoundError(f"Missing {STATIC_GRID_PARQUET}")
    if not STATIC_FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"Missing {STATIC_FEATURES_PARQUET}")

    # --------------------------------------------------------------
    # 1. Load grid & static features
    # --------------------------------------------------------------
    print("- Loading static_grid.parquet ...")
    grid = gpd.read_parquet(STATIC_GRID_PARQUET)
    if "geometry" not in grid.columns:
        raise ValueError("static_grid.parquet must contain 'geometry' column")
    if "cell_id" not in grid.columns:
        raise ValueError("static_grid.parquet must contain 'cell_id' column")

    if not isinstance(grid, gpd.GeoDataFrame):
        grid = gpd.GeoDataFrame(grid, geometry="geometry")
    if grid.crs is None:
        raise ValueError("static_grid.parquet must have a valid CRS")

    print(f"  -> {len(grid)} cells")

    print("- Loading static_features.parquet ...")
    static_feats = pd.read_parquet(STATIC_FEATURES_PARQUET)
    if "cell_id" not in static_feats.columns:
        raise ValueError("static_features.parquet must contain 'cell_id' column")
    print(f"  -> {len(static_feats)} rows")

    # --------------------------------------------------------------
    # 2. Build GeoDataFrames for bus, subway, VZ
    # --------------------------------------------------------------
    
    BUS_PATTERNS = ["*Shelter*.csv", "*Bus*Shelter*.csv", "*Bus_Stop*.csv"]
    SUBWAY_PATTERNS = ["*Subway*.csv", "*subway*.csv"]
    VZ_PATTERNS = ["*VZ*Intersections*.csv", "*Priority*Intersections*.csv", "*VZV*.csv"]

    # --- Bus stops ---
    bus_path = find_raw_file(BUS_PATTERNS)
    print(f"- Using bus stops file: {bus_path}")
    bus_df = pd.read_csv(bus_path)

    # Expect columns 'Longitude' and 'Latitude' as in your printout
    if not {"Longitude", "Latitude"}.issubset(bus_df.columns):
        raise ValueError(
            f"Bus file {bus_path} must contain 'Longitude' and 'Latitude' columns."
        )

    bus_gdf = gpd.GeoDataFrame(
        bus_df,
        geometry=gpd.points_from_xy(bus_df["Longitude"], bus_df["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(grid.crs)

    # --- Subway entrances ---
    subway_path = find_raw_file(SUBWAY_PATTERNS)
    print(f"- Using subway entrances file: {subway_path}")
    subway_df = pd.read_csv(subway_path)

    # Expect 'Entrance Longitude' and 'Entrance Latitude'
    if not {"Entrance Longitude", "Entrance Latitude"}.issubset(subway_df.columns):
        raise ValueError(
            f"Subway file {subway_path} must contain "
            f"'Entrance Longitude' and 'Entrance Latitude' columns."
        )

    subway_gdf = gpd.GeoDataFrame(
        subway_df,
        geometry=gpd.points_from_xy(
            subway_df["Entrance Longitude"], subway_df["Entrance Latitude"]
        ),
        crs="EPSG:4326",
    ).to_crs(grid.crs)

    # --- Vision Zero / priority intersections ---
    vz_path = find_raw_file(VZ_PATTERNS)
    print(f"- Using VZ intersections file: {vz_path}")
    vz_df = pd.read_csv(vz_path)

    # Expect 'LONG' and 'LAT'
    if not {"LONG", "LAT"}.issubset(vz_df.columns):
        raise ValueError(
            f"VZ file {vz_path} must contain 'LONG' and 'LAT' columns."
        )

    vz_gdf = gpd.GeoDataFrame(
        vz_df,
        geometry=gpd.points_from_xy(vz_df["LONG"], vz_df["LAT"]),
        crs="EPSG:4326",
    ).to_crs(grid.crs)

    # Drop any missing/empty geometries
    bus_gdf = bus_gdf[bus_gdf.geometry.notna() & ~bus_gdf.geometry.is_empty]
    subway_gdf = subway_gdf[subway_gdf.geometry.notna() & ~subway_gdf.geometry.is_empty]
    vz_gdf = vz_gdf[vz_gdf.geometry.notna() & ~vz_gdf.geometry.is_empty]

    # --------------------------------------------------------------
    # 3. Compute densities via spatial join (points-in-cell)
    # --------------------------------------------------------------
    print("- Computing bus_density via spatial join ...")
    bus_join = gpd.sjoin(
        bus_gdf[["geometry"]],
        grid[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    bus_counts = (
        bus_join.groupby("cell_id").size().reset_index(name="bus_density")
    )

    print("- Computing subway_density via spatial join ...")
    subway_join = gpd.sjoin(
        subway_gdf[["geometry"]],
        grid[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    subway_counts = (
        subway_join.groupby("cell_id").size().reset_index(name="subway_density")
    )

    print("- Computing vz_density via spatial join ...")
    vz_join = gpd.sjoin(
        vz_gdf[["geometry"]],
        grid[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    vz_counts = (
        vz_join.groupby("cell_id").size().reset_index(name="vz_density")
    )

    # --------------------------------------------------------------
    # 4. Compute dist_to_center_m (e.g. Empire State Building)
    # --------------------------------------------------------------
    print("- Computing dist_to_center_m ...")
    # Empire State Building coords in WGS84
    center_lon, center_lat = -73.985428, 40.748817
    center_point = gpd.GeoSeries(
        [Point(center_lon, center_lat)], crs="EPSG:4326"
    ).to_crs(grid.crs).iloc[0]

    # Cell centroids in same CRS
    grid_centroids = grid.copy()
    grid_centroids["centroid"] = grid_centroids.geometry.centroid
    grid_centroids["dist_to_center_m"] = (
        grid_centroids["centroid"].distance(center_point)
    ).astype("float32")

    center_df = grid_centroids[["cell_id", "dist_to_center_m"]].copy()

    # --------------------------------------------------------------
    # 5. Merge everything into a single infra DataFrame
    # --------------------------------------------------------------
    infra = grid[["cell_id"]].copy()
    infra = infra.merge(bus_counts, on="cell_id", how="left")
    infra = infra.merge(subway_counts, on="cell_id", how="left")
    infra = infra.merge(vz_counts, on="cell_id", how="left")
    infra = infra.merge(center_df, on="cell_id", how="left")

    for col in ["bus_density", "subway_density", "vz_density"]:
        infra[col] = infra[col].fillna(0).astype("int32")

    infra["dist_to_center_m"] = infra["dist_to_center_m"].astype("float32")

    print("- Infrastructure features preview:")
    print(infra.head())

    # --------------------------------------------------------------
    # 6. Merge into static_features.parquet and save
    # --------------------------------------------------------------
    print("- Merging infra features into static_features.parquet ...")
    static_enriched = static_feats.merge(infra, on="cell_id", how="left")

    # Downcast floats
    num_cols = static_enriched.select_dtypes(include=["float64", "float32"]).columns
    for c in num_cols:
        static_enriched[c] = pd.to_numeric(static_enriched[c], downcast="float")

    static_enriched.to_parquet(STATIC_FEATURES_PARQUET, index=False)
    print("Updated static_features.parquet with infrastructure context")


if __name__ == "__main__":
    main()
