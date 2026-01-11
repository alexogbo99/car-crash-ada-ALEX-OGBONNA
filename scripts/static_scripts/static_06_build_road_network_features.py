# scripts/static_06_build_road_network_features.py

from pathlib import Path
import gc
import re

import numpy as np
import pandas as pd
import geopandas as gpd


# --------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"

STATIC_GRID_PARQUET = DATA_INTER / "static_grid.parquet"
STATIC_FEATURES_PARQUET = DATA_INTER / "static_features.parquet"

STREETS_EDGES_GEOJSON = DATA_RAW / "nyc_streets_edges.geojson"
STREETS_NODES_GEOJSON = DATA_RAW / "nyc_streets_nodes.geojson"

STREETS_LEGACY_GEOJSON = DATA_RAW / "nyc_streets.geojson"


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def clean_maxspeed(val):
    """
    Try to convert OSM 'maxspeed' values to km/h (float).

    Handles:
        - None / NaN
        - numpy arrays, lists, tuples -> take first element
        - numeric scalars
        - strings like '50', '50 km/h', '30 mph', '30;40', '30,40'

    Returns np.nan if parsing fails.
    """
    # Unwrap numpy arrays, lists, tuples: take first element
    if isinstance(val, (np.ndarray, list, tuple)):
        if len(val) == 0:
            return np.nan
        val = val[0]

    # None → NaN
    if val is None:
        return np.nan

    # Try NaN check on scalar values
    try:
        if pd.isna(val):
            return np.nan
    except TypeError:
        # If pd.isna can't handle this type, just give up gracefully
        return np.nan

    # Pure numeric → just cast to float (assume km/h)
    if isinstance(val, (int, float, np.number)):
        return float(val)

    # If it's not a string by now, we don't know what it is
    if not isinstance(val, str):
        return np.nan

    # ------------------------------------------------------------------
    # String parsing from here on
    # ------------------------------------------------------------------
    s = val.strip().lower()
    if not s:
        return np.nan

    # Sometimes values like '30;40' or '30,40' -> take the first part
    s = s.replace(";", ",")
    parts = s.split(",")
    s = parts[0].strip()

    # Extract number + optional unit, e.g. '30 mph', '50km/h'
    m = re.match(r"^(\d+(\.\d+)?)(.*)$", s)
    if not m:
        return np.nan

    num = float(m.group(1))
    unit = m.group(3).strip()

    # mph → convert to kph
    if "mph" in unit:
        return num * 1.60934

    # default: km/h
    return num


# --------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------
def main():
    print("=== Static Step 06: Build Road Network Features ===")
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

    # Edge file: prefer edges_geojson, fallback to legacy nyc_streets.geojson
    if STREETS_EDGES_GEOJSON.exists():
        edges_path = STREETS_EDGES_GEOJSON
    elif STREETS_LEGACY_GEOJSON.exists():
        edges_path = STREETS_LEGACY_GEOJSON
    else:
        raise FileNotFoundError(
            f"Could not find street edges file. "
            f"Expected {STREETS_EDGES_GEOJSON} or {STREETS_LEGACY_GEOJSON}"
        )

    print(f"- Using edges from: {edges_path}")

    # Nodes are optional but needed for intersections & degree features
    nodes_available = STREETS_NODES_GEOJSON.exists()
    if nodes_available:
        print(f"- Using nodes from: {STREETS_NODES_GEOJSON}")
    else:
        print("Node file not found; intersection & degree features will be skipped.")

    # ------------------------------------------------------------------
    # 1. Load grid and static_features
    # ------------------------------------------------------------------
    print("- Loading static_grid.parquet ...")
    grid = gpd.read_parquet(STATIC_GRID_PARQUET)
    print(f"  -> {len(grid)} cells")

    if "cell_id" not in grid.columns:
        raise ValueError("static_grid.parquet must contain 'cell_id' column")

    # Ensure GeoDataFrame with CRS
    if not isinstance(grid, gpd.GeoDataFrame):
        grid = gpd.GeoDataFrame(grid, geometry="geometry")
    if grid.crs is None:
        raise ValueError("static_grid.parquet must have a valid CRS")

    # Cell area in m² (UTM CRS)
    grid = grid[["cell_id", "geometry"]].copy()
    grid["cell_area_m2"] = grid.geometry.area.astype("float32")

    print("- Loading static_features.parquet ...")
    static_feats = pd.read_parquet(STATIC_FEATURES_PARQUET)
    print(f"  -> {len(static_feats)} rows")

    if "cell_id" not in static_feats.columns:
        raise ValueError("static_features.parquet must contain 'cell_id' column")

    # ------------------------------------------------------------------
    # 2. Load edges, reproject, compute length & speed
    # ------------------------------------------------------------------
    print("- Loading street edges ...")
    streets = gpd.read_file(edges_path)
    print(f"  -> {len(streets)} raw edges")

    if streets.crs is None:
        # OSM via osmnx should be EPSG:4326
        streets = streets.set_crs(epsg=4326)

    # Reproject to grid CRS
    if streets.crs != grid.crs:
        print(f"  Reprojecting edges from {streets.crs} to {grid.crs} ...")
        streets = streets.to_crs(grid.crs)

    # Keep only what we need
    cols_keep = ["segment_id", "geometry"]
    if "segment_id" not in streets.columns:
        # If for some reason it's missing, create it
        streets = streets.reset_index(drop=True)
        streets["segment_id"] = streets.index.astype("int32")

    streets = streets[cols_keep + [c for c in streets.columns if c == "maxspeed"]]

    # Drop missing geometry
    streets = streets[~streets.geometry.is_empty & streets.geometry.notna()].copy()
    print(f"  -> {len(streets)} edges after cleaning")

    # Length in meters
    streets["road_length_m"] = streets.geometry.length.astype("float32")

    # Clean maxspeed to km/h
    if "maxspeed" in streets.columns:
        print("- Cleaning 'maxspeed' to km/h ...")
        streets["maxspeed_kph"] = streets["maxspeed"].apply(clean_maxspeed).astype("float32")
    else:
        streets["maxspeed_kph"] = np.nan

    # ------------------------------------------------------------------
    # 3. Spatial join edges to grid -> per-cell aggregates
    # ------------------------------------------------------------------
    print("- Spatial join: edges -> grid (for segments & length) ...")
    edges_join = gpd.sjoin(
        streets[["segment_id", "geometry", "road_length_m", "maxspeed_kph"]],
        grid[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    print(f"  -> {len(edges_join)} joined rows")

    # Aggregate per cell
    edges_agg = (
        edges_join
        .groupby("cell_id")
        .agg(
            road_segment_count=("segment_id", "nunique"),
            road_length_m=("road_length_m", "sum"),
            maxspeed_avg_kph=("maxspeed_kph", "mean"),
            maxspeed_max_kph=("maxspeed_kph", "max"),
        )
        .reset_index()
    )

    # Merge cell area to compute density
    edges_agg = edges_agg.merge(
        grid[["cell_id", "cell_area_m2"]],
        on="cell_id",
        how="left",
    )
    edges_agg["road_length_density"] = (
        edges_agg["road_length_m"] / edges_agg["cell_area_m2"].replace(0, np.nan)
    ).astype("float32")

    # Clean up types
    edges_agg["road_segment_count"] = edges_agg["road_segment_count"].astype("int32")
    edges_agg["road_length_m"] = edges_agg["road_length_m"].astype("float32")
    edges_agg["maxspeed_avg_kph"] = edges_agg["maxspeed_avg_kph"].astype("float32")
    edges_agg["maxspeed_max_kph"] = edges_agg["maxspeed_max_kph"].astype("float32")

    del edges_join
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Node-based features: intersections, node degree, complexity
    # ------------------------------------------------------------------
    if nodes_available:
        print("- Loading street nodes for intersection & degree features ...")
        nodes = gpd.read_file(STREETS_NODES_GEOJSON)
        print(f"  -> {len(nodes)} raw nodes")

        if nodes.crs is None:
            nodes = nodes.set_crs(epsg=4326)
        if nodes.crs != grid.crs:
            print(f"  Reprojecting nodes from {nodes.crs} to {grid.crs} ...")
            nodes = nodes.to_crs(grid.crs)

        if "degree" not in nodes.columns:
            print("⚠️  'degree' column missing in nodes; setting degree=0.")
            nodes["degree"] = 0

        # Drop missing geometry
        nodes = nodes[~nodes.geometry.is_empty & nodes.geometry.notna()].copy()

        # Join nodes to grid
        print("- Spatial join: nodes -> grid ...")
        nodes_join = gpd.sjoin(
            nodes[["geometry", "degree"]],
            grid[["cell_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        print(f"  -> {len(nodes_join)} joined node rows")

        # All nodes per cell
        node_stats = (
            nodes_join
            .groupby("cell_id")
            .agg(
                node_count=("degree", "size"),
                node_degree_mean=("degree", "mean"),
            )
            .reset_index()
        )

        # Intersections: nodes with degree >= 3
        intersections = nodes_join[nodes_join["degree"] >= 3]
        if len(intersections) > 0:
            inter_stats = (
                intersections
                .groupby("cell_id")
                .agg(
                    intersection_count=("degree", "size"),
                    intersection_complexity=("degree", "mean"),
                )
                .reset_index()
            )
        else:
            inter_stats = pd.DataFrame(
                columns=["cell_id", "intersection_count", "intersection_complexity"]
            )

        # Merge node + intersection stats
        node_full = node_stats.merge(inter_stats, on="cell_id", how="left")

        # Attach cell area and compute intersection density
        node_full = node_full.merge(
            grid[["cell_id", "cell_area_m2"]],
            on="cell_id",
            how="left",
        )
        node_full["intersection_count"] = node_full["intersection_count"].fillna(0).astype("int32")
        node_full["intersection_complexity"] = node_full["intersection_complexity"].astype("float32")
        node_full["node_count"] = node_full["node_count"].astype("int32")
        node_full["node_degree_mean"] = node_full["node_degree_mean"].astype("float32")

        node_full["intersection_density"] = (
            node_full["intersection_count"] / node_full["cell_area_m2"].replace(0, np.nan)
        ).astype("float32")

        del nodes_join, intersections
        gc.collect()
    else:
        # No node file: prepare empty node_full so merge doesn't break
        node_full = pd.DataFrame(
            columns=[
                "cell_id",
                "node_count",
                "node_degree_mean",
                "intersection_count",
                "intersection_complexity",
                "intersection_density",
            ]
        )

    # ------------------------------------------------------------------
    # 5. Merge edges + node-based features into a single per-cell DF
    # ------------------------------------------------------------------
    road_features = edges_agg.merge(
        node_full,
        on="cell_id",
        how="left",
    )

    # Fill NaNs for cells with no nodes / no edges
    for col in [
        "road_segment_count",
        "road_length_m",
        "road_length_density",
        "maxspeed_avg_kph",
        "maxspeed_max_kph",
        "node_count",
        "node_degree_mean",
        "intersection_count",
        "intersection_complexity",
        "intersection_density",
    ]:
        if col in road_features.columns:
            if road_features[col].dtype.kind in {"i", "u"}:
                road_features[col] = road_features[col].fillna(0).astype(road_features[col].dtype)
            else:
                road_features[col] = road_features[col].fillna(0.0).astype("float32")

    print("- Road network features (per cell) preview:")
    print(road_features.head())

    # ------------------------------------------------------------------
    # 6. Merge into static_features.parquet and save
    # ------------------------------------------------------------------
    print("- Merging road network features into static_features.parquet ...")
    static_enriched = static_feats.merge(road_features, on="cell_id", how="left")

    # Light downcast on floats
    num_cols = static_enriched.select_dtypes(include=["float64", "float32"]).columns
    for c in num_cols:
        static_enriched[c] = pd.to_numeric(static_enriched[c], downcast="float")

    static_enriched.to_parquet(STATIC_FEATURES_PARQUET, index=False)
    print(f"Updated static_features.parquet with road network features.")


if __name__ == "__main__":
    main()
