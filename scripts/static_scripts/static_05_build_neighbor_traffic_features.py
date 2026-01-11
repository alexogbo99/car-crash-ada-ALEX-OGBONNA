# scripts/static_05_build_neighbor_traffic_features.py

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import geopandas as gpd


# --------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_INTER = ROOT / "data" / "intermediate"

STATIC_GRID_PARQUET = DATA_INTER / "static_grid.parquet"
STATIC_FEATURES_PARQUET = DATA_INTER / "static_features.parquet"


# --------------------------------------------------------------------------------------
# NEIGHBOR MAP
# --------------------------------------------------------------------------------------
def build_neighbor_map(grid: gpd.GeoDataFrame) -> dict:
    """
    Build adjacency list: for each cell_id, list of neighboring cell_ids that
    geometrically intersect (touch/overlap). Self-pairs are removed.
    """
    if not isinstance(grid, gpd.GeoDataFrame):
        grid = gpd.GeoDataFrame(grid, geometry="geometry")

    if grid.crs is None:
        grid = grid.set_crs(epsg=4326)

    grid = grid[["cell_id", "geometry"]].copy()

    grid_adj = gpd.sjoin(
        grid,
        grid,
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
    )

    grid_adj = grid_adj[grid_adj["cell_id_left"] != grid_adj["cell_id_right"]]

    neighbor_map = (
        grid_adj.groupby("cell_id_left")["cell_id_right"]
        .apply(list)
        .to_dict()
    )

    del grid_adj
    gc.collect()

    return neighbor_map


# --------------------------------------------------------------------------------------
# NEIGHBOR FEATURES
# --------------------------------------------------------------------------------------
def compute_neighbor_traffic_features(static_df: pd.DataFrame, neighbor_map: dict) -> pd.DataFrame:
    """
    Generic neighbor-mean features for a configurable set of static columns.

    Given static_df with columns:
        cell_id, [base_cols...]

    and neighbor_map {cell_id: [neighbor_ids]},

    compute for each base column c in base_cols ∩ static_df.columns:
        neighbor_c = mean of c over neighbors
        delta_c_vs_neighbors = c - neighbor_c

    Returned DataFrame has one row per cell_id.
    """
    static_df = static_df.set_index("cell_id")

    # Base columns we care about:
    # - original traffic static
    # - plus road network context
    base_cols = [
        "vol_static",
        "max_traffic_volume",
        "local_traffic_uncertainty",
        # Road network
        "road_segment_count",
        "road_length_density",
        "intersection_count",
        # Severity
        "local_severity_uncertainty",
    ]


    cols = [c for c in base_cols if c in static_df.columns]
    if not cols:
        raise ValueError(
            "static_features.parquet must contain at least one of: "
            f"{base_cols}"
        )

    results = {"cell_id": []}
    for c in cols:
        results[f"neighbor_{c}"] = []

    for cid in static_df.index:
        nbrs = neighbor_map.get(cid, [])
        if nbrs:
            nbrs_valid = [n for n in nbrs if n in static_df.index]
            if nbrs_valid:
                neigh_vals = static_df.loc[nbrs_valid, cols]
                mean_vals = neigh_vals.mean(axis=0)
            else:
                mean_vals = pd.Series({c: np.nan for c in cols})
        else:
            mean_vals = pd.Series({c: np.nan for c in cols})

        results["cell_id"].append(cid)
        for c in cols:
            results[f"neighbor_{c}"].append(mean_vals[c])

    neighbor_df = pd.DataFrame(results)

    # Add deltas (cell minus neighbor mean)
    neighbor_df = neighbor_df.set_index("cell_id")
    for c in cols:
        mean_col = f"neighbor_{c}"
        delta_col = f"delta_{c}_vs_neighbors"
        cell_vals = static_df[c]
        neigh_means = neighbor_df[mean_col]
        neighbor_df[delta_col] = cell_vals - neigh_means

    neighbor_df = neighbor_df.reset_index()

    return neighbor_df


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print("=== Static Step 05: Build Neighbor Static Features ===")
    print(f"ROOT:          {ROOT}")
    print(f"DATA_INTER:    {DATA_INTER}")
    print(f"GRID:          {STATIC_GRID_PARQUET}")
    print(f"STATIC FEATS:  {STATIC_FEATURES_PARQUET}")

    if not STATIC_GRID_PARQUET.exists():
        raise FileNotFoundError(f"Missing {STATIC_GRID_PARQUET}")
    if not STATIC_FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"Missing {STATIC_FEATURES_PARQUET}")

    # 1. Load grid + build neighbor map
    print("- Loading static_grid.parquet ...")
    grid = gpd.read_parquet(STATIC_GRID_PARQUET)
    print(f"  -> {len(grid)} cells")

    print("- Building neighbor map from grid geometry ...")
    neighbor_map = build_neighbor_map(grid)
    print(f"  -> Neighbor map built for {len(neighbor_map)} cells")

    # 2. Load static features
    print("- Loading static_features.parquet ...")
    static_feats = pd.read_parquet(STATIC_FEATURES_PARQUET)
    print(f"  -> {len(static_feats)} rows")

    if "cell_id" not in static_feats.columns:
        raise ValueError("static_features.parquet must contain 'cell_id' column")

    # 3. Compute neighbor features
    print("- Computing neighbor averages & deltas ...")
    neighbor_static = compute_neighbor_traffic_features(static_feats, neighbor_map)
    print("  -> Computed neighbor features for "
          f"{neighbor_static['cell_id'].nunique()} cells")

    # 4. Merge back into static_features and save
    print("- Merging neighbor features back into static_features ...")
    static_feats_enriched = static_feats.merge(neighbor_static, on="cell_id", how="left")

    # Light downcast
    num_cols = static_feats_enriched.select_dtypes(include=["float64", "float32"]).columns
    for col in num_cols:
        static_feats_enriched[col] = pd.to_numeric(static_feats_enriched[col], downcast="float")

    static_feats_enriched.to_parquet(STATIC_FEATURES_PARQUET, index=False)
    print(f"Updated static_features.parquet with neighbor static features.")



if __name__ == "__main__":
    main()
