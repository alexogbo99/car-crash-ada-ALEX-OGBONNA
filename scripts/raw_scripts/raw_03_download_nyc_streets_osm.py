# scripts/raw_03_download_nyc_streets_osm.py

from pathlib import Path

import geopandas as gpd
import osmnx as ox


# --------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Edges / nodes outputs
STREETS_EDGES_GEOJSON = DATA_RAW / "nyc_streets_edges.geojson"
STREETS_NODES_GEOJSON = DATA_RAW / "nyc_streets_nodes.geojson"

# Backward-compat alias for the edges
STREETS_LEGACY_GEOJSON = DATA_RAW / "nyc_streets.geojson"


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    place_name = "New York City, New York, USA"

    print("=== Raw Step 03: Download NYC Street Network from OSM ===")
    print(f"ROOT:      {ROOT}")
    print(f"DATA_RAW:  {DATA_RAW}")
    print(f"Place:     {place_name}")
    print()

    # ------------------------------------------------------------------
    # 1. Download OSM graph (drivable network)
    # ------------------------------------------------------------------
    print(f"- Downloading drivable street network for: {place_name}")
    G = ox.graph_from_place(place_name, network_type="drive")
    print(G)

    # ------------------------------------------------------------------
    # 2. Convert to GeoDataFrames (nodes + edges)
    # ------------------------------------------------------------------
    print("- Converting graph to GeoDataFrames (nodes + edges) ...")
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    print(f"  -> {len(nodes)} nodes, {len(edges)} edges")

    # Add a simple integer segment_id for edges
    if "segment_id" not in edges.columns:
        edges = edges.reset_index(drop=True)
        edges["segment_id"] = edges.index.astype("int32")

    # Compute node degree from graph and attach
    print("- Computing node degree ...")
    degree_dict = dict(G.degree())
    # nodes index matches graph nodes
    nodes["degree"] = nodes.index.map(degree_dict).astype("int32")

    # ------------------------------------------------------------------
    # 3. Save to GeoJSON (no Parquet, to avoid pyarrow issues)
    # ------------------------------------------------------------------
    print("- Saving edges to GeoJSON ...")
    edges.to_file(STREETS_EDGES_GEOJSON, driver="GeoJSON")
    # Backward compatibility: also save as nyc_streets.geojson
    edges.to_file(STREETS_LEGACY_GEOJSON, driver="GeoJSON")
    print(f"  -> {STREETS_EDGES_GEOJSON}")
    print(f"  -> {STREETS_LEGACY_GEOJSON}")

    print("- Saving nodes to GeoJSON ...")
    nodes.to_file(STREETS_NODES_GEOJSON, driver="GeoJSON")
    print(f"  -> {STREETS_NODES_GEOJSON}")

    print("Done: NYC street network (nodes + edges) saved in data/raw/.")


if __name__ == "__main__":
    main()
