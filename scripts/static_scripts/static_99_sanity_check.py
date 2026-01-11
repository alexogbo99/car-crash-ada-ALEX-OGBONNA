# scripts/static_99_sanity_check.py

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_INTER = ROOT / "data" / "intermediate"

STATIC_FEATURES_PARQUET = DATA_INTER / "static_features.parquet"


def main():
    print("=== Static 99: Sanity Check Static Features ===")
    print(f"ROOT:       {ROOT}")
    print(f"STATIC_FEATS: {STATIC_FEATURES_PARQUET}\n")

    df = pd.read_parquet(STATIC_FEATURES_PARQUET)
    print("Shape:", df.shape)

    # --------------------------------------------------
    # 1. Basic presence check for expected new columns
    # --------------------------------------------------
    expected_groups = {
        "road": [
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
        ],
        "infra": [
            "bus_density",
            "subway_density",
            "vz_density",
            "dist_to_center_m",
        ],
        "neighbors": [
            "neighbor_road_segment_count",
            "delta_road_segment_count_vs_neighbors",
            "neighbor_road_length_density",
            "delta_road_length_density_vs_neighbors",
            "neighbor_intersection_count",
            "delta_intersection_count_vs_neighbors",
            "neighbor_local_severity_uncertainty",
            "delta_local_severity_uncertainty_vs_neighbors",
        ],
    }

    for group, cols in expected_groups.items():
        present = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        print(f"\n[{group}] Present ({len(present)}): {present}")
        if missing:
            print(f"[{group}] Missing ({len(missing)}): {missing}")

    # --------------------------------------------------
    # 2. Descriptive stats for numeric new columns
    # --------------------------------------------------
    new_cols = []
    for cols in expected_groups.values():
        new_cols.extend(cols)
    new_cols = [c for c in new_cols if c in df.columns]

    print("\n--- Descriptive stats for new static features ---")
    print(df[new_cols].describe(percentiles=[0.01, 0.5, 0.99]).T)

    # --------------------------------------------------
    # 3. Sanity checks: non-negativity, weird values
    # --------------------------------------------------
    non_negative_cols = [
        "road_segment_count",
        "road_length_m",
        "road_length_density",
        "maxspeed_avg_kph",
        "maxspeed_max_kph",
        "node_count",
        "intersection_count",
        "intersection_density",
        "bus_density",
        "subway_density",
        "vz_density",
    ]
    non_negative_cols = [c for c in non_negative_cols if c in df.columns]

    print("\n--- Non-negativity check ---")
    for c in non_negative_cols:
        min_val = df[c].min()
        n_neg = (df[c] < 0).sum()
        print(f"{c}: min={min_val}, n_negative={n_neg}")

    # --------------------------------------------------
    # 4. Check neighbor vs own values for a few cols
    #    (delta should be centered roughly around 0)
    # --------------------------------------------------
    check_pairs = [
        ("road_segment_count", "neighbor_road_segment_count", "delta_road_segment_count_vs_neighbors"),
        ("road_length_density", "neighbor_road_length_density", "delta_road_length_density_vs_neighbors"),
        ("intersection_count", "neighbor_intersection_count", "delta_intersection_count_vs_neighbors"),
        ("local_severity_uncertainty",
         "neighbor_local_severity_uncertainty",
         "delta_local_severity_uncertainty_vs_neighbors"),
    ]

    print("\n--- Neighbor vs own consistency (delta distributions) ---")
    for own, neigh, delta in check_pairs:
        if own in df.columns and neigh in df.columns and delta in df.columns:
            print(f"\n{own} vs {neigh}:")
            print("  own.describe():")
            print(df[own].describe())
            print("  neigh.describe():")
            print(df[neigh].describe())
            print("  delta.describe():")
            print(df[delta].describe())
        else:
            missing = [x for x in [own, neigh, delta] if x not in df.columns]
            print(f"\nSkipping {own} / {neigh} / {delta} (missing: {missing})")

    print("\nStatic sanity check completed.")


if __name__ == "__main__":
    main()
