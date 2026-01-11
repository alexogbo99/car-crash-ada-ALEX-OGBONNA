"""
scripts/feature_tiers_C7.py
"""

from pathlib import Path
import pandas as pd

# =============================================================================
# 0. CONFIG
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT / "data" / "processed" / "train_dataset.parquet"

# =============================================================================
# 1. FEATURE GROUPS
# =============================================================================
FEATURES_TIME = ["time_bin", "year", "month", "dow", "doy"]
FEATURES_HISTORY_BASE = ["y_roll7"]
FEATURES_HISTORY_LAG1 = ["y_lag1"]

FEATURES_INFRA = ["dist_to_center_m", "bus_density", "subway_density", "vz_density"]

FEATURES_ROAD_CORE = ["intersection_density", "intersection_count", "road_length_density", "road_segment_count"]
FEATURES_ROAD_EXT = ["road_length_m", "node_count", "maxspeed_max_kph", "maxspeed_avg_kph", "intersection_complexity", "node_degree_mean"]

FEATURES_STATIC_TRAFFIC_CORE = ["vol_static_tb", "tod_share", "vol_static", "max_traffic_volume"]
FEATURES_STATIC_TRAFFIC_EXT = [
    "max_traffic_volume_tb", "local_traffic_uncertainty", "neighbor_local_traffic_uncertainty",
    "neighbor_vol_static", "neighbor_max_traffic_volume", "delta_vol_static_vs_neighbors",
    "delta_max_traffic_volume_vs_neighbors", "delta_local_traffic_uncertainty_vs_neighbors",
]

FEATURES_OTHER_CORE = ["local_severity_uncertainty"]

# ---- Realized weather --------------------------------------------------------
FEATURES_WEATHER_REAL = [
    "w_temp_mean",
    "w_wind_mean",
    "w_precip_sum",
    "w_snow_sum",
]

# ---- Climatological weather --------------------------------------------------
FEATURES_WEATHER_CLIM = [
    "temp_mean", "temp_sigma",
    "wind_mean", "wind_sigma",
    "precip_mean", "precip_sigma",
    "snow_mean", "snow_sigma",
]

# ---- Traffic pseudo-forecasts -----------------------------------------------
FEATURES_TRAFFIC_FC = [
    "fc_traffic_mean_h1d", "fc_traffic_mean_h7d",
    "fc_traffic_q95_h1d", "fc_traffic_q95_h7d",
]

# ---- Weather forecasts ---------------------------------------
FEATURES_WEATHER_FC_CORE = [
    # Temp / Wind
    "fc_temp_mean_h1d", "fc_temp_mean_h7d",
    "fc_wind_mean_h1d", "fc_wind_mean_h7d",
    "fc_temp_q95_h1d", "fc_temp_q95_h7d",
    "fc_temp_q05_h7d",
    "fc_wind_q05_h1d", "fc_wind_q05_h7d", "fc_wind_q95_h7d",
    
    #Precipitation (Sum)
    "fc_precip_mean_h1d", "fc_precip_mean_h7d",
    "fc_precip_q95_h1d", "fc_precip_q95_h7d",
    
    #Snow (Sum)
    "fc_snow_mean_h1d", "fc_snow_mean_h7d",
    "fc_snow_q95_h1d", "fc_snow_q95_h7d",
]

# =============================================================================
# 2. TIERS
# =============================================================================
TIER1_ISLAND_BASE = (FEATURES_TIME + FEATURES_INFRA + FEATURES_ROAD_CORE + FEATURES_STATIC_TRAFFIC_CORE + FEATURES_OTHER_CORE)
TIER2_WEATHER_BASE = (TIER1_ISLAND_BASE + FEATURES_WEATHER_REAL + FEATURES_WEATHER_CLIM + FEATURES_TRAFFIC_FC + FEATURES_WEATHER_FC_CORE)
TIER3_FULL_BASE = (TIER2_WEATHER_BASE + FEATURES_ROAD_EXT + FEATURES_STATIC_TRAFFIC_EXT)

TIERS = {
    "Tier1_Island": TIER1_ISLAND_BASE,
    "Tier2_Weather": TIER2_WEATHER_BASE,
    "Tier3_Full": TIER3_FULL_BASE,
}

HISTORY_VARIANTS = {
    "NoLag1": FEATURES_HISTORY_BASE,
    "WithLag1": FEATURES_HISTORY_BASE + FEATURES_HISTORY_LAG1,
}

ROLLING_OPTS = list(HISTORY_VARIANTS.keys())

def validate_against_train(sample_n: int = 5000):
    if not TRAIN_FILE.exists():
        print(f"TRAIN_FILE not found: {TRAIN_FILE}"); return
    print(f"Loading sample from: {TRAIN_FILE}")
    df = pd.read_parquet(TRAIN_FILE)
    if sample_n and len(df) > sample_n: df = df.sample(sample_n, random_state=42)
    available_cols = set(df.columns.tolist())
    all_feats = set()
    for tier_name, feats in TIERS.items():
        print(f"\n--- Validating {tier_name} ({len(feats)} features) ---")
        missing = [f for f in feats if f not in available_cols]
        all_feats.update(feats)
        if missing: print(f"MISSING: {missing}")
        else: print("All features present.")
    print(f"\nTotal distinct features: {len(all_feats)}")

if __name__ == "__main__":
    validate_against_train()