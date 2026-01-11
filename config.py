# config.py

from pathlib import Path

# ROOT
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"
DATA_PROCESSED = ROOT / "data" / "processed"

# Paths
CRASHES_PATH = DATA_RAW / "crashes.parquet"
TRAFFIC_SENSORS_PATH = DATA_RAW / "traffic_sensors.parquet"
TRAFFIC_TIMESERIES_PATH = DATA_RAW / "traffic_timeseries.parquet"
WEATHER_HOURLY_PATH = DATA_RAW / "weather_hourly.parquet"

STATIC_GRID_PATH = DATA_INTER / "static_grid.parquet"
STATIC_FEATURES_PATH = DATA_INTER / "static_features.parquet"
TRAFFIC_SIGMA_SEASON_PATH = DATA_INTER / "traffic_sigma_season.parquet"

PANEL_BASE_TRAIN_PATH = DATA_INTER / "panel_base_train.parquet"
PANEL_BASE_TEST_PATH  = DATA_INTER / "panel_base_test.parquet"

PANEL_WEATHER_TRAIN_PATH = DATA_INTER / "panel_weather_train.parquet"
PANEL_WEATHER_TEST_PATH  = DATA_INTER / "panel_weather_test.parquet"

PANEL_WEATHER_TRAFFIC_TRAIN_PATH = DATA_INTER / "panel_weather_traffic_train.parquet"
PANEL_WEATHER_TRAFFIC_TEST_PATH  = DATA_INTER / "panel_weather_traffic_test.parquet"

TRAIN_DATASET_PATH = DATA_PROCESSED / "train_dataset.parquet"
TEST_DATASET_PATH  = DATA_PROCESSED / "test_dataset.parquet"

# Time ranges
TRAIN_START = "2021-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

# Grid config
GRID_RES_DEG = 0.005
NYC_BOUNDS = {
    "minx": -74.257,
    "miny": 40.495,
    "maxx": -73.699,
    "maxy": 40.916,
}

# Time bins per day (e.g. 4x6h)
TIME_BINS = [0, 1, 2, 3]   # you already use this

# Horizons
HORIZONS = list(range(1, 8))  # 1..7
