import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from pathlib import Path
from shapely.geometry import box
import matplotlib.dates as mdates

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "descriptive_stats"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Grid Config
GRID_RES_DEG = 0.005
NYC_BOUNDS = {
    "minx": -74.257, "miny": 40.495,
    "maxx": -73.699, "maxy": 40.916,
}

# --- VISUAL PRESETS ---
sns.set_theme(style="ticks", context="talk", font_scale=1.0)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True

# Colors
COLOR_RAW = "#95a5a6"    # Slate Grey
COLOR_TREND = "#c0392b"  # Crimson Red
COLOR_PRE = "#2980b9"    # Belize Blue
COLOR_POST = "#e67e22"   # Carrot Orange

# ==============================================================================
# HELPER
# ==============================================================================
def reconstruct_nyc_grid():
    print("   🛠️  Reconstructing Grid Geometry...")
    minx, miny = NYC_BOUNDS["minx"], NYC_BOUNDS["miny"]
    maxx, maxy = NYC_BOUNDS["maxx"], NYC_BOUNDS["maxy"]
    res = GRID_RES_DEG
    x_coords = np.arange(minx, maxx, res)
    y_coords = np.arange(miny, maxy, res)
    polys = []
    for x in x_coords:
        for y in y_coords:
            polys.append(box(x, y, x + res, y + res))
    gdf = gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:4326")
    gdf["cell_id"] = gdf.index.astype(int)
    return gdf

# ==============================================================================
# PART 1: RAW CRASH DATA ANALYSIS
# ==============================================================================
def analyze_raw_crashes():
    print("\n🔵 PART 1: Analyzing Raw Crash Data...")
    raw_crash_path = DATA_RAW / "Motor_Vehicle_Collisions_-_Crashes_20251129.csv"
    
    if not raw_crash_path.exists():
        print(f"   ⚠️ Raw crash file missing: {raw_crash_path}")
        return

    df = pd.read_csv(raw_crash_path, usecols=["CRASH DATE"])
    df["CRASH DATE"] = pd.to_datetime(df["CRASH DATE"])
    daily_counts = df.groupby("CRASH DATE").size().reset_index(name="crash_count")
    daily_counts = daily_counts.sort_values("CRASH DATE")

    # Stats
    covid_start = pd.Timestamp("2020-03-01")
    pre = daily_counts[daily_counts["CRASH DATE"] < covid_start]
    post = daily_counts[daily_counts["CRASH DATE"] >= covid_start]

    # --- STATISTICS WITH VARIANCE & STD ---
    stats = {
        "GLOBAL": {
            "Mean": daily_counts["crash_count"].mean(), 
            "Median": daily_counts["crash_count"].median(),
            "Max": daily_counts["crash_count"].max(),
            "Std Dev": daily_counts["crash_count"].std(),
            "Variance": daily_counts["crash_count"].var()
        },
        "PRE-2020": {
            "Mean": pre["crash_count"].mean(), 
            "Median": pre["crash_count"].median(),
            "Std Dev": pre["crash_count"].std(),
            "Variance": pre["crash_count"].var()
        },
        "POST-2020": {
            "Mean": post["crash_count"].mean(), 
            "Median": post["crash_count"].median(),
            "Std Dev": post["crash_count"].std(),
            "Variance": post["crash_count"].var()
        }
    }

    with open(RESULTS_DIR / "01_raw_crash_statistics.txt", "w") as f:
        f.write("=== RAW CRASH STATISTICS (STRUCTURAL BREAK) ===\n")
        for k, v in stats.items():
            f.write(f"\n[{k}]\n")
            for m, val in v.items(): f.write(f"{m:<12}: {val:,.2f}\n")

    # --- PLOT 1: TIME SERIES ---
    fig, ax = plt.subplots(figsize=(14, 7))
    # Raw Data
    ax.plot(daily_counts["CRASH DATE"], daily_counts["crash_count"], 
            color=COLOR_RAW, alpha=0.25, label="Daily Count", linewidth=1)
    # Rolling Trend
    daily_counts["rolling"] = daily_counts["crash_count"].rolling(30).mean()
    ax.plot(daily_counts["CRASH DATE"], daily_counts["rolling"], 
            color=COLOR_TREND, linewidth=2.5, label="30-Day Trend")
    
    # Break Line
    ax.axvline(covid_start, color="#2c3e50", linestyle="--", alpha=0.8, linewidth=1.5, label="Covid-19 Lockdowns")
    
    ax.set_title("NYC Car Crashes: Structural Break Analysis (2012-2025)", fontweight='bold', pad=20)
    ax.set_ylabel("Crashes per Day")
    ax.set_xlabel("")
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    # Improve Date Axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_crash_time_series_structural.png")
    plt.close()

    # --- PLOT 2: DISTRIBUTION SPLIT ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.kdeplot(pre["crash_count"], fill=True, color=COLOR_PRE, label="Pre-2020", alpha=0.2, linewidth=2, ax=ax)
    sns.kdeplot(post["crash_count"], fill=True, color=COLOR_POST, label="Post-2020", alpha=0.2, linewidth=2, ax=ax)
    
    # Mean Lines
    ax.axvline(pre["crash_count"].mean(), color=COLOR_PRE, linestyle="--", linewidth=1.5)
    ax.axvline(post["crash_count"].mean(), color=COLOR_POST, linestyle="--", linewidth=1.5)
    
    ax.set_title("Shift in Crash Frequency Distribution", fontweight='bold', pad=15)
    ax.set_xlabel("Daily Crashes")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_crash_distribution_split.png")
    plt.close()

# ==============================================================================
# PART 2: PROCESSED DATA ANALYSIS
# ==============================================================================
def analyze_processed_data():
    print("\n🔵 PART 2: Analyzing Processed Data...")
    train_path = DATA_PROCESSED / "train_dataset.parquet"
    if not train_path.exists(): return

    cols = ["cell_id", "date", "time_bin", "vol_static", "w_temp_mean", "w_precip_sum", "road_length_m"]
    try: df = pd.read_parquet(train_path, columns=cols)
    except: return

    # --- A. TRAFFIC MAP ---
    cell_traffic = df.groupby("cell_id")["vol_static"].mean().reset_index()
    mi, ma = cell_traffic["vol_static"].min(), cell_traffic["vol_static"].max()
    cell_traffic["vol_norm"] = (cell_traffic["vol_static"] - mi) / (ma - mi)
    
    grid = reconstruct_nyc_grid()
    map_data = grid.merge(cell_traffic, on="cell_id", how="inner")
    
    if map_data.crs is None: map_data.set_crs(epsg=4326, inplace=True)
    map_data = map_data.to_crs(epsg=3857)

    if not map_data.empty:
        fig, ax = plt.subplots(figsize=(10, 10))
        map_data.plot(column="vol_norm", cmap="Reds", 
                      ax=ax, alpha=0.9, edgecolor="none") # No edge for smoother heatmap look
        
        try: ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except: pass
        
        ax.set_title("Traffic Intensity Heatmap (Normalized)", fontsize=16, fontweight='bold', pad=10)
        ax.set_axis_off() # Completely remove box
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "02_traffic_heatmap_normalized.png")
        plt.close()

    # --- B. WEATHER & INFRASTRUCTURE ---
    daily_weather = df[["date", "w_temp_mean", "w_precip_sum"]].drop_duplicates().copy()
    daily_weather["date"] = pd.to_datetime(daily_weather["date"])
    weather_2022 = daily_weather[daily_weather["date"].dt.year == 2022].sort_values("date").copy()

    # 1. Temperature Time Series
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(weather_2022["date"], weather_2022["w_temp_mean"], color=COLOR_RAW, alpha=0.5, label="Daily Temp")
    
    weather_2022["rolling"] = weather_2022["w_temp_mean"].rolling(window=7).mean()
    ax.plot(weather_2022["date"], weather_2022["rolling"], color=COLOR_TREND, linewidth=3, label="7-Day Avg")
    
    ax.set_title("Annual Temperature Cycle (2022)", fontweight='bold', pad=15)
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    
    # Format Dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_weather_temperature_2022.png")
    plt.close()

    # 2. Precipitation
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(daily_weather["w_precip_sum"], bins=40, color="#3498db", alpha=0.7, edgecolor="white", ax=ax)
    
    ax.set_yscale('log')
    ax.set_title("Precipitation Frequency", fontweight='bold', pad=15)
    ax.set_xlabel("Daily Precipitation (mm)")
    ax.set_ylabel("Days (Log Scale)")
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_weather_precipitation.png")
    plt.close()

    # 3. Infrastructure
    cell_infra = df[["cell_id", "road_length_m"]].drop_duplicates()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(cell_infra["road_length_m"], kde=True, color="#7f8c8d", fill=True, alpha=0.4, ax=ax)
    
    ax.set_title("Road Network Density", fontweight='bold', pad=15)
    ax.set_xlabel("Road Length per Cell (m)")
    ax.set_ylabel("Frequency")
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_infrastructure_road_density.png")
    plt.close()

    print("   ✅ All visualizations saved.")

if __name__ == "__main__":
    print("=== GENERATING DESCRIPTIVE STATISTICS (VISUAL POLISH) ===")
    analyze_raw_crashes()
    analyze_processed_data()
    print(f"\n Done! Results: {RESULTS_DIR}")