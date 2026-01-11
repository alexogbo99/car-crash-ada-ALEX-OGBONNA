import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set academic style
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Color palette - simplified for black and white with gray
PALETTE = {
    'Direct': '#000000',      # Solid black for Direct
    'Hurdle': '#000000',      # Black for Hurdle (but will use dotted line)
    'bar_gray': '#666666',    # Gray for bars
    'bar_white': '#FFFFFF',   # White for bars (with black edge)
}

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "results" / "plots" / "00_championship_ladder"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment Structure
MODEL_STRUCTURE = [
    ("C7", "Direct", "Heavy", "C7_Direct_Heavy"),
    ("C7", "Hurdle", "Heavy", "C7_Hurdle_Heavy"),
    ("C1", "Direct", "Heavy", "C1_Direct_Heavy"),
    ("C1", "Hurdle", "Heavy", "C1_Hurdle_Heavy"),
    ("C7", "Direct", "Light", "C7_Direct_Light"),
    ("C7", "Hurdle", "Light", "C7_Hurdle_Light"),
    ("C1", "Direct", "Light", "C1_Direct_Light"),
    ("C1", "Hurdle", "Light", "C1_Hurdle_Light"),
]

ANALYSIS_PASSES = [
    {
        "metric_key": "rmse",
        "json_suffix": "final_test_metrics_rmse.json",
        "pretty_name": "RMSE",
        "csv_name": "master_championship_results_rmse.csv"
    },
    {
        "metric_key": "poisson_dev",
        "json_suffix": "final_test_metrics_dev_poisson.json",
        "pretty_name": "Poisson Deviance",
        "csv_name": "master_championship_results_poisson_dev.csv"
    }
]

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
def load_championship_table(metric_key, json_filename):
    print(f"Loading results from: {json_filename}")
    records = []

    for family, method, mode, folder in MODEL_STRUCTURE:
        path = RESULTS_DIR / folder / json_filename
        
        if not path.exists():
            print(f"Missing: {path}")
            continue
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                
            for target_name, tiers in data.items():
                for tier_name, metrics in tiers.items():
                    score = metrics.get(metric_key)
                    
                    if score is not None:
                        # Get algorithm - accept any algorithm name
                        algo = metrics.get("algo", "Unknown")
                        # Simply use the algorithm name as is
                        algo_display = algo if algo else "Unknown"
                        
                        records.append({
                            "Family": family,
                            "Target": target_name,
                            "Method": method,
                            "Mode": mode,
                            "Tier": tier_name,
                            "Algo": algo_display,
                            "Score": float(score),
                        })
        except Exception as e:
            print(f"Error reading {path}: {e}")

    df = pd.DataFrame(records)
    
    if not df.empty:
        # Create tier order
        tier_order = ["Tier1_Island", "Tier2_Weather", "Tier3_Full"]
        df["Tier_Order"] = pd.Categorical(df["Tier"], categories=tier_order, ordered=True)
        df["Tier_Number"] = df["Tier"].str.extract(r'Tier(\d+)').astype(int)
        
        # Create nicer labels
        tier_map = {
            "Tier1_Island": "Tier 1",
            "Tier2_Weather": "Tier 2",
            "Tier3_Full": "Tier 3"
        }
        df["Tier_Label"] = df["Tier"].map(tier_map)
        
    return df
# ==============================================================================
# 2. MODIFIED PLOTTING FUNCTIONS
# ==============================================================================
def plot_simple_ladder(df, metric_name):
    """Simple ladder plot with model names, black lines, and adjusted y-axis"""
    if df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    combinations = [
        ("C7", "Heavy"),
        ("C7", "Light"),
        ("C1", "Heavy"),
        ("C1", "Light")
    ]
    
    for idx, (family, mode) in enumerate(combinations):
        ax = axes[idx]
        subset = df[(df["Family"] == family) & (df["Mode"] == mode)].copy()
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{family} - {mode}")
            continue
        
        # Plot each method
        for method in subset["Method"].unique():
            method_data = subset[subset["Method"] == method].sort_values("Tier_Number")
            
            # Set line style based on method
            linestyle = '-' if method == "Direct" else '--'
            marker = 'o' if method == "Direct" else 's'
            
            # Plot line - both in black, different styles
            ax.plot(method_data["Tier_Number"], method_data["Score"], 
                   color='black', linewidth=2, marker=marker, markersize=8,
                   linestyle=linestyle, label=method)
            
            # Add value labels with algorithm name
            for _, row in method_data.iterrows():
                label = f"{row['Score']:.3f}\n({row['Algo']})"
                ax.annotate(label, 
                          xy=(row["Tier_Number"], row["Score"]),
                          xytext=(0, 8), textcoords='offset points',
                          ha='center', va='bottom', fontsize=8,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   alpha=0.8, edgecolor='black', linewidth=0.5))
        
        ax.set_title(f"{family} - {mode}", fontweight='bold')
        ax.set_xlabel("Model Tier")
        ax.set_ylabel(metric_name)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Base", "+Weather", "+Traffic"])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust y-axis for better visibility
        if not subset.empty:
            # Get min and max scores for this subplot
            y_min = subset["Score"].min()
            y_max = subset["Score"].max()
            
            # Calculate buffer: subtract 5% of range from min, add 10% to max
            y_range = y_max - y_min
            buffer_lower = 0.1 * y_range  # 10% buffer for lower bound
            buffer_upper = 0.2 * y_range  # 20% buffer for upper bound
            
            # Set new y-axis limits
            new_y_min = max(0, y_min - buffer_lower)  # Don't go below 0 for positive metrics
            new_y_max = y_max + buffer_upper
            
            # Make sure we have some range if min and max are the same
            if new_y_min == new_y_max:
                new_y_min = max(0, new_y_min - 0.1 * new_y_min)
                new_y_max = new_y_max + 0.1 * new_y_max
            
            ax.set_ylim(new_y_min, new_y_max)
        
        if idx == 0:
            # Create custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', lw=2, linestyle='-', label='Direct'),
                Line2D([0], [0], color='black', lw=2, linestyle='--', label='Hurdle')
            ]
            ax.legend(handles=legend_elements, title="Method")
    
    plt.suptitle(f"Model Performance Comparison: {metric_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    safe_name = metric_name.replace(" ", "_").lower()
    out_name = f"ladder_{safe_name}.png"
    plt.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Ladder Plot: {out_name}")

def plot_grouped_bar(df, metric_name):
    """Grouped bar chart with gray and white colors"""
    if df.empty:
        return
    
    # Filter to Tier 3 only
    tier3_data = df[df["Tier"] == "Tier3_Full"].copy()
    if tier3_data.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Heavy models
    heavy_data = tier3_data[tier3_data["Mode"] == "Heavy"]
    if not heavy_data.empty:
        ax = axes[0]
        heavy_pivot = heavy_data.pivot_table(index='Family', columns='Method', values='Score', aggfunc='mean')
        
        # Plot with gray and white bars
        colors = [PALETTE['bar_gray'], PALETTE['bar_white']]
        edgecolors = ['black', 'black']
        
        # Plot bars
        x = np.arange(len(heavy_pivot.index))
        width = 0.35
        
        for i, method in enumerate(['Direct', 'Hurdle']):
            if method in heavy_pivot.columns:
                offset = (i - 0.5) * width
                ax.bar(x + offset, heavy_pivot[method], width, 
                      color=colors[i], edgecolor=edgecolors[i], 
                      linewidth=1, label=method)
        
        ax.set_title("Heavy Models (Tier 3)", fontweight='bold')
        ax.set_xlabel("Output")  # Changed from "Family" to "Output"
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(heavy_pivot.index)
        ax.legend(title="Method")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, method in enumerate(['Direct', 'Hurdle']):
            if method in heavy_pivot.columns:
                offset = (i - 0.5) * width
                for j, value in enumerate(heavy_pivot[method]):
                    if not pd.isna(value):
                        ax.text(j + offset, value, f'{value:.3f}', 
                               ha='center', va='bottom', fontsize=8)
    
    # Plot Light models
    light_data = tier3_data[tier3_data["Mode"] == "Light"]
    if not light_data.empty:
        ax = axes[1]
        light_pivot = light_data.pivot_table(index='Family', columns='Method', values='Score', aggfunc='mean')
        
        # Plot with gray and white bars
        colors = [PALETTE['bar_gray'], PALETTE['bar_white']]
        edgecolors = ['black', 'black']
        
        # Plot bars
        x = np.arange(len(light_pivot.index))
        width = 0.35
        
        for i, method in enumerate(['Direct', 'Hurdle']):
            if method in light_pivot.columns:
                offset = (i - 0.5) * width
                ax.bar(x + offset, light_pivot[method], width, 
                      color=colors[i], edgecolor=edgecolors[i], 
                      linewidth=1, label=method)
        
        ax.set_title("Light Models (Tier 3)", fontweight='bold')
        ax.set_xlabel("Output")  
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(light_pivot.index)
        ax.legend(title="Method")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, method in enumerate(['Direct', 'Hurdle']):
            if method in light_pivot.columns:
                offset = (i - 0.5) * width
                for j, value in enumerate(light_pivot[method]):
                    if not pd.isna(value):
                        ax.text(j + offset, value, f'{value:.3f}', 
                               ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f"Tier 3 Performance Comparison: {metric_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    safe_name = metric_name.replace(" ", "_").lower()
    out_name = f"bar_comparison_{safe_name}.png"
    plt.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Bar Chart: {out_name}")

def plot_heatmap_simple(df, metric_name):
    """Simple heatmap visualization with inverted colors"""
    if df.empty:
        return
    
    # Create pivot table
    pivot_data = df.pivot_table(
        index=['Family', 'Mode', 'Method'],
        columns='Tier_Label',
        values='Score',
        aggfunc='mean'
    ).reset_index()
    
    # Sort for consistent display
    pivot_data = pivot_data.sort_values(['Family', 'Mode', 'Method'])
    
    # Extract just the score values for heatmap
    heatmap_values = pivot_data.iloc[:, 3:]  # Skip the first 3 columns (Output, Mode, Method)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap with inverted colormap (darker = worse)
    im = ax.imshow(heatmap_values.values, cmap='YlOrRd', aspect='auto')
    
    # Add text annotations
    for i in range(heatmap_values.shape[0]):
        for j in range(heatmap_values.shape[1]):
            value = heatmap_values.iloc[i, j]
            if not pd.isna(value):
                # Choose text color based on cell brightness
                cell_color = im.cmap(im.norm(value))
                # Calculate luminance
                luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
                text_color = 'white' if luminance < 0.5 else 'black'
                
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    # Set labels
    ax.set_xticks(range(len(heatmap_values.columns)))
    ax.set_yticks(range(len(pivot_data)))
    ax.set_xticklabels(heatmap_values.columns, rotation=45, ha='right')
    
    # Create row labels
    row_labels = []
    for _, row in pivot_data.iterrows():
        label = f"{row['Family']} {row['Mode']}\n{row['Method']}"
        row_labels.append(label)
    
    ax.set_yticklabels(row_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_name, fontweight='bold')
    
    plt.title(f"Performance Matrix: {metric_name}", fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    
    safe_name = metric_name.replace(" ", "_").lower()
    out_name = f"heatmap_{safe_name}.png"
    plt.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Heatmap: {out_name}")

# ==============================================================================
# MAIN EXECUTION (REMOVED SUMMARY PLOTS)
# ==============================================================================
if __name__ == "__main__":
    print("=== 00 CHAMPIONSHIP LADDER GENERATOR ===")
    
    for pass_cfg in ANALYSIS_PASSES:
        metric_key = pass_cfg["metric_key"]
        fname = pass_cfg["json_suffix"]
        pretty_name = pass_cfg["pretty_name"]
        csv_name = pass_cfg["csv_name"]
        
        print(f"\nPROCESSING: {pretty_name}")
        print("-" * 40)
        
        # 1. Load Data
        df = load_championship_table(metric_key, fname)
        
        if df.empty:
            print("No data found. Skipping...")
            continue
            
        # 2. Save CSV
        csv_path = OUTPUT_DIR / csv_name
        df.to_csv(csv_path, index=False)
        print(f"Saved Table: {csv_name}")
        
        # 3. Generate Visualizations (without summary plots)
        print(f"Generating Visualizations...")
        
        # Generate each plot with modified styling
        plot_simple_ladder(df, pretty_name)
        plot_grouped_bar(df, pretty_name)
        plot_heatmap_simple(df, pretty_name)
        
        print(f"{pretty_name} visualizations completed")
    
    print("ALL VISUALIZATIONS GENERATED")
    print(f"\nOutput directory: {OUTPUT_DIR}")
