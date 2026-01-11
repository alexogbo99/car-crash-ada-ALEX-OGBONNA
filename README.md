# car-crash-ada-ALEX-OGBONNA

# Spatiotemporal Crash Risk Prediction in NYC (500m Grid, 6-hour Bins)

## Overview / Abstract

This project builds a spatiotemporal forecasting pipeline to predict crash risk in New York City on a **500m × 500m grid** across **four 6-hour time bins**, using a panel dataset spanning **2021–2024**.  
The modeling challenge is extreme sparsity at local resolution (for the short-horizon target, the vast majority of cell–time observations contain zero crashes). To address this, we compare:

- **Direct count regression**: a single model predicts the crash count directly.
- **Two-stage Hurdle modeling**: a probabilistic **Gatekeeper** estimates crash occurrence `P(y>0)`, and a **Regressor** estimates expected intensity `E[y | y>0)`, with final prediction `\hat{y} = P(y>0) \times E[y \mid y>0]`.

We also evaluate the value of richer context via cumulative feature tiers:
Tier 1 (structural baseline), Tier 2 (weather context), and Tier 3 (full context including traffic and neighborhood features).

---

## ⚠️ Important: Data Availability
**Note to Graders/TAs:**
The dataset files for this project are too large for GitHub (over 100MB).
The full datasets and the presentation video are hosted on **Google Drive**. 
> **The link to the Google Drive folder has been sent directly to the Teacher Assistant via email.**

---

## Data Sources

This project integrates multiple public datasets to construct the crash-risk panel:

- **NYC Motor Vehicle Collisions (Crashes)**: crash events (time + geolocation).
- **Traffic volume signals**: sensor-based traffic exposure proxies interpolated to the grid.
- **Weather**: hourly historical weather used to build features and leakage-safe pseudo-forecasts.
- **Street network / infrastructure**: OpenStreetMap-derived road network attributes plus transit / POI densities.

---

## Project Structure

A simplified view of the repository layout:

```text
Project_ADA_Crash_Car_NYC_2025/
│──config.py
│──pipeline_utils.py
│──requirements.txt
├── data/
│   ├── raw/                     # Raw downloads (usually git-ignored)
│   ├── intermediate/            # Intermediate artifacts
│   └── processed/               # Train/Val/Test parquet datasets
├── scripts/
│   ├── raw_scripts/             # Ingestion & harmonization
│   ├── static_scripts/          # Static spatial features on the grid
│   ├── dynamic_script/          # Panel construction + pseudo-forecasts + targets
│   ├── features_scripts/        # Feature tier configs + feature screening
│   ├── run_experiment_scripts/  # Training pipelines (Direct + Hurdle; Light + Heavy)
│   └── visualisation_analysis_scripts/ # Evaluation & diagnostics
└── results/
    ├── *_Direct_*/              # Direct experiments
    ├── *_Hurdle_*/              # Hurdle experiments (gatekeeper + regressor)
    └── feature_screening/       # Screening outputs

```

## Installation & Requirements

To reproduce the analysis, it is recommended to run this project in a standard Python 3 environment.

**Dependencies:**
The required Python libraries are listed in `requirements.txt`.
You can install them using:

```bash
pip install -r requirements.txt

```

## Usage / Pipeline

The workflow is organized into sequential stages corresponding to the folders in `scripts/`.
**Crucial:** Inside each folder, scripts are prefixed with numbers (e.g., `01_...`, `02_...`). Please run them in strictly numerical order.

### 1. Data Ingestion
**Folder:** `scripts/raw_scripts/`
* Run scripts `01` through `03`.
* **Action:** Downloads raw data and standardizes formats for crashes, traffic, and weather.

### 2. Static Feature Engineering
**Folder:** `scripts/static_scripts/`
* Run scripts `01` through `08`.
* **Action:** Builds the active 500m grid and computes time-invariant features (network structure, infrastructure densities, baseline traffic exposure proxies).

### 3. Dynamic Panel Construction
**Folder:** `scripts/dynamic_scripts/`
* Run scripts `01` through `04`.
* **Action:** Creates the spatiotemporal panel (cell_id, date, time_bin), attaches history/weather features, and generates leakage-safe pseudo-forecasts.

### 4. Feature Screening & Tiering
**Folder:** `scripts/features_scripts/`
* Run all 6 scripts in order.
* **Action:** Runs screening utilities to refine the Tier 1, Tier 2, and Tier 3 feature sets.
* *Note:* Includes a specific sanity check script to ensure data integrity before training.

### 5. Training Experiments
**Folder:** `scripts/run_experiment_scripts/`
This folder contains separate pipelines for the C1 (Short Horizon) and C7 (Medium Horizon) targets.
* **For C1:** Run the 10 scripts labeled for C1 (Gatekeepers → Train/Val → Testing).
* **For C7:** Run the 10 scripts labeled for C7 (Gatekeepers → Train/Val → Testing).
* **Action:** Trains and evaluates both Direct and Hurdle models across all feature tiers.

### 6. Evaluation & Visualization
**Folder:** `scripts/visualisation_analysis_scripts/`
* Run scripts `01` through `07`.
* **Action:** Generates the "championship ladder," feature importance plots, and residual diagnostics found in the final report.

Modeling strategies
Direct regression: learn \hat{y} = f(\mathbf{x})

Hurdle: learn \hat{y} = P(y>0 \mid \mathbf{x}) \times E[y \mid y>0, \mathbf{x}]

Key Results
Across both targets (C1 and C7), both execution modes (Heavy and Light), and all feature tiers, the Hurdle approach consistently outperforms Direct regression on the test set. This supports the idea that explicitly separating occurrence from positive intensity is beneficial under extreme zero inflation.

Tier 3 champions (best available information)
Using Tier 3 (Full) as the most informative setting, the relative gains of Hurdle over Direct are substantial:

C1 (Heavy, Tier 3):

Direct: RMSE ≈ 0.297

Hurdle: RMSE ≈ 0.119

Result: Largest relative improvement occurs on C1.

C7 (Heavy, Tier 3):

Direct: RMSE ≈ 0.584

Hurdle: RMSE ≈ 0.320

Authors & Acknowledgments
Author: Alex Ogbonna

University: University of Lausanne

Course: Advanced Data Analysis

Semester: Autumn 2025
