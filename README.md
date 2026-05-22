# AQ-MultiCal: Air Quality Multi-Model Calibration Platform (v2)

An advanced, no-code interactive web application developed using Streamlit for the calibration and comparative evaluation of Low-Cost Air Quality Sensors (LCS). 

This platform framework is engineered to systematically calibrate raw sensor outputs against reference instruments using 10 distinct machine learning algorithms, integrated hyperparameter optimization loops, and leakage-safe feature engineering layers.

## Data Sets and Variable Specifications

The repository includes a set of pre-processed files reflecting real-world co-location experiment campaigns. These datasets can be fed directly into the application control panel for instant replication of calibration tests.

### 1. Full_data.csv (Main Campaign Dataset)
This file serves as the core evaluation matrix. It tracks 3,672 hours of co-located observations and compiles reference measurements alongside low-cost sensor parameters and auxiliary synoptic variables (delimited by semicolon):
- timestamp: Temporal tracking formatted as DD.MM.YYYY HH:MM.
- reference_PM25: High-accuracy reference monitor PM2.5 concentration (ug/m3).
- raw_PM25: Uncalibrated optical low-cost sensor PM2.5 response (ug/m3).
- raw_temp: Micro-climatic temperature channel recorded inside the LCS chassis (Celsius).
- raw_humidity: Relative humidity channel recorded inside the LCS optical chamber (%RH).
- station_CO: Co-located regulatory station Carbon Monoxide concentration (ppb).
- station_NO2: Co-located regulatory station Nitrogen Dioxide concentration (ppb).
- station_SO2: Co-located regulatory station Sulfur Dioxide concentration (ug/m3).
- station_O3: Co-located regulatory station Ground-level Ozone concentration (ug/m3).
- pressure_hPa: Ambient synoptic barometric pressure (hPa).
- wind_speed_ms: Local surface wind speed (m/s).
- station_Radiation: Solar radiation energy flux (W/m2).
- dew_point: Calculated atmospheric dew point temperature (Celsius).

### 2. raw_temp.csv (Dedicated Temperature Metrology File)
- Zaman: Standard temporal tracking indexed as YYYY-MM-DD HH:MM:SS.
- Reference_Sicaklik: Certified ambient reference temperature (Celsius).
- RAW_TEMPERATURE: Low-cost sensor on-board temperature registry (Celsius).

### 3. raw_humidity.csv (Dedicated Humidity Metrology File)
- Zaman: Standard temporal tracking indexed as YYYY-MM-DD HH:MM:SS.
- reference_Nem: Certified ambient reference relative humidity (%RH).
- RAW_HUMIDITY: Low-cost sensor on-board relative humidity registry (%RH).

---

## Core Architecture and System Features

### 1. Modeling Layer (10 Regressors)
The framework integrates a diverse set of linear, instance-based, tree-based, and boosting regression models:
- Linear Models: Linear Regression, Ridge Regression, Lasso Regression, ElasticNet Regression.
- Instance/Tree Models: k-Nearest Neighbors (kNN), Decision Tree, Random Forest.
- Boosting Frameworks: Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost.

### 2. Hyperparameter Optimization Engines
Supports robust automated search strategies constrained to a fair, computationally aligned 18-iteration limit across all methods to ensure rigorous comparative efficiency analysis:
- GridSearchCV: Systematic grid exploration.
- RandomizedSearchCV: Stochastic hyperparameter sampling.
- BayesSearchCV (Bayesian Optimization): Sequential model-based optimization using Gaussian Processes (skopt).

### 3. Rigorous Evaluation and Ablation Study Modes
Quantifies the incremental improvement of auxiliary environmental and gaseous covariates tensor fields through strict ablation experiments:
- RAW_ONLY: Calibrates using raw LCS target pollutant signal only.
- METEOROLOGY_ONLY: Adds sensor-side and location-specific meteorological covariates (raw_temp, raw_humidity, pressure_hPa, wind_speed_ms, station_Radiation, dew_point).
- POLLUTANTS_ONLY: Adds co-located station-based gaseous pollutants (station_CO, station_NO2, station_SO2, station_O3).
- FULL_MODEL: Combines all available feature matrices.

### 4. Methodological Safeguards
- Data Leakage Prevention: Lag, rolling windows, and interaction features are computed strictly after cross-dataset splitting.
- Location-Based Cross-Validation: Leverages GroupShuffleSplit for random validation strategies to guarantee spatial evaluation consistency.
- Clean Feature Matrices: Automated numerical casting and train-median imputation to prevent linear/kNN model execution failures without masking underlying sensor anomalies.

## Repository Structure

```text
AQ-MultiCal_v2/
├── Full_data.csv           # Main calibration database containing 3672 campaign records
├── raw_temp.csv            # Co-located secondary registry for temperature cross-examination
├── raw_humidity.csv        # Co-located secondary registry for humidity cross-examination
├── app_refactor_1.py       # Main Streamlit UI panel & multi-tab coordination hub
├── config.py               # Global plotting styles, hex colors, and pollutant definitions
├── data_processing.py      # Robust auto-delimiter CSV parser & datetime format normalizer
├── model_registry.py       # Algorithmic constraints, model definitions, and parameter spaces
├── training.py             # Leakage-safe data splitting, training loops, and optimization
├── plotting.py             # Publication-ready Matplotlib, Seaborn, and Plotly graphics
├── .gitignore              # Pre-configured file exclusions (caches, local logs, dataset extensions)
└── requirements.txt        # Verified environment deployment dependencies
