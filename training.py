import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, PredefinedSplit, RandomizedSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler

from config import PLOTLY_CONFIG
from data_processing import merge_and_prepare_data, add_engineered_features
from model_registry import (
    AUTO_OPTIMIZE_PARAMS,
    AUTO_SEARCH_PARAMS,
    BAYES_PARAM_SPACES,
    COMMON_PARAM_GRID,
    MODELS,
    RANDOM_PARAM_SPACES,
    EXTENDED_COMMON_PARAM_GRID,
    EXTENDED_RANDOM_PARAM_SPACES,
    EXTENDED_BAYES_PARAM_SPACES,
    ALL_OPTIMIZABLE_PARAM_NAMES,
)
from plotting import plot_optimization_progress, plot_comparative_optimization_graphic

try:
    from skopt import BayesSearchCV
    _SKOPT_AVAILABLE = True
except ImportError:
    _SKOPT_AVAILABLE = False

try:
    import scipy.stats as stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _download_dataframe_buttons(df, base_filename, key_prefix):
    """Create stable CSV and Excel download buttons from an in-memory DataFrame."""
    if df is None or df.empty:
        return

    clean_df = df.copy()
    drop_cols = [c for c in clean_df.columns if "Values" in str(c) or str(c).lower() in {"y_test_values", "y_pred_values"}]
    clean_df = clean_df.drop(columns=drop_cols, errors="ignore")

    csv_data = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download analysis table (CSV)",
        data=csv_data,
        file_name=f"{base_filename}.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv_download",
        use_container_width=True,
    )

    try:
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            clean_df.to_excel(writer, index=False, sheet_name="Analysis Results")
        st.download_button(
            label="📥 Download analysis table (Excel)",
            data=buffer.getvalue(),
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_xlsx_download",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Excel export could not be prepared: {exc}")


def update_progress_bar_with_eta(progress_bar_obj, progress_percent, message):
    elapsed_time = time.time() - st.session_state.start_time
    
    remaining_str = "Estimating..."
    if progress_percent > 0:
        estimated_total = elapsed_time / (progress_percent / 100.0)
        remaining_time = estimated_total - elapsed_time
        
        if remaining_time < 0:
            remaining_str = "Completed."
        elif remaining_time < 60:
            remaining_str = f"~{remaining_time:.1f}s remaining"
        else:
            remaining_str = f"~{remaining_time / 60:.1f} min remaining"
    
    progress_bar_obj.progress(progress_percent, text=f"{message} (Elapsed: {elapsed_time:.1f}s | {remaining_str})")

# --- Main Functions (Function Definitions) ---
def add_evaluation_metrics_to_cv_results(
    search,
    base_model,
    X_train_val_processed,
    y_train_val,
    X_val_processed,
    y_val,
    X_test_processed,
    y_test,
    pds,
):
    """Compute per-trial VALIDATION RMSE for each hyperparameter set.

    Academic note:
    The test set is intentionally not evaluated per iteration.
    Optimization-process graphics must use validation/CV metrics only, while
    test metrics are reserved for final evaluation after selecting the best model.
    """
    params_list = search.cv_results_["params"]
    val_rmse_values = []

    train_mask = np.array([fold_id == -1 for fold_id in pds.test_fold])
    X_train_fold = X_train_val_processed[train_mask]
    y_train_fold = y_train_val[train_mask]

    for params in params_list:
        model = clone(base_model)
        model.set_params(**params)
        model.fit(X_train_fold, y_train_fold)

        y_val_pred = model.predict(X_val_processed)
        val_rmse_values.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))

    return val_rmse_values

def calculate_all_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten() 
    
    non_zero_indices = y_true != 0
    if not np.any(non_zero_indices):
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}
    y_true_filtered = y_true[non_zero_indices]
    y_pred_filtered = y_pred[non_zero_indices]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}




ENGINEERED_FEATURE_OPTIONS = {
    # Temporal behavior of the LCS PM2.5 signal
    "Lag 1": "raw_pollutant_lag1",
    "Lag 2": "raw_pollutant_lag2",
    "Lag 3": "raw_pollutant_lag3",
    "Rolling mean 3": "raw_pollutant_roll3",
    "Rolling mean 6": "raw_pollutant_roll6",
    "Rolling mean 12": "raw_pollutant_roll12",
    "Rolling mean 24": "raw_pollutant_roll24",
    "EMA 3": "raw_pollutant_ema3",
    "EMA 6": "raw_pollutant_ema6",
    "Difference 1": "raw_pollutant_diff1",
    "Absolute difference 1": "raw_pollutant_abs_diff1",
    "Spike score": "raw_pollutant_spike_score",
    "Rolling ratio": "raw_pollutant_roll_ratio",
    # LCS-environment correction terms: use sensor-side temp/humidity because
    # they represent the micro-environment experienced by the optical PM sensor.
    "Pollutant × Humidity": "raw_pollutant_x_humidity",
    "Pollutant × Temperature": "raw_pollutant_x_temp",
    "Humidity²": "raw_humidity_sq",
    "Humidity growth factor": "humidity_growth_factor",
    "Temperature × Humidity": "temp_x_humidity",
    "Dew point approximation": "dew_point_approx",
    # Temporal cycles and optional time flags
    "Hour sin": "hour_sin",
    "Hour cos": "hour_cos",
    "Month sin": "month_sin",
    "Month cos": "month_cos",
    "Weekend flag": "is_weekend",
    "Night flag": "is_night",
    "Rush-hour flag": "is_rush_hour",
}



def clean_feature_matrices(X_train_df, X_val_df, X_test_df):
    """Make feature matrices safe for every registered model.

    Some engineered features (lag/rolling/interactions) can introduce NaN or
    infinite values. Tree boosting libraries may tolerate some missing values,
    but linear models and kNN do not. Batch mode should therefore clean the
    same feature matrix once, using train-set medians only, so all 12 platform
    models can run instead of silently failing.
    """
    X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_val_df = X_val_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test_df = X_test_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    medians = X_train_df.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train_df = X_train_df.fillna(medians).fillna(0.0)
    X_val_df = X_val_df.fillna(medians).fillna(0.0)
    X_test_df = X_test_df.fillna(medians).fillna(0.0)

    return X_train_df, X_val_df, X_test_df


def get_auxiliary_feature_columns(available_columns):
    """Return auxiliary covariate columns preserved from the uploaded file."""
    return [c for c in available_columns if str(c).startswith("aux_")]


def _norm_feature_name(name):
    return str(name).lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _is_aux_column(col):
    return str(col).startswith("aux_")


def _is_external_meteorology_feature(col):
    """Location-specific/external meteorological covariates.

    This intentionally excludes raw_temp/raw_humidity from aux matching; those
    are added separately as sensor-side environmental variables.
    """
    name = _norm_feature_name(col)
    if not _is_aux_column(name):
        return False
    # Prevent duplicated sensor-side variables such as aux_raw_temp_y / aux_raw_hum_y.
    if "raw_temp" in name or "raw_hum" in name or "raw_humidity" in name:
        return False
    pollutant_exclusion_tokens = [
        "pm1", "pm10", "pm25", "pm2_5", "co2", "co_2", "co", "no2", "no_2",
        "nox", "so2", "so_2", "o3", "o_3", "ozone", "voc", "tvoc", "ch4",
    ]
    if any(tok in name for tok in pollutant_exclusion_tokens):
        return False
    meteorology_tokens = [
        "meteo_temp", "meteo_temperature", "temperature", "temp",
        "meteo_hum", "meteo_humidity", "humidity", "hum",
        "pressure", "press", "barometric",
        "wind", "wind_speed", "windspeed",
        "dew_point", "dewpoint", "dew",
    ]
    return any(tok in name for tok in meteorology_tokens)


def _sensor_environment_columns(available_columns):
    """Sensor-side environmental readings measured at the LCS micro-site."""
    cols = []
    for col in ["raw_temp", "raw_humidity"]:
        if col in available_columns and col not in cols:
            cols.append(col)
    return cols


def _external_meteorology_columns(available_columns):
    cols = []
    for col in available_columns:
        if _is_external_meteorology_feature(col) and col not in cols:
            cols.append(col)
    return cols


def _meteorology_columns(available_columns):
    """Calibration meteorology set: raw PM2.5 + sensor-side + external meteo.

    raw_pollutant is added in build_feature_list. This function only returns
    environmental covariates.
    """
    cols = []
    for col in _sensor_environment_columns(available_columns) + _external_meteorology_columns(available_columns):
        if col not in cols:
            cols.append(col)
    return cols


def _is_station_gaseous_pollutant_feature(col):
    """Station-based gaseous pollutant covariates only.

    PM-related auxiliary columns (station PM10, raw PM1, raw PM10, raw PM25) are
    intentionally excluded from the main calibration experiments to avoid strong
    proxy variables masking the LCS calibration problem.
    """
    name = _norm_feature_name(col)
    if not _is_aux_column(name):
        return False
    if any(tok in name for tok in ["pm1", "pm10", "pm25", "pm2_5"]):
        return False
    gas_tokens = ["station_co", "station_no2", "station_so2", "station_o3", "co", "no2", "no_2", "so2", "so_2", "o3", "o_3", "ozone"]
    return any(tok in name for tok in gas_tokens)


def _pollutant_columns(available_columns):
    """Station gaseous pollutants used as auxiliary calibration covariates."""
    cols = []
    for col in available_columns:
        if _is_station_gaseous_pollutant_feature(col) and col not in cols:
            cols.append(col)
    return cols


def _base_raw_column(available_columns):
    return ["raw_pollutant"] if "raw_pollutant" in available_columns else []


def _append_unique(base, extras):
    out = list(base)
    for col in extras:
        if col in extras and col not in out:
            out.append(col)
    return out


def _selected_engineered_columns(config, available_columns):
    """Return selected FE columns when the UI FE checkbox is active.

    The same FE gate is shared by every ablation mode. This means RAW_ONLY,
    METEOROLOGY_ONLY, POLLUTANTS_ONLY and FULL_MODEL can all be run either
    without FE or with the exact same user-selected FE set.
    """
    if not config.get("use_feature_engineering", False):
        return [], []

    selected_labels = config.get("selected_engineered_features", []) or []
    if not selected_labels:
        # Minimal tested default: strongest signal-to-complexity balance.
        selected_labels = [
            "Lag 1",
            "Rolling mean 3",
            "Pollutant × Humidity",
        ]

    selected_engineered = []
    missing_engineered = []
    for label in selected_labels:
        col = ENGINEERED_FEATURE_OPTIONS.get(label)
        if not col:
            continue
        if col in available_columns:
            if col not in selected_engineered:
                selected_engineered.append(col)
        else:
            missing_engineered.append(col)

    return selected_engineered, missing_engineered


def _append_engineered_features(features_to_use, feature_labels, selected_engineered, missing_engineered):
    """Append selected FE columns to a feature list and document what happened."""
    features_to_use = _append_unique(features_to_use, selected_engineered)

    if selected_engineered:
        feature_labels.append("Selected engineered features: " + ", ".join(selected_engineered))
    else:
        feature_labels.append("Selected engineered features: none")

    if missing_engineered:
        feature_labels.append("Selected engineered features not available after preprocessing: " + ", ".join(missing_engineered))

    return features_to_use, feature_labels


def build_feature_list(config, available_columns):
    """Return calibration feature columns for each ablation mode.

    All main ablation modes are calibration experiments: raw LCS PM2.5 remains
    the core input, and meteorological / gaseous pollutant groups are added
    according to the selected ablation mode. Feature engineering is controlled
    globally by the UI checkbox: when it is enabled, the selected FE columns are
    appended to every ablation mode; when it is disabled, FE is excluded from
    every ablation mode. PM proxy channels such as station_PM10, raw_PM1 and
    raw_PM10 are intentionally excluded from the main covariate groups.
    """
    ablation_mode = config.get("ablation_mode", "FULL_MODEL") or "FULL_MODEL"
    ablation_mode = str(ablation_mode).upper()

    raw_cols = _base_raw_column(available_columns)
    met_cols = _meteorology_columns(available_columns)
    gas_cols = _pollutant_columns(available_columns)
    selected_engineered, missing_engineered = _selected_engineered_columns(config, available_columns)

    if ablation_mode == "RAW_ONLY":
        features_to_use = list(raw_cols)
        feature_labels = [
            "Ablation: RAW_ONLY | Raw LCS PM2.5" + (" + selected FE" if selected_engineered else " only"),
            "Raw LCS PM2.5",
        ]
        return _append_engineered_features(features_to_use, feature_labels, selected_engineered, missing_engineered)

    if ablation_mode == "METEOROLOGY_ONLY":
        features_to_use = _append_unique(raw_cols, met_cols)
        feature_labels = [
            "Ablation: RAW_PLUS_METEOROLOGY" + (" + selected FE" if selected_engineered else ""),
            "Raw LCS PM2.5",
            "Sensor-side and location-specific meteorology: " + (", ".join(met_cols) if met_cols else "none"),
        ]
        return _append_engineered_features(features_to_use, feature_labels, selected_engineered, missing_engineered)

    if ablation_mode == "POLLUTANTS_ONLY":
        features_to_use = _append_unique(raw_cols, gas_cols)
        feature_labels = [
            "Ablation: RAW_PLUS_GASEOUS_POLLUTANTS" + (" + selected FE" if selected_engineered else ""),
            "Raw LCS PM2.5",
            "Station gaseous pollutants: " + (", ".join(gas_cols) if gas_cols else "none"),
        ]
        return _append_engineered_features(features_to_use, feature_labels, selected_engineered, missing_engineered)

    # FULL_MODEL: raw PM2.5 + meteorology + station gaseous pollutants,
    # with selected FE appended only when the UI checkbox is enabled.
    features_to_use = _append_unique(raw_cols, met_cols)
    features_to_use = _append_unique(features_to_use, gas_cols)
    feature_labels = [
        "Ablation: FULL_MODEL" + (" + selected FE" if selected_engineered else ""),
        "Raw LCS PM2.5",
        "Sensor-side and location-specific meteorology: " + (", ".join(met_cols) if met_cols else "none"),
        "Station gaseous pollutants: " + (", ".join(gas_cols) if gas_cols else "none"),
    ]

    return _append_engineered_features(features_to_use, feature_labels, selected_engineered, missing_engineered)

def get_param_space_for_run(model_name, optimization_mode, config):
    if optimization_mode == "Bayesian Optimization (skopt)":
        return EXTENDED_BAYES_PARAM_SPACES.get(model_name, BAYES_PARAM_SPACES.get(model_name, {}))
    if optimization_mode == "RandomizedSearchCV":
        return EXTENDED_RANDOM_PARAM_SPACES.get(model_name, RANDOM_PARAM_SPACES.get(model_name, {}))
    return EXTENDED_COMMON_PARAM_GRID.get(model_name, COMMON_PARAM_GRID.get(model_name, {}))


def filter_param_space_by_selection(model_name, optimization_mode, config):
    full_space = get_param_space_for_run(model_name, optimization_mode, config)
    if not full_space:
        return {}

    param_scope = config.get("param_scope", "Manual Selection")

    if param_scope == "Top 3 Parameters":
        selected_keys = AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3]
    elif param_scope == "Auto Search":
        selected_keys = AUTO_SEARCH_PARAMS.get(model_name, AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3])
    else:
        selected_keys = config.get("manual_params", []) or []

    return {key: full_space[key] for key in selected_keys if key in full_space}



def limit_gridsearch_to_n_combinations(param_grid, n_combinations=18):
    """Return a GridSearchCV-compatible grid with at most n explicit combinations."""
    if not param_grid:
        return param_grid

    all_combinations = list(ParameterGrid(param_grid))
    if len(all_combinations) <= n_combinations:
        return param_grid

    # Pick evenly spaced combinations so lower, middle and upper parts of the
    # original ranges are represented instead of taking only the first 18.
    indices = np.linspace(0, len(all_combinations) - 1, n_combinations, dtype=int)
    selected = [all_combinations[int(i)] for i in indices]
    return [{key: [value] for key, value in combo.items()} for combo in selected]


def _filter_param_space_for_comparison(model_name, opt_mode, config):
    """Use recommended top-3 parameters for fair comparison across optimization methods."""
    if opt_mode == "GridSearchCV":
        full_space = EXTENDED_COMMON_PARAM_GRID.get(model_name, COMMON_PARAM_GRID.get(model_name, {}))
    elif opt_mode == "RandomizedSearchCV":
        full_space = EXTENDED_RANDOM_PARAM_SPACES.get(model_name, RANDOM_PARAM_SPACES.get(model_name, {}))
    elif opt_mode == "Bayesian Optimization":
        full_space = EXTENDED_BAYES_PARAM_SPACES.get(model_name, BAYES_PARAM_SPACES.get(model_name, {}))
    else:
        return {}

    if not full_space:
        return {}

    top_3_keys = AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3]
    return {key: full_space[key] for key in top_3_keys if key in full_space}
def run_comparative_optimization_analysis(model_name, optimizers, config, pollutant_data, temp_data, hum_data, progress_bar_obj, status_text_container):
    results = []
    cv_results_dict = {}
    total_optimizers = len(optimizers)
    
    # Run data preparation once for all analyses
    status_text_container.info(f"[{model_name}] Data preparation for comparative analysis...")
    df_long_format, pollutant_unit, display_unit = merge_and_prepare_data(pollutant_data, temp_data, hum_data)
    
    resample_code = {'1 Minute (Original)': None, '2 Minutes': '2min','3 Minutes': '3min','5 Minutes': '5min', '10 Minutes': '10min', '15 Minutes': '15min', '30 Minutes': '30min', '60 Minutes': '60min'}.get(config['interval_label'])
    if resample_code:
        df_processed = df_long_format.set_index('timestamp').groupby('location').resample(resample_code).mean().dropna().reset_index()
    else:
        df_processed = df_long_format

    if df_processed.empty:
        status_text_container.warning(f"[{model_name}] No data found for the selected interval.")
        return pd.DataFrame(), {}, None, None

    X_full = df_processed; y_full = df_processed['reference_pollutant']
    test_perc = 100 - config['train_perc'] - config['val_perc']

    # Split data only once
    if config['split_method'] == "Time-Based":
        df_sorted = X_full.sort_values(by='timestamp', ascending=True)
        train_end = int(len(df_sorted) * (config['train_perc'] / 100)); val_end = train_end + int(len(df_sorted) * (config['val_perc'] / 100))
        train_df, val_df, test_df = df_sorted.iloc[:train_end], df_sorted.iloc[train_end:val_end], df_sorted.iloc[val_end:]
    else:
        groups = df_processed['location']
        gss_test = GroupShuffleSplit(n_splits=1, test_size=(test_perc/100), random_state=42)
        train_val_idx, test_idx = next(gss_test.split(df_processed, groups=groups))
        train_val_df = df_processed.iloc[train_val_idx]
        test_df = df_processed.iloc[test_idx]
        val_size_corrected = config['val_perc'] / (config['train_perc'] + config['val_perc'])
        groups_val = train_val_df['location']
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_corrected, random_state=42)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=groups_val))
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    # Leakage-safe feature engineering: create lag/rolling features AFTER the split.
    # This prevents validation/test rows from using temporal context across split boundaries.
    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)
    test_df = add_engineered_features(test_df)

    y_train = train_df['reference_pollutant']
    y_val = val_df['reference_pollutant']
    y_test = test_df['reference_pollutant']
    
    features_to_use_in_model, _feature_labels = build_feature_list(config, train_df.columns)

    X_train_dummies = pd.get_dummies(train_df[features_to_use_in_model])
    X_val_dummies = pd.get_dummies(val_df[features_to_use_in_model])
    X_test_dummies = pd.get_dummies(test_df[features_to_use_in_model])

    common_cols = sorted(set(X_train_dummies.columns) | set(X_val_dummies.columns) | set(X_test_dummies.columns))
    X_train_dummies = X_train_dummies.reindex(columns=common_cols, fill_value=0)
    X_val_dummies = X_val_dummies.reindex(columns=common_cols, fill_value=0)
    X_test_dummies = X_test_dummies.reindex(columns=common_cols, fill_value=0)
    X_train_dummies, X_val_dummies, X_test_dummies = clean_feature_matrices(
        X_train_dummies, X_val_dummies, X_test_dummies
    )
    model_feature_names = X_train_dummies.columns.tolist()
    
    scaler = None
    if model_name in ["k-Nearest Neighbors (kNN)", "Ridge Regression", "Lasso Regression", "ElasticNet Regression"]:
        scaler = StandardScaler()
        X_train_processed = pd.DataFrame(
            scaler.fit_transform(X_train_dummies),
            columns=common_cols,
            index=X_train_dummies.index,
        )
        X_val_processed = pd.DataFrame(
            scaler.transform(X_val_dummies),
            columns=common_cols,
            index=X_val_dummies.index,
        )
        X_test_processed = pd.DataFrame(
            scaler.transform(X_test_dummies),
            columns=common_cols,
            index=X_test_dummies.index,
        )
    else:
        X_train_processed, X_val_processed, X_test_processed = X_train_dummies, X_val_dummies, X_test_dummies

    y_train_val = np.concatenate((y_train, y_val))
    X_train_val_processed = pd.concat([X_train_processed, X_val_processed])
    split_index = [-1] * len(X_train_processed) + [0] * len(X_val_processed)
    pds = PredefinedSplit(test_fold=split_index)
    
    
    for i, opt_mode in enumerate(optimizers):
        if st.session_state.stop_auto_run:
            status_text_container.warning("Comparative analysis stopped by user.")
            break
            
        status_text_container.info(f"[{model_name}] Running {opt_mode} optimization...")
        progress_bar_obj.progress(int((i / total_optimizers) * 100), text=f"Running {opt_mode} for {model_name}...")
        
        start_time_opt = time.time()
        
        model = clone(MODELS[model_name])
        best_params = None
        cv_results = None

        try:
            param_space_for_run = {}
            if opt_mode == "GridSearchCV":
                param_space_for_run = _filter_param_space_for_comparison(model_name, opt_mode, config)
                if not param_space_for_run:
                    raise ValueError(f"No GridSearchCV parameters available for {model_name} after filtering.")
                grid_for_search = limit_gridsearch_to_n_combinations(param_space_for_run, 18)
                search = GridSearchCV(estimator=model, param_grid=grid_for_search, cv=pds, n_jobs=-1, scoring='r2', return_train_score=True)
                search.fit(X_train_val_processed, y_train_val)
                model = search.best_estimator_
                best_params = search.best_params_
                cv_results = search.cv_results_
                cv_results["val_rmse"] = add_evaluation_metrics_to_cv_results(
                    search=search,
                    base_model=clone(MODELS[model_name]),
                    X_train_val_processed=X_train_val_processed,
                    y_train_val=y_train_val,
                    X_val_processed=X_val_processed,
                    y_val=y_val,
                    X_test_processed=X_test_processed,
                    y_test=y_test,
                    pds=pds
                )
            
            elif opt_mode == "RandomizedSearchCV":
                if not _SCIPY_AVAILABLE:
                    st.warning(f"Skipping RandomizedSearchCV: `scipy` library is not available.")
                    continue
                param_space_for_run = _filter_param_space_for_comparison(model_name, opt_mode, config)
                if not param_space_for_run:
                    raise ValueError(f"No RandomizedSearchCV parameters available for {model_name} after filtering.")
                search = RandomizedSearchCV(estimator=model, param_distributions=param_space_for_run, n_iter=18, cv=pds, n_jobs=-1, scoring='r2', random_state=42, return_train_score=True)
                search.fit(X_train_val_processed, y_train_val)
                model = search.best_estimator_
                best_params = search.best_params_
                cv_results = search.cv_results_
                cv_results["val_rmse"] = add_evaluation_metrics_to_cv_results(
                    search=search,
                    base_model=clone(MODELS[model_name]),
                    X_train_val_processed=X_train_val_processed,
                    y_train_val=y_train_val,
                    X_val_processed=X_val_processed,
                    y_val=y_val,
                    X_test_processed=X_test_processed,
                    y_test=y_test,
                    pds=pds
                )

            elif opt_mode == "Bayesian Optimization":
                if not _SKOPT_AVAILABLE:
                    st.warning(f"Skipping Bayesian Optimization: `skopt` library is not available.")
                    continue
                param_space_for_run = _filter_param_space_for_comparison(model_name, opt_mode, config)
                if not param_space_for_run:
                    raise ValueError(f"No Bayesian parameters available for {model_name} after filtering.")
                search = BayesSearchCV(estimator=model, search_spaces=param_space_for_run, n_iter=18, cv=pds, n_jobs=1, scoring='r2', random_state=42, return_train_score=True)
                search.fit(X_train_val_processed, y_train_val)
                model = search.best_estimator_
                best_params = search.best_params_
                cv_results = search.cv_results_
                cv_results["val_rmse"] = add_evaluation_metrics_to_cv_results(
                    search=search,
                    base_model=clone(MODELS[model_name]),
                    X_train_val_processed=X_train_val_processed,
                    y_train_val=y_train_val,
                    X_val_processed=X_val_processed,
                    y_val=y_val,
                    X_test_processed=X_test_processed,
                    y_test=y_test,
                    pds=pds
                )
            
            else: # No optimization
                model.fit(X_train_processed, y_train)
                cv_results = {'params': [{}], 'mean_test_score': [calculate_all_metrics(y_val, model.predict(X_val_processed))['r2']], 'val_rmse': [calculate_all_metrics(y_val, model.predict(X_val_processed))['rmse']], 'mean_fit_time': [0], 'mean_train_score': [calculate_all_metrics(y_train, model.predict(X_train_processed))['r2']]}
                best_params = {}

            duration_opt = time.time() - start_time_opt
            
            # Use the best model from the search to predict on all sets
            y_train_pred = model.predict(X_train_processed)
            y_val_pred = model.predict(X_val_processed)
            y_test_pred = model.predict(X_test_processed)

            # Calculate full metrics for all sets
            train_metrics = calculate_all_metrics(y_train, y_train_pred)
            val_metrics = calculate_all_metrics(y_val, y_val_pred)
            test_metrics = calculate_all_metrics(y_test, y_test_pred)
            
            results.append({
                "Model_Name": model_name,
                "Optimization_Method": opt_mode,
                "Duration (s)": duration_opt,
                "Combinations_Tried": len(cv_results['params']),
                "Best_Params": str(best_params),
                "Train_R2": train_metrics['r2'],
                "Train_RMSE": train_metrics['rmse'],
                "Train_MAE": train_metrics['mae'],
                "Train_MAPE": train_metrics['mape'],
                "Val_R2": val_metrics['r2'],
                "Val_RMSE": val_metrics['rmse'],
                "Val_MAE": val_metrics['mae'],
                "Val_MAPE": val_metrics['mape'],
                "Test_R2": test_metrics['r2'],
                "Test_RMSE": test_metrics['rmse'],
                "Test_MAE": test_metrics['mae'],
                "Test_MAPE": test_metrics['mape'],
                # Stored only for the comparative calibration plot; hidden from display/download tables.
                "Y_Test_Values": np.asarray(y_test).tolist(),
                "Y_Pred_Values": np.asarray(y_test_pred).tolist()
            })
            
            # Store cv_results for plotting
            cv_results_dict[opt_mode] = pd.DataFrame(cv_results).copy(deep=True).copy(deep=True).copy(deep=True)

        except Exception as e:
            status_text_container.error(f"Error during {opt_mode} optimization for {model_name}: {e}")
            cv_results_dict[opt_mode] = None # Store None if it failed
    
    return pd.DataFrame(results), cv_results_dict, pollutant_unit, display_unit

def display_optimization_comparison_results(results_df, cv_results_dict, model_name, display_unit):
    st.header(f"📈 Optimization Comparison for {model_name}")

    if results_df.empty:
        st.warning("No results to display for comparative analysis.")
        return

    tab1, tab2, tab3 = st.tabs(["📊 Performance Comparison", "📈 Optimization Process", "🧩 Comparative Optimization Graphic"])

    with tab1:
        st.subheader("R² Score Comparison")
        col1, col2 = st.columns(2)
        with col1:
            fig_r2 = px.bar(results_df, x='Optimization_Method', y='Test_R2', color='Optimization_Method',
                            title=f"Test R² Scores by Optimization Method for {model_name}",
                            labels={'Test_R2': 'Test R² Score', 'Optimization_Method': 'Method'},
                            color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_r2.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_r2, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            st.subheader("Total Analysis Duration Comparison")
            fig_time = px.bar(results_df, x='Optimization_Method', y='Duration (s)', color='Optimization_Method',
                              title=f"Total Analysis Duration for {model_name}",
                              labels={'Duration (s)': 'Duration (s)', 'Optimization_Method': 'Method'},
                              color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_time.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_time, use_container_width=True, config=PLOTLY_CONFIG)
        
        st.markdown("---")
        st.subheader("Detailed Performance Comparison Table")
        
        # Reorder columns for better readability
        detailed_columns = [
            'Optimization_Method', 'Duration (s)', 'Combinations_Tried', 'Test_R2',
            'Test_RMSE', 'Test_MAE', 'Test_MAPE', 'Val_R2', 'Val_RMSE', 'Val_MAE',
            'Val_MAPE', 'Train_R2', 'Train_RMSE', 'Train_MAE', 'Train_MAPE',
            'Best_Params'
        ]
        
        # Ensure only existing columns are used to prevent errors if a metric is missing
        columns_to_display = [col for col in detailed_columns if col in results_df.columns]
        display_df = results_df[columns_to_display]
        
        st.dataframe(
            display_df.style.format({
                "Duration (s)": "{:.2f}",
                "Test_R2": "{:.4f}", "Test_RMSE": "{:.2f}", "Test_MAE": "{:.2f}", "Test_MAPE": "{:.2f}%",
                "Val_R2": "{:.4f}", "Val_RMSE": "{:.2f}", "Val_MAE": "{:.2f}", "Val_MAPE": "{:.2f}%",
                "Train_R2": "{:.4f}", "Train_RMSE": "{:.2f}", "Train_MAE": "{:.2f}", "Train_MAPE": "{:.2f}%"
            })
        )


        _download_dataframe_buttons(
            display_df,
            base_filename=f"{model_name}_optimization_comparison_results",
            key_prefix=f"{model_name}_optimization_comparison_results",
        )

    with tab2:
        plot_optimization_progress(cv_results_dict, model_name)

    with tab3:
        plot_comparative_optimization_graphic(
            results_df=results_df,
            cv_results_dict=cv_results_dict,
            model_name=model_name,
            auto_optimize_params=AUTO_OPTIMIZE_PARAMS,
        )

def run_model_analysis(current_config, pollutant_data, temp_data, hum_data, progress_bar_obj, status_text_container):
    """
    Performs a single model analysis with the specified configuration.
    This function will be used for both single runs and the automatic batch loop.
    """
    st.session_state.start_time = time.time()
    
    try:
        status_text_container.info(f"[{current_config['model_name']}] Initializing...")
        update_progress_bar_with_eta(progress_bar_obj, 0, f"[{current_config['model_name']}] Initializing...")

        df_long_format, pollutant_unit, display_unit = merge_and_prepare_data(pollutant_data, temp_data, hum_data)
        status_text_container.info(f"[{current_config['model_name']}] Reading and merging data...")
        update_progress_bar_with_eta(progress_bar_obj, 10, f"[{current_config['model_name']}] Reading and merging data...")
        
        resample_code = {'1 Minute (Original)': None, '2 Minutes': '2min','3 Minutes': '3min','5 Minutes': '5min', '10 Minutes': '10min', '15 Minutes': '15min', '30 Minutes': '30min','60 Minutes': '60min'}.get(current_config['interval_label'])
        if resample_code:
            status_text_container.info(f"[{current_config['model_name']}] Resampling data...")
            update_progress_bar_with_eta(progress_bar_obj, 20, f"[{current_config['model_name']}] Resampling data...")
            df_processed = df_long_format.set_index('timestamp').groupby('location').resample(resample_code).mean().dropna().reset_index()
        else:
            df_processed = df_long_format

        if df_processed.empty: 
            status_text_container.warning(f"[{current_config['model_name']}] No data found for the selected interval.")
            return None
        
        status_text_container.info(f"[{current_config['model_name']}] Splitting datasets...")
        update_progress_bar_with_eta(progress_bar_obj, 30, f"[{current_config['model_name']}] Splitting datasets...")
        X_full = df_processed; y_full = df_processed['reference_pollutant']
        test_perc = 100 - current_config['train_perc'] - current_config['val_perc']
        
        y_train, y_val, y_test = None, None, None 
        X_train, X_val, X_test = None, None, None 

        if current_config['split_method'] == "Time-Based":
            df_sorted = X_full.sort_values(by='timestamp', ascending=True)
            train_end = int(len(df_sorted) * (current_config['train_perc'] / 100)); val_end = train_end + int(len(df_sorted) * (current_config['val_perc'] / 100))
            train_df, val_df, test_df = df_sorted.iloc[:train_end], df_sorted.iloc[train_end:val_end], df_sorted.iloc[val_end:]
            
            y_train = train_df['reference_pollutant']
            y_val = val_df['reference_pollutant']
            y_test = test_df['reference_pollutant']
            
            X_train = train_df.drop(columns=['reference_pollutant'])
            X_val = val_df.drop(columns=['reference_pollutant'])
            X_test = test_df.drop(columns=['reference_pollutant'])

        else: # For 'Random' option, LOCATION-BASED SAFE method
            groups = df_processed['location']

            gss_test = GroupShuffleSplit(n_splits=1, test_size=(test_perc/100), random_state=42)
            train_val_idx, test_idx = next(gss_test.split(df_processed, groups=groups))

            train_val_df = df_processed.iloc[train_val_idx]
            test_df = df_processed.iloc[test_idx]

            val_size_corrected = current_config['val_perc'] / (current_config['train_perc'] + current_config['val_perc'])
            groups_val = train_val_df['location']
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_corrected, random_state=42)
            train_idx, val_idx = next(gss_val.split(train_val_df, groups=groups_val))
            
            train_df = train_val_df.iloc[train_idx]
            val_df = train_val_df.iloc[val_idx]

            y_train = train_df['reference_pollutant']
            y_val = val_df['reference_pollutant']
            y_test = test_df['reference_pollutant']

            X_train = train_df.drop(columns=['reference_pollutant'])
            X_val = val_df.drop(columns=['reference_pollutant'])
            X_test = test_df.drop(columns=['reference_pollutant'])
        
        # Leakage-safe feature engineering: create lag/rolling features AFTER the split.
        # This is the strict academic setting. Validation/test feature values are
        # computed only within their own split, not from previous splits.
        train_df = add_engineered_features(train_df)
        val_df = add_engineered_features(val_df)
        test_df = add_engineered_features(test_df)

        y_train = train_df['reference_pollutant']
        y_val = val_df['reference_pollutant']
        y_test = test_df['reference_pollutant']
        X_train = train_df.drop(columns=['reference_pollutant'])
        X_val = val_df.drop(columns=['reference_pollutant'])
        X_test = test_df.drop(columns=['reference_pollutant'])

        status_text_container.info(f"[{current_config['model_name']}] Preparing features for the model...")
        update_progress_bar_with_eta(progress_bar_obj, 40, f"[{current_config['model_name']}] Preparing features for the model...")
        
        features_to_use_in_model, feature_info_for_display = build_feature_list(current_config, X_train.columns)

        X_train_dummies = pd.get_dummies(X_train[features_to_use_in_model])
        X_val_dummies = pd.get_dummies(X_val[features_to_use_in_model])
        X_test_dummies = pd.get_dummies(X_test[features_to_use_in_model])
        
        common_cols = sorted(set(X_train_dummies.columns) | set(X_val_dummies.columns) | set(X_test_dummies.columns))
        X_train_dummies = X_train_dummies.reindex(columns=common_cols, fill_value=0)
        X_val_dummies = X_val_dummies.reindex(columns=common_cols, fill_value=0)
        X_test_dummies = X_test_dummies.reindex(columns=common_cols, fill_value=0)
        X_train_dummies, X_val_dummies, X_test_dummies = clean_feature_matrices(
            X_train_dummies, X_val_dummies, X_test_dummies
        )
        
        model_feature_names = X_train_dummies.columns.tolist()

        scaler = None
        # Scaling is important for these models
        if current_config['model_name'] in ["k-Nearest Neighbors (kNN)", "Ridge Regression", "Lasso Regression", "ElasticNet Regression"]:
            scaler = StandardScaler()
            X_train_processed = pd.DataFrame(
                scaler.fit_transform(X_train_dummies),
                columns=common_cols,
                index=X_train_dummies.index,
            )
            X_val_processed = pd.DataFrame(
                scaler.transform(X_val_dummies),
                columns=common_cols,
                index=X_val_dummies.index,
            )
            X_test_processed = pd.DataFrame(
                scaler.transform(X_test_dummies),
                columns=common_cols,
                index=X_test_dummies.index,
            )
        else:
            X_train_processed, X_val_processed, X_test_processed = X_train_dummies, X_val_dummies, X_test_dummies
        
        model = clone(MODELS[current_config['model_name']])
        best_params = None 
        cv_results = None
        
        param_space_for_run = {}

        if current_config['optimize']:
            y_train_val = np.concatenate((y_train, y_val))
            X_train_val_processed = pd.concat([X_train_processed, X_val_processed])
            
            split_index = [-1] * len(X_train_processed) + [0] * len(X_val_processed)
            pds = PredefinedSplit(test_fold=split_index)
            
            # Select the correct parameter space based on optimization mode
            if current_config['optimization_mode'] == "Bayesian Optimization (skopt)":
                if _SKOPT_AVAILABLE:
                    param_space_for_run = EXTENDED_BAYES_PARAM_SPACES.get(current_config['model_name'], BAYES_PARAM_SPACES.get(current_config['model_name'], {}))
                else:
                    status_text_container.warning(f"[{current_config['model_name']}] `skopt` library not found. Reverting to default parameters.")
                    current_config['optimize'] = False
            
            elif current_config['optimization_mode'] == "RandomizedSearchCV":
                if _SCIPY_AVAILABLE:
                    param_space_for_run = EXTENDED_RANDOM_PARAM_SPACES.get(current_config['model_name'], RANDOM_PARAM_SPACES.get(current_config['model_name'], {}))
                else:
                    status_text_container.warning(f"[{current_config['model_name']}] `scipy` library not found. Reverting to default parameters.")
                    current_config['optimize'] = False

            elif current_config['optimization_mode'] == "Automatic Optimization (Top 3 Params)":
                top_3_keys = AUTO_OPTIMIZE_PARAMS.get(current_config['model_name'], [])
                full_params_grid = EXTENDED_COMMON_PARAM_GRID.get(current_config['model_name'], COMMON_PARAM_GRID.get(current_config['model_name'], {}))
                for key in top_3_keys:
                    if key in full_params_grid:
                        param_space_for_run[key] = full_params_grid[key]
                # param_scope is implicitly 'Top 3' here
                current_config['param_scope'] = "Top 3 Parameters"

            else: # Manual Optimization (GridSearch)
                full_params_grid = EXTENDED_COMMON_PARAM_GRID.get(current_config['model_name'], COMMON_PARAM_GRID.get(current_config['model_name'], {}))
                param_space_for_run = {key: full_params_grid[key] for key in current_config['manual_params'] if key in full_params_grid}

            # --- NEW FEATURE: DYNAMIC PARAMETER SCOPE ---
            # If "All Parameters" is selected, we do nothing as param_space_for_run is already populated.
            # If "Top 3 Parameters" is selected for a non-auto mode, we filter the list.
            if current_config.get('param_scope') == "Top 3 Parameters" and current_config['optimization_mode'] not in ["Automatic Optimization (Top 3 Params)"]:
                top_3_keys = AUTO_OPTIMIZE_PARAMS.get(current_config['model_name'], [])
                filtered_param_space = {}
                for key in top_3_keys:
                    if key in param_space_for_run:
                        filtered_param_space[key] = param_space_for_run[key]
                param_space_for_run = filtered_param_space

            elif current_config.get('param_scope') == "Auto Search":
                auto_keys = AUTO_SEARCH_PARAMS.get(current_config['model_name'], AUTO_OPTIMIZE_PARAMS.get(current_config['model_name'], [])[:3])
                param_space_for_run = {key: param_space_for_run[key] for key in auto_keys if key in param_space_for_run}
            elif current_config.get('param_scope') == "Manual Selection":
                manual_keys = current_config.get('manual_params', []) or []
                param_space_for_run = {key: param_space_for_run[key] for key in manual_keys if key in param_space_for_run}

            if not param_space_for_run:
                 status_text_container.warning(f"[{current_config['model_name']}] No parameters selected for optimization. Skipping and training with default parameters.")
                 current_config['optimize'] = False
            
            if current_config['optimize']:
                if current_config['optimization_mode'] == "Bayesian Optimization (skopt)":
                    status_text_container.info(f"[{current_config['model_name']}] Running Bayesian Optimization (BayesSearchCV)...")
                    search = BayesSearchCV(
                        estimator=model,
                        search_spaces=param_space_for_run,
                        cv=pds,
                        n_iter=current_config.get('n_iter_bayes', 18),
                        n_jobs=1,
                        scoring='r2',
                        random_state=42,
                        return_train_score=True
                    )
                elif current_config['optimization_mode'] == "RandomizedSearchCV":
                    status_text_container.info(f"[{current_config['model_name']}] Running RandomizedSearchCV...")
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_space_for_run,
                        n_iter=current_config.get('n_iter_random', 18),
                        cv=pds,
                        n_jobs=-1,
                        scoring='r2',
                        random_state=42,
                        return_train_score=True
                    )
                else: # GridSearchCV
                    status_text_container.info(f"[{current_config['model_name']}] Running GridSearchCV optimization...")
                    if current_config.get('param_scope') == "Manual Selection":
                        # Manual Selection is exploratory: Grid Search evaluates the full selected parameter space.
                        grid_for_search = param_space_for_run
                    else:
                        # Fair comparison modes cap Grid Search at 18 combinations.
                        grid_for_search = limit_gridsearch_to_n_combinations(param_space_for_run, 18)
                    search = GridSearchCV(estimator=model, param_grid=grid_for_search, cv=pds, n_jobs=-1, scoring='r2', return_train_score=True)
                
                update_progress_bar_with_eta(progress_bar_obj, 65, f"[{current_config['model_name']}] Optimization running...") 
                search.fit(X_train_val_processed, y_train_val)
                best_params = search.best_params_
                cv_results = search.cv_results_

                # Rebuild and refit the final model explicitly to avoid unfitted-estimator issues.
                model = clone(MODELS[current_config['model_name']])
                model.set_params(**best_params)
                model.fit(X_train_val_processed, y_train_val)
            else:
                best_params = {}
                cv_results = None
                model = clone(MODELS[current_config['model_name']])
                model.fit(X_train_processed, y_train)
        else:
            best_params = {}
            cv_results = None
            model = clone(MODELS[current_config['model_name']])
            model.fit(X_train_processed, y_train)

        # Optional final refit: after choosing hyperparameters on validation, fit on
        # train+validation before evaluating the untouched test period. This is
        # already standard during optimization above; the switch also enables it
        # for non-optimized/default-parameter runs.
        X_train_val_processed = pd.concat([X_train_processed, X_val_processed])
        y_train_val = np.concatenate((y_train, y_val))
        if current_config.get('final_fit_on_train_val', True) and not current_config.get('optimize', False):
            model = clone(MODELS[current_config['model_name']])
            if best_params:
                model.set_params(**best_params)
            model.fit(X_train_val_processed, y_train_val)

        status_text_container.info(f"[{current_config['model_name']}] Generating reports...")
        update_progress_bar_with_eta(progress_bar_obj, 90, f"[{current_config['model_name']}] Generating reports...")
        
        # Final safety net: ensure model is fitted before prediction.
        try:
            y_train_pred = model.predict(X_train_processed).flatten()
            y_val_pred = model.predict(X_val_processed).flatten()
            y_test_pred = model.predict(X_test_processed).flatten()
        except Exception as pred_error:
            if current_config.get('optimize', False) and best_params:
                model = clone(MODELS[current_config['model_name']])
                model.set_params(**best_params)
                model.fit(X_train_val_processed, y_train_val)
            else:
                model = clone(MODELS[current_config['model_name']])
                model.fit(X_train_processed, y_train)

            y_train_pred = model.predict(X_train_processed).flatten()
            y_val_pred = model.predict(X_val_processed).flatten()
            y_test_pred = model.predict(X_test_processed).flatten()
        
        end_time = time.time()
        duration = end_time - st.session_state.start_time
        st.session_state.analysis_duration = duration

        detailed_loc_results = []
        unique_locations = df_processed['location'].unique()

        for loc in sorted(unique_locations):
            train_indices_loc = X_train[X_train['location'] == loc].index
            if not train_indices_loc.empty:
                y_pred_train_loc = pd.Series(y_train_pred, index=y_train.index).loc[train_indices_loc]
                metrics = calculate_all_metrics(y_train.loc[train_indices_loc], y_pred_train_loc)
                metrics.update({'Location': loc, 'Set': 'Training'})
                detailed_loc_results.append(metrics)
            val_indices_loc = X_val[X_val['location'] == loc].index
            if not val_indices_loc.empty:
                y_pred_val_loc = pd.Series(y_val_pred, index=y_val.index).loc[val_indices_loc]
                metrics = calculate_all_metrics(y_val.loc[val_indices_loc], y_pred_val_loc)
                metrics.update({'Location': loc, 'Set': 'Validation'})
                detailed_loc_results.append(metrics)
            test_indices_loc = X_test[X_test['location'] == loc].index
            if not test_indices_loc.empty:
                y_pred_test_loc = pd.Series(y_test_pred, index=y_test.index).loc[test_indices_loc]
                metrics = calculate_all_metrics(y_test.loc[test_indices_loc], y_pred_test_loc)
                metrics.update({'Location': loc, 'Set': 'Test'})
                detailed_loc_results.append(metrics)
        
        detailed_loc_metrics_df = pd.DataFrame(detailed_loc_results)

        analysis_results = { 
            "y_test": y_test, "y_pred_series": pd.Series(y_test_pred, index=y_test.index), 
            "X_test": X_test, "df_processed": df_processed, "model_name": current_config['model_name'], 
            "interval": current_config['interval_label'], "split_method": current_config['split_method'], 
            "train_perc": current_config['train_perc'], "val_perc": current_config['val_perc'], "test_perc": test_perc, 
            "features": feature_info_for_display, "pollutant_unit": pollutant_unit, "display_unit": display_unit, 
            "train_metrics": calculate_all_metrics(y_train, y_train_pred),
            "val_metrics": calculate_all_metrics(y_val, y_val_pred),
            "test_metrics": calculate_all_metrics(y_test, y_test_pred),
            "val_test_metrics": calculate_all_metrics(
                np.concatenate((y_val, y_test)),
                np.concatenate((y_val_pred, y_test_pred))
            ),
            "final_fit_on_train_val": current_config.get('final_fit_on_train_val', True),
            "optimized": current_config['optimize'], "best_params": best_params, "cv_results_": cv_results, 
            "model": model, "feature_names_for_model": model_feature_names, "analysis_duration": duration,
            "detailed_loc_metrics_df": detailed_loc_metrics_df,
            "y_train": y_train, "y_train_pred_series": pd.Series(y_train_pred, index=y_train.index),
            "y_val": y_val, "y_val_pred_series": pd.Series(y_val_pred, index=y_val.index),
            "X_train": X_train, "X_val": X_val,
            "X_test_processed_df": pd.DataFrame(X_test_processed, index=y_test.index, columns=model_feature_names),
            "optimization_mode": current_config['optimization_mode'],
            "param_scope": current_config.get('param_scope')
        }
        
        update_progress_bar_with_eta(progress_bar_obj, 100, f"[{current_config['model_name']}] Analysis Complete!")
        return analysis_results

    except Exception as e:
        status_text_container.error(f"[{current_config['model_name']}] ERROR: An error occurred during analysis: {e}")
        progress_bar_obj.empty()
        return None
