import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit # Using TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from xgboost import XGBRegressor
import warnings
import time
from sklearn.preprocessing import StandardScaler 

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Helper Functions ---

def create_lag_features(data, target_col, lag):
    """Creates lag features for a specified column."""
    df_lag = data.copy()
    for i in range(1, lag + 1):
        col_name = f'{target_col}_lag_{i}'
        if col_name in df_lag.columns and i == 1:
             pass
        df_lag[col_name] = df_lag[target_col].shift(i)
    return df_lag

def custom_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error, ignoring zeros in y_true.
    Returns result multiplied by 100 (percentage).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    y_true_filt = y_true[non_zero_mask]
    y_pred_filt = y_pred[non_zero_mask]
    if len(y_true_filt) == 0:
        return 0.0 if np.all(y_pred_filt == 0) else np.inf
    return np.mean(np.abs((y_true_filt - y_pred_filt) / y_true_filt)) * 100

def rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Main Script ---

# Load Data
try:
    df_full = pd.read_csv('data/merged_file1.csv')
except FileNotFoundError:
    print("Error: Data file 'project/data/merged_file.csv' not found.")
    exit()

# Preprocessing & Feature Engineering
df_processed = create_lag_features(df_full, target_col='ConfirmedCases', lag=3)
df_processed = df_processed.dropna().reset_index(drop=True)

if 'YearWeek' in df_processed.columns:
    df_processed['YearWeek'] = df_processed['YearWeek'].astype(str)
    df_processed['Year'] = df_processed['YearWeek'].str[:4].astype(int)
    df_processed['Week'] = df_processed['YearWeek'].str[4:].astype(int)
else:
    print("Warning: 'YearWeek' column not found. Cannot extract Year/Week.")

target_column = 'ConfirmedCases'

# --- Define FIXED Feature Sets ---
svm_fixed_features = ['ConfirmedCases_lag_1', 'ExcludedCases', 'ConfirmedCases_lag_2', 'Year', 'ConfirmedCases_lag_3', 'AverageTemperature']
xgb_fixed_features = ['ConfirmedCases_lag_1', 'AverageTemperature', 'Week', 'ConfirmedCases_lag_3', 'ExcludedCases']

# Verify all features exist
all_needed_features = list(set(svm_fixed_features + xgb_fixed_features + [target_column]))
missing_features = [f for f in all_needed_features if f not in df_processed.columns]
if missing_features:
    print(f"Error: The following required features are missing: {missing_features}")
    print(f"Available columns: {df_processed.columns.tolist()}")
    exit()

print(f"Using fixed features for SVM: {svm_fixed_features}")
print(f"Using fixed features for XGBoost: {xgb_fixed_features}")

# Prepare data
X_all = df_processed[list(set(svm_fixed_features + xgb_fixed_features))]
y = df_processed[target_column]
N_samples = len(X_all) # Total number of samples after preprocessing

# --- Define FIXED Best Hyperparameters ---
svm_best_params = {
    'C': 10,
    'epsilon': 0.001,
    'gamma': 0.001,
    'kernel': 'rbf'
}
xgb_best_params = {
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 200,
    'objective': 'reg:squarederror',
    'random_state': 42
}
print("\nUsing fixed hyperparameters for SVM:", svm_best_params)
print("Using fixed hyperparameters for XGBoost:", xgb_best_params)

# --- Model Definitions (Instantiated Later) ---
models_config = {
    "SVM": {
        "model_class": SVR,
        "features": svm_fixed_features,
        "params": svm_best_params
    },
    "XGBoost": {
        "model_class": XGBRegressor,
        "features": xgb_fixed_features,
        "params": xgb_best_params
    }
}

# --- Time Series Cross-Validation Setup ---
# Adjust parameters for longer validation spans
N_SPLITS = 3  # Reduce splits to allow for larger test_size with reasonable train size
TEST_SIZE = 52 # Set validation fold size (e.g., 52 weeks for a year)

# Check if data is sufficient for the chosen split strategy
min_required_samples = TEST_SIZE * (N_SPLITS) + 1 # Need at least 1 sample for initial train
if N_samples < min_required_samples:
    print(f"\nError: Not enough data ({N_samples} samples) for TimeSeriesSplit with n_splits={N_SPLITS} and test_size={TEST_SIZE}.")
    print(f"  Minimum required samples: {min_required_samples}")
    # Option 1: Reduce N_SPLITS or TEST_SIZE
    # Option 2: Exit
    exit()
else:
    initial_train_size = N_samples - (N_SPLITS * TEST_SIZE)
    print(f"\nUsing TimeSeriesSplit with n_splits={N_SPLITS}, test_size={TEST_SIZE}.")
    print(f"  Total samples: {N_samples}")
    print(f"  Implied initial training set size (first fold): {initial_train_size}")

tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
print("--- IMPORTANT: Running models WITHOUT feature scaling & WITHOUT hyperparameter tuning in loop ---")

# Define scorers (direct function references)
scorers = {
    'custom_mape': custom_mape,
    'mae': mean_absolute_error,
    'rmse': rmse
}

# --- Run Time Series Cross-Validation ---
results = {model_name: {metric: [] for metric in scorers} for model_name in models_config}
start_time_cv = time.time()

for fold_idx, (train_index, val_index) in enumerate(tscv.split(X_all)):
    print(f"\n--- Processing Fold {fold_idx + 1}/{N_SPLITS} ---")
    # Should not happen with the check above, but good practice
    if len(train_index) == 0 or len(val_index) == 0:
         print(f"Warning: Fold {fold_idx + 1} resulted in empty train or validation set. Skipping.")
         continue

    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    print(f"  Train indices: {train_index[0]} to {train_index[-1]} (Size: {len(train_index)})")
    # val_index size should equal TEST_SIZE
    print(f"  Validation indices: {val_index[0]} to {val_index[-1]} (Size: {len(val_index)})")


    for model_name, config in models_config.items():
        print(f"  Model: {model_name}")
        fold_start_time = time.time()

        # 1. Select features
        current_features = config['features']
        X_train_fold_selected = X_all.iloc[train_index][current_features]
        X_val_fold_selected = X_all.iloc[val_index][current_features]
        if model_name == "SVM":
            print("    Applying StandardScaler for SVM...")
            # 2a. 對每個 fold 實例化一個新的縮放器
            scaler = StandardScaler()
            
            # 2b. 僅用當前的"訓練集"來擬合縮放器，然後轉換訓練集
            X_train_fold_scaled = scaler.fit_transform(X_train_fold_selected)
            
            # 2c. 用"同一個"已經擬合好的縮放器來轉換驗證集
            X_val_fold_scaled = scaler.transform(X_val_fold_selected)
        else:
            # XGBoost 不需要縮放，保持原樣
            X_train_fold_scaled = X_train_fold_selected
            X_val_fold_scaled = X_val_fold_selected
        # 2. Instantiate model
        model = config['model_class'](**config['params'])

        try:
            # 3. Fit model
            model.fit(X_train_fold_scaled, y_train_fold)

            # 4. Predict
            y_pred_val = model.predict(X_val_fold_scaled)
            y_pred_val = np.maximum(0, y_pred_val) # Ensure non-negative

            # 5. Calculate metrics
            print(f"    Calculating metrics for {model_name}...")
            for metric_name, metric_func in scorers.items():
                try:
                    score = metric_func(y_val_fold, y_pred_val)
                    if np.isinf(score) or pd.isna(score):
                         inf_nan = "Infinite" if np.isinf(score) else "NaN"
                         print(f"    Warning: {inf_nan} {metric_name} detected for {model_name} in fold {fold_idx + 1}. Storing as NaN.")
                         results[model_name][metric_name].append(np.nan)
                    else:
                        results[model_name][metric_name].append(score)
                        unit = "%" if metric_name == 'custom_mape' else ""
                        print(f"      Validation {metric_name.upper()}: {score:.4f}{unit}")

                except Exception as metric_e:
                     print(f"    Error calculating metric {metric_name} for {model_name} in fold {fold_idx + 1}: {metric_e}")
                     results[model_name][metric_name].append(np.nan)

        except Exception as e:
            print(f"  Error during fitting/prediction for {model_name} in fold {fold_idx + 1}: {e}")
            for metric_name in scorers:
                results[model_name][metric_name].append(np.nan)

        fold_end_time = time.time()
        print(f"  Finished {model_name} for fold {fold_idx+1} in {fold_end_time - fold_start_time:.2f} seconds.")

end_time_cv = time.time()
print(f"\n--- Time Series Cross-Validation Completed in {end_time_cv - start_time_cv:.2f} seconds ---")

# --- Process and Print Aggregated Results ---
print("\n--- Aggregated Time Series Cross-Validation Results (Mean ± Std Dev) ---")
print(f"--- (Using TimeSeriesSplit: n_splits={N_SPLITS}, test_size={TEST_SIZE}) ---")
print("--- (Using FIXED features and FIXED hyperparameters) ---")

summary = {}
for model_name, metrics_data in results.items():
    print(f"\n{model_name}:")
    print(f"  Features Used: {models_config[model_name]['features']}")
    print(f"  Hyperparameters Used: {models_config[model_name]['params']}")

    summary[model_name] = {}
    for metric_name, scores_list in metrics_data.items():
        valid_scores = np.array([s for s in scores_list if pd.notna(s) and np.isfinite(s)])

        if len(valid_scores) > 0:
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            unit = "%" if metric_name == 'custom_mape' else ""
            print(f"  Avg TS Cross-Validated {metric_name.upper()}: {mean_score:.2f}{unit} ± {std_score:.2f}{unit} ({len(valid_scores)}/{N_SPLITS} valid folds)")
            summary[model_name][metric_name] = {'mean': mean_score, 'std': std_score}
        else:
             print(f"  Avg TS Cross-Validated {metric_name.upper()}: No valid scores calculated.")
             summary[model_name][metric_name] = {'mean': np.nan, 'std': np.nan}


# --- Plotting Function ---
def plot_metric(summary_data, metric_key, n_splits, test_size):
    """Helper function to plot a specific metric."""
    labels = list(summary_data.keys())
    means = [summary_data[name].get(metric_key, {}).get('mean', np.nan) for name in labels]
    stds = [summary_data[name].get(metric_key, {}).get('std', np.nan) for name in labels]

    valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
    if not valid_indices:
        print(f"\nCannot plot {metric_key.upper()} as no valid scores were obtained.")
        return

    labels_valid = [labels[i] for i in valid_indices]
    means_valid = [means[i] for i in valid_indices]
    stds_valid = [stds[i] for i in valid_indices]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels_valid, means_valid, yerr=stds_valid, capsize=5, color=['skyblue', 'lightgreen'])

    unit_label = "(%)" if metric_key == 'custom_mape' else ""
    metric_name_upper = metric_key.upper()
    plt.ylabel(f'Avg TS Cross-Validated {metric_name_upper} {unit_label}')
    plt.title(f'{metric_name_upper} Comparison ({n_splits}-Fold TS CV, Test Size={test_size})')

    for bar, mean, std in zip(bars, means_valid, stds_valid):
        height = bar.get_height()
        text_y = height + (std if not np.isnan(std) else 0) * 1.1
        text_y = max(text_y, height * 0.5) # Prevent text going too low
        std_text = f'\n±{std:.2f}' if not np.isnan(std) else ''
        plt.text(bar.get_x() + bar.get_width() / 2., text_y,
                 f'{mean:.2f}{std_text}',
                 ha='center', va='bottom', fontsize=9)

    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Generate Plots ---
print("\n--- Generating Plots ---")
plot_metric(summary, 'custom_mape', N_SPLITS, TEST_SIZE) # Plot MAPE
plot_metric(summary, 'mae', N_SPLITS, TEST_SIZE)         # Plot MAE
# plot_metric(summary, 'rmse', N_SPLITS, TEST_SIZE) # Optional: Plot RMSE

print("\n--- Script Finished ---")