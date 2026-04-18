import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set the chart style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans'] # Prioritize Arial
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

# ============================================
# Classification Task Functions
# ============================================

def preprocess_classification_xgb(df):
    """Preprocesses data for the classification task"""
    df = df.copy()
    df['HighPositivity'] = (df['PositivityRate'] > df['PositivityRate'].median()).astype(int)
    le_country = LabelEncoder()
    le_region = LabelEncoder()
    le_season = LabelEncoder()
    df['COUNTRY_ENCODED'] = le_country.fit_transform(df['COUNTRY_AREA_TERRITORY'].astype(str))
    df['REGION_ENCODED'] = le_region.fit_transform(df['WHOREGION'].astype(str))
    df['SEASON_ENCODED'] = le_season.fit_transform(df['SEASON'].astype(str))
    df['WEEK_SIN'] = np.sin(2 * np.pi * df['ISO_WEEK'] / 52)
    df['WEEK_COS'] = np.cos(2 * np.pi * df['ISO_WEEK'] / 52)
    features = ['COUNTRY_ENCODED', 'REGION_ENCODED', 'ISO_YEAR', 
                'WEEK_SIN', 'WEEK_COS', 'SEASON_ENCODED', 'Urbanisation']
    X = df[features].fillna(0)
    y = df['HighPositivity']
    return X, y

def train_xgb_classifier(X, y):
    """Trains the XGBoost Classifier"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("=" * 50)
    print("Method A: Basic XGBoost Classifier")
    print("=" * 50)
    xgb_basic = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_basic.fit(X_train, y_train)
    y_pred = xgb_basic.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "=" * 50)
    print("Method B: Grid Search for Hyperparameters")
    print("=" * 50)
    param_grid = {
        'n_estimators': [100, 200], 'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
    return best_model

# ============================================
# Regression Task Functions
# ============================================

def preprocess_regression_xgb(df):
    """Preprocesses data for the regression task and returns feature names"""
    df = df.copy()
    le_country = LabelEncoder()
    le_region = LabelEncoder()
    le_season = LabelEncoder()
    df['COUNTRY_ENCODED'] = le_country.fit_transform(df['COUNTRY_AREA_TERRITORY'].astype(str))
    df['REGION_ENCODED'] = le_region.fit_transform(df['WHOREGION'].astype(str))
    df['SEASON_ENCODED'] = le_season.fit_transform(df['SEASON'].astype(str))
    df['WEEK_SIN'] = np.sin(2 * np.pi * df['ISO_WEEK'] / 52)
    df['WEEK_COS'] = np.cos(2 * np.pi * df['ISO_WEEK'] / 52)
    features = ['COUNTRY_ENCODED', 'REGION_ENCODED', 'ISO_YEAR', 
                'WEEK_SIN', 'WEEK_COS', 'SEASON_ENCODED', 'Urbanisation']
    X = df[features].fillna(0)
    y = df['PositivityRate'].fillna(df['PositivityRate'].median())
    return X, y, features

def train_xgb_regressor(X_train, X_test, y_train, y_test):
    """Trains the XGBoost Regressor (accepts pre-split data)"""
    print("=" * 50)
    print("XGBoost Regression Model")
    print("=" * 50)
    xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    print("\nGrid searching for best parameters...")
    param = {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 100, 'subsample': 0.8}
    """grid_search = GridSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', random_state=42), 
        param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    """
    best_model = XGBRegressor(**param)
    best_model.fit(X_train,y_train)
    y_pred_best = best_model.predict(X_test)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)
    print(f"Best Model RMSE: {rmse_best:.4f}")
    print(f"Best Model R²: {r2_best:.4f}")
    
    return best_model

# ============================================
# NEW: Analysis Functions (in English)
# ============================================

def analyze_feature_importance(model, features):
    """1. Analyzes and visualizes feature importance"""
    print("\n" + "=" * 50)
    print("Analysis 1: Feature Importance")
    print("=" * 50)
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("Feature contribution ranking:")
    print(importance_df)
    
    # --- Plotting (with professional style) ---
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    sns.barplot(x='Importance', y='Feature', data=importance_df, color='#8b5cf6', ax=ax)
    
    # Style settings
    ax.set_title('XGBoost Feature Importance Analysis', loc='left', fontsize=14, weight='bold')
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_ylabel('') # Y-label is omitted for clarity
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', color='lightgray', linewidth=0.7)
    
    # Add value labels to bars
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.001, p.get_y() + p.get_height() / 2,
                f'{width:.3f}',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("(XGB)Feature Importance Analysis.jpg")
    plt.show()

def analyze_errors(model, X_test, y_test, original_df):
    """2. Analyzes the model's prediction errors"""
    print("\n" + "=" * 50)
    print("Analysis 2: Error Analysis")
    print("=" * 50)
    
    y_pred = model.predict(X_test)
    
    # --- 2a. Plot Actual vs. Predicted values ---
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.scatter(y_test, y_pred, alpha=0.5, color='#3b82f6', s=30, edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            '--', color='#ef4444', linewidth=1.5, label='Perfect Prediction Line (y=x)')
    
    # Style settings
    ax.set_title('Actual vs. Predicted Values', loc='left', fontsize=14, weight='bold')
    ax.set_xlabel('Actual Positivity Rate', fontsize=11)
    ax.set_ylabel('Predicted Positivity Rate', fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', color='lightgray', linewidth=0.7)
    ax.legend(frameon=False, loc='upper left')
    
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig("(XGB)Actual Positivity Rate vs Predicted.jpg")
    plt.show()

    # --- 2b. Identify the largest errors ---
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    results_df['Error'] = results_df['Actual'] - results_df['Predicted']
    results_df['Abs_Error'] = abs(results_df['Error'])
    
    original_context = original_df.loc[X_test.index]
    full_results = pd.concat([original_context, results_df], axis=1)
    
    top_10_errors = full_results.sort_values(by='Abs_Error', ascending=False).head(10)
    
    print("\nTop 10 Predictions with the Largest Errors:")
    display_cols = ['COUNTRY_AREA_TERRITORY', 'ISO_YEAR', 'ISO_WEEK', 'Actual', 'Predicted', 'Error']
    print(top_10_errors[display_cols].round(3))

# ============================================
# Main Execution Block (in English)
# ============================================

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('/kaggle/input/influenza/data.csv') # Please change to your actual file path
    
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows of data:")
    print(df.head())
    
    # --- Classification Task ---
    print("\n\n" + "=" * 60)
    print("Running XGBoost Classification Task: Predicting High/Low Positivity")
    print("=" * 60)
    X_cls_xgb, y_cls_xgb = preprocess_classification_xgb(df)
    model_xgb_cls = train_xgb_classifier(X_cls_xgb, y_cls_xgb)
    
    # --- Regression Task and Analysis ---
    print("\n\n" + "=" * 60)
    print("Running XGBoost Regression Task: Predicting Positivity Rate")
    print("=" * 60)
    
    # 1. Preprocess data
    X_reg_xgb, y_reg_xgb, reg_features = preprocess_regression_xgb(df)
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg_xgb, y_reg_xgb, test_size=0.2, random_state=42
    )
    
    # 3. Train the model
    model_xgb_reg = train_xgb_regressor(X_train, X_test, y_train, y_test)
    
    # --- Run New Analysis Steps ---
    print("\n\n" + "=" * 60)
    print("Running Regression Model Analysis")
    print("=" * 60)
    
    # 4. Analyze Feature Importance
    analyze_feature_importance(model_xgb_reg, reg_features)
    
    # 5. Analyze Prediction Errors
    analyze_errors(model_xgb_reg, X_test, y_test, df)
    
    print("\n\nAnalysis Complete!")