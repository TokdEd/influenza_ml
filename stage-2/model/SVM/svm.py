import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 方法1: 分類問題 - 預測高/低陽性率
# ============================================

def preprocess_classification(df):
    """預處理資料用於分類任務"""
    df = df.copy()
    
    # 建立目標變數：將PositivityRate二分類
    df['HighPositivity'] = (df['PositivityRate'] > df['PositivityRate'].median()).astype(int)
    
    # 處理類別特徵
    le_country = LabelEncoder()
    le_region = LabelEncoder()
    le_season = LabelEncoder()
    
    df['COUNTRY_ENCODED'] = le_country.fit_transform(df['COUNTRY_AREA_TERRITORY'].astype(str))
    df['REGION_ENCODED'] = le_region.fit_transform(df['WHOREGION'].astype(str))
    df['SEASON_ENCODED'] = le_season.fit_transform(df['SEASON'].astype(str))
    
    # 時間週期特徵（sin/cos變換處理週期性）
    df['WEEK_SIN'] = np.sin(2 * np.pi * df['ISO_WEEK'] / 52)
    df['WEEK_COS'] = np.cos(2 * np.pi * df['ISO_WEEK'] / 52)
    
    # 選擇特徵
    features = ['COUNTRY_ENCODED', 'REGION_ENCODED', 'ISO_YEAR', 
                'WEEK_SIN', 'WEEK_COS', 'SEASON_ENCODED', 'Urbanisation']
    
    X = df[features].fillna(0)
    y = df['HighPositivity']
    
    return X, y

def train_svm_classifier(X, y):
    """訓練SVM分類器"""
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 方法A: 基本SVM
    print("=" * 50)
    print("方法A: 基本線性SVM")
    print("=" * 50)
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    y_pred = svm_linear.predict(X_test_scaled)
    print(f"準確率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))
    
    # 方法B: RBF核SVM
    print("\n" + "=" * 50)
    print("方法B: RBF核SVM")
    print("=" * 50)
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    y_pred_rbf = svm_rbf.predict(X_test_scaled)
    print(f"準確率: {accuracy_score(y_test, y_pred_rbf):.4f}")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred_rbf))
    
    # 方法C: 網格搜尋最佳參數
    print("\n" + "=" * 50)
    print("方法C: 網格搜尋調參")
    print("=" * 50)
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    grid_search = GridSearchCV(
        SVC(random_state=42), 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"最佳參數: {grid_search.best_params_}")
    print(f"最佳交叉驗證分數: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test_scaled)
    print(f"測試集準確率: {accuracy_score(y_test, y_pred_best):.4f}")
    
    return best_model, scaler


# ============================================
# 方法2: 回歸問題 - 預測PositivityRate數值
# ============================================

def preprocess_regression(df):
    """預處理資料用於回歸任務"""
    df = df.copy()
    
    # 處理類別特徵
    le_country = LabelEncoder()
    le_region = LabelEncoder()
    le_season = LabelEncoder()
    
    df['COUNTRY_ENCODED'] = le_country.fit_transform(df['COUNTRY_AREA_TERRITORY'].astype(str))
    df['REGION_ENCODED'] = le_region.fit_transform(df['WHOREGION'].astype(str))
    df['SEASON_ENCODED'] = le_season.fit_transform(df['SEASON'].astype(str))
    
    # 時間週期特徵
    df['WEEK_SIN'] = np.sin(2 * np.pi * df['ISO_WEEK'] / 52)
    df['WEEK_COS'] = np.cos(2 * np.pi * df['ISO_WEEK'] / 52)
    
    # 選擇特徵
    features = ['COUNTRY_ENCODED', 'REGION_ENCODED', 'ISO_YEAR', 
                'WEEK_SIN', 'WEEK_COS','SEASON_ENCODED', 'Urbanisation']
    
    X = df[features].fillna(0)
    y = df['PositivityRate'].fillna(df['PositivityRate'].median())
    
    return X, y

def train_svm_regressor(X, y):
    """訓練SVM回歸器"""
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=" * 50)
    print("SVR回歸模型")
    print("=" * 50)
    
    # 訓練SVR
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    
    # 預測
    y_pred = svr.predict(X_test_scaled)
    
    # 評估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # 網格搜尋
    print("\n網格搜尋最佳參數...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'kernel': ['rbf', 'linear']
    }
    
    grid_search = GridSearchCV(
        SVR(), 
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n最佳參數: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test_scaled)
    
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)
    
    print(f"最佳模型 RMSE: {rmse_best:.4f}")
    print(f"最佳模型 R²: {r2_best:.4f}")
    
    return best_model, scaler


# ============================================
# 主程式執行範例
# ============================================

if __name__ == "__main__":
    # 載入資料（請替換為實際資料路徑）
    df = pd.read_csv('stage-2/data/data.csv')
    
    # 示範用假資料
    print("資料集大小:", df.shape)
    print("\n前5筆資料:")
    print(df.head())
    
    # 執行分類任務
    print("\n\n" + "=" * 60)
    print("執行分類任務：預測高/低陽性率")
    print("=" * 60)
    X_cls, y_cls = preprocess_classification(df)
    model_cls, scaler_cls = train_svm_classifier(X_cls, y_cls)
    
    # 執行回歸任務
    print("\n\n" + "=" * 60)
    print("執行回歸任務：預測陽性率數值")
    print("=" * 60)
    X_reg, y_reg = preprocess_regression(df)
    model_reg, scaler_reg = train_svm_regressor(X_reg, y_reg)
    
    print("\n\n訓練完成！")