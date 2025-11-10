import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import wandb

# 自定義 MAPE 評分器
def mape(y_true, y_pred):
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
def create_lag_features(data, lag):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['ConfirmedCases'].shift(i)
    return data
def mape(y_true, y_pred):
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# 初始化 wandb
wandb.init(project="influenza_formal_test", name="SVR Model with Feature Selection t2 ")

# 讀取數據
data = pd.read_csv("project/data/merged_file.csv")
data = create_lag_features(data, lag=3)  # 創建滯後特徵
data.dropna(inplace=True)

# 提取年份和周數
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', 'AverageTemperature', 'lag_1', 'lag_2', 'lag_3']]
y = data['ConfirmedCases']

# 按時間順序分割數據
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 初始化變量
candidate_features = list(X.columns)  # 候選特徵集合
selected_features = []  # 已選特徵集合
best_mape = float('inf')  # 初始化最佳 MAPE
score_history = []
# 定義參數網格
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'epsilon': [0.001, 0.01, 0.1],
    'kernel': ['rbf']
}


# 特徵選擇循環
while candidate_features:
    performance = []  # 用於儲存當前輪次的 MAPEs
    for feature in candidate_features:
        # 試加入特徵
        current_features = selected_features + [feature]
        X_train_selected = X_train[current_features]
        X_test_selected = X_test[current_features]
        
        # 使用 GridSearchCV 訓練模型
        grid_search = GridSearchCV(SVR(), param_grid, cv=3, scoring=make_scorer(mape, greater_is_better=False))
        grid_search.fit(X_train_selected, y_train)
        y_pred = grid_search.predict(X_test_selected)
        current_mape = mape(y_test, y_pred)
        performance.append((current_mape, feature))
    
    # 選取降低 MAPE 的最佳特徵
    performance.sort()
    best_performance = performance[0]
    
    if best_performance[0] < best_mape:
        best_mape = best_performance[0]
        selected_features.append(best_performance[1])
        candidate_features.remove(best_performance[1])
        score_history.append(best_mape)
        print(f"Selected Feature: {best_performance[1]}, MAPE: {best_performance[0]:.2f}")
        
        # Log Wandb
        wandb.log({"Selected Feature": best_performance[1], "MAPE": best_performance[0]})
    else:
        # 如果沒有特徵可以進一步提升 MAPE，則終止迴圈
        break

# 使用選定的特徵進行最終訓練並評估
X_train_final = X_train[selected_features]
X_test_final = X_test[selected_features]

# 使用 GridSearchCV 獲得最優模型
grid_search = GridSearchCV(SVR(), param_grid, cv=3, scoring=make_scorer(mape, greater_is_better=False))
grid_search.fit(X_train_final, y_train)
best_model = grid_search.best_estimator_

# 評估最終模型
y_pred = best_model.predict(X_test_final)
final_mape = mape(y_test, y_pred)
final_mse = mean_squared_error(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
print("Best parameters:", grid_search.best_params_)
# 輸出最終結果
print(f"Final selected features: {selected_features}")
print(f"Final MAPE: {final_mape:.2f}%")
print(f"Final MSE: {final_mse:.2f}")
print(f"Final MAE: {final_mae:.2f}")

wandb.log({"Final MAPE": final_mape, "Final MSE": final_mse, "Final MAE": final_mae})
# 實際值與預測值的比較圖
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
plt.plot(y_pred, label="Predicted", linestyle='--', marker='x')
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Confirmed Cases")
plt.title("Actual vs Predicted Confirmed Cases")
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(score_history) + 1), score_history, marker='o', linestyle='-')
plt.xlabel("Number of Features Selected")
plt.ylabel("Mean Absolute Percentage Error (MAPE)")
plt.title("Loss Change with Feature Selection (SVM)")
plt.savefig("svr_loss.png", dpi=300)
plt.show()
# 完成 Wandb 記錄
wandb.finish()