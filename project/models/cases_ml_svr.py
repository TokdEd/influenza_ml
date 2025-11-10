import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import wandb

def create_lag_features(data, lag):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['ConfirmedCases'].shift(i)
    return data

def mape(y_true, y_pred):
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# 初始化 wandb 项目
wandb.init(project="influenza_formal_test", name="SVR Model - Best vs Worst Parameters")

# 询问是否记录到 wandb
record_wandb = input("是否记录 wandb 数据？(y/n): ").strip().lower() == 'y'

# 读取病例数据
data = pd.read_csv("project/data/merged_file.csv")
data = create_lag_features(data, lag=3)
data.dropna(inplace=True)

# 提取年份和周数
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', 'AverageTemperature', 'lag_1', 'lag_2', 'lag_3']]
y = data['ConfirmedCases']

# 按时间顺序分割数据
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 0.0001, 0.001, 0.01, 0.1, 1, 10],
    'epsilon': [0.001, 0.01, 0.1, 1, 5],
    'kernel': ['rbf']
}

scoring = {
    'MAPE': make_scorer(mape, greater_is_better=False),
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
    'MAE': make_scorer(mean_absolute_error, greater_is_better=False)
}

# 使用 GridSearchCV 进行网格搜索
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring=scoring, refit='MAPE', return_train_score=True, verbose=2)
grid_search.fit(X_train, y_train)

# 找出最差的参数组合
cv_results = pd.DataFrame(grid_search.cv_results_)
worst_idx = cv_results['mean_test_MAPE'].argmin()  # 因为是负数，所以用argmin
worst_params = grid_search.cv_results_['params'][worst_idx]
worst_mape = -cv_results['mean_test_MAPE'][worst_idx]
"""
Worst parameters: {'C': 1000, 'epsilon': 5, 'gamma': 0.1, 'kernel': 'rbf'}
"""
print("Worst parameters:", worst_params)
print("Worst MAPE score:", worst_mape)

# 使用最差参数创建模型
worst_model = SVR(**worst_params)
worst_model.fit(X_train, y_train)
y_pred_worst = worst_model.predict(X_test)

# 计算最差模型的指标
worst_final_mape = mape(y_test, y_pred_worst)
worst_final_mse = mean_squared_error(y_test, y_pred_worst)
worst_final_mae = mean_absolute_error(y_test, y_pred_worst)

print(f"Worst Model Final MAPE: {worst_final_mape:.2f}%")
print(f"Worst Model Final MSE: {worst_final_mse:.2f}")
print(f"Worst Model Final MAE: {worst_final_mae:.2f}")

# 创建对比可视化
plt.figure(figsize=(15, 8))

# 绘制最差模型的预测结果
plt.subplot(1, 2, 1)
plt.plot(y_test.reset_index(drop=True).values, label="Actual", linestyle='--', marker='o')
plt.plot(pd.Series(y_pred_worst).reset_index(drop=True).values, label="Predicted (Worst)", linestyle='--', marker='x')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Confirmed Cases')
plt.title('Actual vs Predicted (Worst Parameters)')
plt.savefig("plot_svm_worst.png")

# 绘制最佳模型的预测结果（用于对比）
y_pred_best = grid_search.best_estimator_.predict(X_test)
plt.subplot(1, 2, 2)
plt.plot(y_test.reset_index(drop=True).values, label="Actual", linestyle='--', marker='o')
plt.plot(pd.Series(y_pred_best).reset_index(drop=True).values, label="Predicted (Best)", linestyle='--', marker='x')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Confirmed Cases')
plt.title('Actual vs Predicted (Best Parameters)')
plt.savefig("plot_svm_best.png")
plt.tight_layout()
plt.savefig("plot_svr_worst_vs_best.png")
plt.show()

# 记录到 wandb（如果选择记录）
if record_wandb:
    plt.savefig('plot_svr_worst_vs_best.png')
    wandb.log({
        "Worst Parameters": worst_params,
        "Worst MAPE": worst_final_mape,
        "Worst MSE": worst_final_mse,
        "Worst MAE": worst_final_mae,
        "Comparison Chart": wandb.Image('plot_svr_worst_vs_best.png')
    })
    wandb.finish()