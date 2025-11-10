import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import wandb

def create_lag_features(data, lag):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['ConfirmedCases'].shift(i)
    return data

def calculate_mape(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = (y_true != 0) & (y_pred != 0)
    absolute_percentage_error = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / (y_true[non_zero_mask] + epsilon))
    return np.mean(absolute_percentage_error) * 100 if len(absolute_percentage_error) > 0 else np.nan

def wandb_log(x, y, grid_search=None, mape=None, mse=None, mae=None, worst_mape=None, worst_mse=None, worst_mae=None):
    if x == "n":
        return 0
    else:
        if y == "init":
            proj_name = str(input("input your project name:"))
            wandb.init(project="influenza_formal_test", name=proj_name)
        elif y == "gridsearch" and grid_search is not None:
            for i, (params, mean_score, std_score) in enumerate(zip(grid_search.cv_results_['params'],
                                                                    grid_search.cv_results_['mean_test_MSE'],
                                                                    grid_search.cv_results_['std_test_MSE'])):
                wandb.log({
                    "Fold": i,
                    "Mean MSE": -mean_score,
                    "Std MSE": std_score,
                    "Parameters": params
                })
        elif y == "log_final_data":
            wandb.log({
                "Best Model MAPE": mape, 
                "Best Model MSE": mse, 
                "Best Model MAE": mae,
                "Worst Model MAPE": worst_mape,
                "Worst Model MSE": worst_mse,
                "Worst Model MAE": worst_mae
            })
            wandb.log({
                "Best vs Worst Comparison": wandb.Image('xgboost_comparison.png'),
                "Best Model Tree": wandb.Image("best_tree.png"),
                "Worst Model Tree": wandb.Image("worst_tree.png")
            })

def main():
    # 读取数据
    data = pd.read_csv("project/data/merged_file.csv")
    data = create_lag_features(data, lag=3)
    data.dropna(inplace=True)
    
    # 提取年份和周数
    data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
    data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)
    
    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', 'AverageTemperature', 'lag_1', 'lag_2', 'lag_3']]
    y = data['ConfirmedCases']
    
    # 数据分割
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # 初始化 W&B
    use_wandb = input("Do you want to use Weights & Biases for logging? (y/n): ")
    wandb_log(use_wandb, "init")
    
    # 定义网格搜索参数
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # 定义评分函数
    mape_scorer = make_scorer(calculate_mape, greater_is_better=False)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_dist,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring={'MAPE': mape_scorer, 'MSE': mse_scorer, 'MAE': mae_scorer},
        refit='MSE'
    )
    
    grid_search.fit(X_train, y_train)
    
    # 找出最差参数
    cv_results = pd.DataFrame(grid_search.cv_results_)
    worst_idx = cv_results['mean_test_MSE'].argmin()  # 因为是负数，所以用argmin
    worst_params = grid_search.cv_results_['params'][worst_idx]
    
    print("Best parameters:", grid_search.best_params_)
    print("Worst parameters:", worst_params)
    
    # 创建最差模型
    worst_model = xgb.XGBRegressor(**worst_params, objective='reg:squarederror', random_state=42)
    worst_model.fit(X_train, y_train)
    
    # 最佳和最差模型预测
    best_pred = grid_search.best_estimator_.predict(X_test)
    worst_pred = worst_model.predict(X_test)
    
    # 计算指标
    best_mse = mean_squared_error(y_test, best_pred)
    best_mae = mean_absolute_error(y_test, best_pred)
    best_mape = calculate_mape(y_test, best_pred)
    
    worst_mse = mean_squared_error(y_test, worst_pred)
    worst_mae = mean_absolute_error(y_test, worst_pred)
    worst_mape = calculate_mape(y_test, worst_pred)
    
    print("\nBest Model Metrics:")
    print(f"MSE: {best_mse:.2f}")
    print(f"MAE: {best_mae:.2f}")
    print(f"MAPE: {best_mape:.2f}%")
    
    print("\nWorst Model Metrics:")
    print(f"MSE: {worst_mse:.2f}")
    print(f"MAE: {worst_mae:.2f}")
    print(f"MAPE: {worst_mape:.2f}%")
    
    # 创建对比可视化
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
    plt.plot(best_pred, label="Best Model", linestyle='--', marker='x')
    plt.legend()
    plt.title('Best Model Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Confirmed Cases')
    
    plt.subplot(1, 2, 2)
    plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
    plt.plot(worst_pred, label="Worst Model", linestyle='--', marker='x')
    plt.legend()
    plt.title('Worst Model Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Confirmed Cases')
    
    plt.tight_layout()
    plt.savefig('xgboost_comparison.png', dpi=300)
    plt.show()
    
    # 可视化最佳和最差树
    plt.figure(figsize=(10, 6))
    plot_tree(grid_search.best_estimator_, num_trees=0)
    plt.title("Best Model - First Tree")
    plt.savefig('best_tree.png', dpi=1200, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plot_tree(worst_model, num_trees=0)
    plt.title("Worst Model - First Tree")
    plt.savefig('worst_tree.png', dpi=1200, bbox_inches='tight')
    plt.close()
    
    # 记录到wandb
    if use_wandb == "y":
        wandb_log(use_wandb, "log_final_data", 
                 mape=best_mape, mse=best_mse, mae=best_mae,
                 worst_mape=worst_mape, worst_mse=worst_mse, worst_mae=worst_mae)

if __name__ == "__main__":
    main()