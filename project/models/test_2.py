import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt
import wandb 

wandb.init(project="influenza_formal_test", name="xgboost_t4")

def calculate_mape(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = (y_true != 0) & (y_pred != 0)
    absolute_percentage_error = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / (y_true[non_zero_mask] + epsilon))
    return np.mean(absolute_percentage_error) * 100 if len(absolute_percentage_error) > 0 else np.nan

def create_lag_features(data, lag):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['ConfirmedCases'].shift(i)
    return data

# Softmax 函数
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# 使用贪婪策略来选择特征组合
def greedy_feature_selection(data, target, features, max_features=5):
    selected_features = []
    score_history = []
    current_best_score = float("inf")
    
    for _ in range(max_features):
        best_feature = None
        for feature in features:
            if feature in selected_features:
                continue
            trial_features = selected_features + [feature]
            X_trial = data[trial_features]
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            model.fit(X_trial, target)
            y_pred = model.predict(X_trial)
            trial_score = calculate_mape(target, y_pred)
            
            # 如果 MAPE 改善，則更新最佳特徵
            if trial_score < current_best_score:
                current_best_score = trial_score
                best_feature = feature
        if best_feature:
            selected_features.append(best_feature)
            score_history.append(current_best_score)
            print(f"Selected feature: {best_feature}, Current Best MAPE: {current_best_score:.4f}")

    return selected_features, score_history


# 加入 Softmax 策略进行加权平均
def softmax_ensemble(predictions):
    weights = softmax([-mean_squared_error(y_test, pred) for pred in predictions])
    final_prediction = sum(weight * pred for weight, pred in zip(weights, predictions))
    return final_prediction

# 在代码中应用这两种策略
if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv("../data/merged_file.csv")
    data = create_lag_features(data, lag=3).dropna()
    data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
    data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)
    
    X = data[['Year','Week', 'ExcludedCases', 'PendingCases', 'AverageTemperature', 'lag_1', 'lag_2', 'lag_3']]
    y = data['ConfirmedCases']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 使用贪婪策略选择特征
    selected_features ,score_history  = greedy_feature_selection(X_train, y_train, features=X_train.columns.tolist())
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # 网格搜索
    param_dist = {
        'n_estimators': [100, 200, 300, 500],  # 树的数量
        'max_depth': [3, 5, 7, 10, 15],  # 树的最大深度
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率
    }
    grid_search = GridSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_dist,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train_selected, y_train)
    print("Best parameters:", grid_search.best_params_)
    # 使用Softmax策略进行集成
    best_model = grid_search.best_estimator_
    y_pred_single = best_model.predict(X_test_selected)
    y_pred_ensemble = softmax_ensemble([y_pred_single])  # 可以扩展以使用多模型预测

    # 计算损失
    mse = mean_squared_error(y_test, y_pred_ensemble)
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    mape = calculate_mape(y_test, y_pred_ensemble)
    wandb.log({"Final MAPE": mape, "Final MSE": mse, "Final MAE": mae})
    
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"最终模型 MAPE: {mape:.2f}%")
    # 可视化结果
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred_ensemble, label="Predicted")
    plt.legend()
    plt.show()
    plt.plot(range(1, len(score_history) + 1), score_history, marker='o', linestyle='-')
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Mean Abolute Percentage Error (MSE)")
    plt.title("Loss Change with Feature Selection (XGBoost)")
    plt.savefig("lose.png", dpi = 300)
    plt.show()