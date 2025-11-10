import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv('data/merged_file1.csv')

# 1. 顯示資料基本資訊
print("資料基本資訊:")
print(df.info())

# 2. 顯示資料的統計摘要
print("\n資料統計摘要:")
print(df.describe())

# 3. 計算缺失值比例
missing_data = df.isnull().mean() * 100  # 計算每個特徵的缺失比例
print("\n缺失值比例:")
print(missing_data[missing_data > 0])  # 只顯示有缺失值的特徵

# 4. 檢查資料是否不平衡（以ConfirmedCases為例）
# 假設 "ConfirmedCases" 是關鍵的預測目標變數，來檢查是否資料不平衡
if 'ConfirmedCases' in df.columns:
    case_counts = df['ConfirmedCases'].value_counts(normalize=True)
    print("\n'ConfirmedCases'資料分佈（比例）:")
    print(case_counts)
    
    # 視覺化資料不平衡情況
    case_counts.plot(kind='bar', color=['red', 'green'], title="Confirmed Cases Distribution")
    plt.xlabel("ConfirmedCases")
    plt.ylabel("Proportion")
    plt.show()

# 5. 特徵篩選（示範：基於與ConfirmedCases的相關性篩選）
correlation_matrix = df.corr()
print("\n特徵間的相關性矩陣:")
print(correlation_matrix)

# 篩選與目標變數（ConfirmedCases）相關性最高的特徵
correlation_with_target = correlation_matrix['ConfirmedCases'].sort_values(ascending=False)
print("\n與'ConfirmedCases'的相關性:")
print(correlation_with_target)

# 取與目標變數相關性較高的特徵（假設門檻為 0.1）
high_correlation_features = correlation_with_target[abs(correlation_with_target) > 0.1].index
print("\n選取的特徵:")
print(high_correlation_features)