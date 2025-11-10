import pandas as pd

# 載入基礎數據集
df = pd.read_csv("stage-2/data/Asia_FluNet_1.csv")

print("原始 shape:", df.shape)

# 1. 保留 2019 年及以前的資料
df = df[df["ISO_YEAR"] <= 2019]

print("過濾後 shape:", df.shape)

# 2. PositivityRate 四捨五入到小數點後三位
if "PositivityRate" in df.columns:
    df["PositivityRate"] = df["PositivityRate"].round(3)

# 輸出清理後的檔案
output_path = "stage-2/data/Asia_FluNet_1_clean.csv"
df.to_csv(output_path, index=False)

print(f"✅ 已輸出 {output_path}")
print(df.head())