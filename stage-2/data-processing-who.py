import pandas as pd

# === 讀取原始清理後的亞洲資料 ===
in_file = "stage-2/data/Asia_FluNet.csv"
df = pd.read_csv(in_file, encoding="utf-8-sig")

print("原始 shape:", df.shape)

# === 缺失值處理 ===

# 1. 刪除沒有國家或時間標籤的列
df = df.dropna(subset=["COUNTRY_AREA_TERRITORY", "ISO2", "WHOREGION", "ISO_YEAR", "ISO_WEEK", "ISOYW"])

# 2. 補 0：檢測數據
for col in ["INF_A", "INF_B", "INF_ALL", "INF_NEGATIVE"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

# 3. 補 0：陽性率（PositivityRate）
if "PositivityRate" not in df.columns:
    df["PositivityRate"] = df["INF_ALL"] / (df["INF_ALL"] + df["INF_NEGATIVE"])
df["PositivityRate"] = df["PositivityRate"].fillna(0)

# === 四季標籤 ===
def assign_season(row):
    try:
        week = int(row["ISO_WEEK"])
    except:
        return "Unknown"
    hemisphere = row.get("HEMISPHERE", "NH").upper()
    if hemisphere == "NH":
        if 10 <= week <= 22:
            return "Spring"
        elif 23 <= week <= 35:
            return "Summer"
        elif 36 <= week <= 48:
            return "Autumn"
        else:
            return "Winter"
    else:  # 南半球
        if 10 <= week <= 22:
            return "Spring"
        elif 23 <= week <= 35:
            return "Summer"
        elif 36 <= week <= 48:
            return "Autumn"
        else:
            return "Winter"

df["SEASON"] = df.apply(assign_season, axis=1)

# === 排序 ===
df = df.sort_values(["COUNTRY_AREA_TERRITORY", "ISO_YEAR", "ISO_WEEK"])

# === 輸出乾淨檔案 ===
out_file = "stage-2/data/Asia_FluNet_1.csv"
df.to_csv(out_file, index=False, encoding="utf-8-sig")

print("✅ 已輸出乾淨資料:", out_file)
print("最終 shape:", df.shape)
print(df.head(10))