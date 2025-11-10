import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

# fucking output dir
OUTPUT_DIR = "output_analysis"


def load_and_process_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{csv_path}'。")
        return None, []
    numeric_cols = ['INF_A', 'INF_B', 'INF_ALL', 'PositivityRate', 'ISO_YEAR', 'ISO_WEEK']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols).copy()
    
    df['ISO_WEEK'] = df['ISO_WEEK'].astype(int)
    df['ISO_YEAR'] = df['ISO_YEAR'].astype(int)
    df['year_week_str'] = df['ISO_YEAR'].astype(str) + '-W' + df['ISO_WEEK'].astype(str).str.zfill(2)

    # setup countries list
    countries = sorted(df['COUNTRY_AREA_TERRITORY'].unique())
    print(f"成功載入數據。在檔案中偵測到 {len(countries)} 個國家/地區。")
    return df, countries

def plot_niid_style_chart(df, country_name, seasons, virus_types, colors):
    """
    bar chart 
    """
    num_seasons = len(seasons)
    fig, axes = plt.subplots(
        nrows=num_seasons, ncols=1, figsize=(12, 4 * num_seasons), sharey=True
    )
    if num_seasons == 1:
        axes = [axes]
    fig.set_facecolor('white')

    has_data = False
    for i, (season_label, (start_year, end_year)) in enumerate(seasons.items()):
        ax = axes[i]
        ax.set_facecolor('white')

        season_data = df[((df['ISO_YEAR'] == start_year) & (df['ISO_WEEK'] >= 40)) |
                         ((df['ISO_YEAR'] == end_year) & (df['ISO_WEEK'] <= 20))].copy()
        
        season_data = season_data.sort_values(['ISO_YEAR', 'ISO_WEEK']).reset_index(drop=True)
        if season_data.empty:
            ax.text(0.5, 0.5, f"No data available for {season_label}", ha='center', va='center')
            continue

        has_data = True
        x = np.arange(len(season_data))
        bottom = np.zeros(len(season_data))
        
        for virus, color in zip(virus_types, colors):
            if virus in season_data.columns:
                counts = season_data[virus].fillna(0).values
                ax.bar(x, counts, bottom=bottom, label=virus, color=color, width=0.8)
                bottom += counts
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', linestyle='--', color='lightgray', linewidth=0.7)

        tick_spacing = 4 if len(x) > 10 else 1
        ax.set_xticks(x[::tick_spacing])
        ax.set_xticklabels(season_data['year_week_str'].iloc[::tick_spacing], rotation=45, ha='right', fontsize=8)
        
        ax.set_title(f"Influenza Detections for {country_name} ({season_label})", loc='left', fontsize=12, weight='bold')
        ax.set_ylabel("Weekly Detections")
        
        legend = ax.legend(loc='upper right', frameon=False, fontsize=9, title='Virus Type')
        legend.get_title().set_fontweight('bold')
        
    if has_data:
        plt.tight_layout(h_pad=3)
        filepath = os.path.join(OUTPUT_DIR, country_name, "0_niid_surveillance_chart.png")
        plt.savefig(filepath)
    plt.close()

def plot_time_series(df, country_name):
    """
    function 1: generate and save time series plot
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    fig.set_facecolor('white')
    
    ax1.plot(df['year_week_str'], df['INF_ALL'], label='Total Influenza', color='#8b5cf6', linewidth=2)
    ax1.plot(df['year_week_str'], df['INF_A'], label='Influenza A', color='#3b82f6', linewidth=1.5)
    ax1.plot(df['year_week_str'], df['INF_B'], label='Influenza B', color='#10b981', linewidth=1.5)
    ax1.set_title(f'Influenza Cases Time Series for {country_name}', loc='left', fontsize=12, weight='bold')
    ax1.set_ylabel('Number of Cases')
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid(axis='y', linestyle='--', color='lightgray')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2.plot(df['year_week_str'], df['PositivityRate'], label='Positivity Rate (%)', color='#f59e0b', linewidth=2)
    ax2.set_title(f'Positivity Rate Trend for {country_name}', loc='left', fontsize=12, weight='bold')
    ax2.set_ylabel('Rate (%)')
    ax2.legend(loc='upper right', frameon=False)
    ax2.grid(axis='y', linestyle='--', color='lightgray')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    tick_spacing = max(1, len(df) // 15)
    plt.xticks(df['year_week_str'][::tick_spacing], rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, country_name, "1_time_series.png")
    plt.savefig(filepath)
    plt.close()

def plot_acf_analysis(df, country_name):
    """
    function 2 : ACF ( ???) analysis plot
    """
    inf_all_values = df['INF_ALL'].values
    if len(inf_all_values) < 104:
        return # 如果數據太少，無法進行有意義的ACF分析 bro i need to claim how ACF works
    
    acf_values = acf(inf_all_values, nlags=104, fft=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('white')
    
    ax.bar(range(len(acf_values)), acf_values, width=0.8, color='#8b5cf6')
    conf_level = 1.96 / np.sqrt(len(inf_all_values))
    ax.axhline(y=conf_level, color='red', linestyle='--', linewidth=1)
    ax.axhline(y=-conf_level, color='red', linestyle='--', linewidth=1)

    ax.set_title(f'Autocorrelation Function (ACF) for {country_name}', loc='left', fontsize=12, weight='bold')
    ax.set_xlabel('Lag (Weeks)'); ax.set_ylabel('Autocorrelation')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.axvline(x=52, color='darkgreen', linestyle=':', linewidth=2, label='1-Year Lag (52 weeks)')
    ax.legend(frameon=False)
        
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, country_name, "2_acf_analysis.png")
    plt.savefig(filepath)
    plt.close()

def plot_seasonal_patterns(df, country_name):
    """
    seasonal pattern ~~~
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor('white')

    weekly_avg = df.groupby('ISO_WEEK')[['INF_ALL', 'INF_A', 'INF_B']].mean()
    ax.plot(weekly_avg.index, weekly_avg['INF_ALL'], label='Avg Total Influenza', color='#8b5cf6', linewidth=2.5)
    ax.plot(weekly_avg.index, weekly_avg['INF_A'], label='Avg Influenza A', color='#3b82f6', linewidth=1.5, linestyle='--')
    ax.plot(weekly_avg.index, weekly_avg['INF_B'], label='Avg Influenza B', color='#10b981', linewidth=1.5, linestyle='--')
    
    ax.set_title(f'Average Weekly Cases - Seasonal Pattern for {country_name}', loc='left', fontsize=12, weight='bold')
    ax.set_ylabel('Average Number of Cases'); ax.set_xlabel('Week of the Year')
    ax.legend(frameon=False)
    ax.grid(axis='y', linestyle='--', color='lightgray')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlim(1, 52)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, country_name, "3_seasonal_patterns.png")
    plt.savefig(filepath)
    plt.close()

def analyze_yearly_stats(df, country_name):
    """
    just a chart 
    """
    yearly_stats = df.groupby('ISO_YEAR')['INF_ALL'].agg(['sum', 'mean', 'max']).reset_index()
    peak_weeks = df.loc[df.groupby('ISO_YEAR')['INF_ALL'].idxmax()][['ISO_YEAR', 'ISO_WEEK']]
    yearly_stats = pd.merge(yearly_stats, peak_weeks, on='ISO_YEAR', how='left')
    yearly_stats = yearly_stats.rename(columns={
        'sum': 'Total Cases', 'mean': 'Avg Weekly Cases', 
        'max': 'Max Weekly Cases', 'ISO_WEEK': 'Peak Week'
    })

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    fig.set_facecolor('white')

    ax1.bar(yearly_stats['ISO_YEAR'], yearly_stats['Total Cases'], color='#8b5cf6')
    ax1.set_title(f'Total Annual Influenza Cases for {country_name}', loc='left', fontsize=12, weight='bold')
    ax1.set_ylabel('Total Cases')
    ax1.set_xticks(yearly_stats['ISO_YEAR']); ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', color='lightgray')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2.scatter(yearly_stats['ISO_YEAR'], yearly_stats['Peak Week'], color='#ef4444', s=100, zorder=3)
    ax2.set_title(f'Annual Peak Week Distribution for {country_name}', loc='left', fontsize=12, weight='bold')
    ax2.set_ylabel('Peak Week'); ax2.set_xlabel('Year')
    ax2.set_ylim(0, 53); ax2.set_xticks(yearly_stats['ISO_YEAR'])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', color='lightgray')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    
    plt.tight_layout(h_pad=3)
    filepath = os.path.join(OUTPUT_DIR, country_name, "4_yearly_statistics.png")
    plt.savefig(filepath)
    plt.close()
    
    return yearly_stats

# --- 主執行程式 ---
if __name__ == '__main__':
    # --- 使用者設定 ---
    CSV_FILE_PATH = "Asia_FluNet_1_clean.csv" # <--- 請將此處改為您的檔案名稱
    
    # 載入所有數據
    full_df, countries = load_and_process_data(CSV_FILE_PATH)
    
    if full_df is not None:
        # 創建主輸出資料夾
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"所有分析結果將儲存於 '{OUTPUT_DIR}' 資料夾中。")
        
        all_yearly_stats = []

        # 遍歷每一個國家進行分析
        for country in countries:
            print(f"\n--- 正在處理: {country} ---")
            
            # 為該國創建子資料夾
            country_dir = os.path.join(OUTPUT_DIR, country)
            os.makedirs(country_dir, exist_ok=True)
            
            # 篩選該國數據並排序
            country_df = full_df[full_df['COUNTRY_AREA_TERRITORY'] == country].copy()
            country_df = country_df.sort_values(['ISO_YEAR', 'ISO_WEEK']).reset_index(drop=True)

            if country_df.empty:
                print(f"'{country}' 的數據為空，已跳過。")
                continue

            # --- 執行所有分析與繪圖功能 ---
            
            # 1. NIID 風格監測圖 (可自定義季節)
            seasons_to_plot = {
                '2019/20': (2019, 2020),
                '2021/22': (2021, 2022)
            }
            plot_niid_style_chart(
                country_df, country, seasons=seasons_to_plot,
                virus_types=['INF_A', 'INF_B'],
                colors=['cyan', 'magenta']
            )

            # 2. 時間序列分析
            plot_time_series(country_df, country)

            # 3. 自相關分析 (ACF)
            plot_acf_analysis(country_df, country)

            # 4. 季節性模式分析
            plot_seasonal_patterns(country_df, country)

            # 5. 年度統計
            yearly_report = analyze_yearly_stats(country_df, country)
            all_yearly_stats.append((country, yearly_report))
        
        # --- 處理完成後，在終端機印出所有國家的年度報告摘要 ---
        # --- 處理完成後，輸出統計報告 ---
        
        # 1. 合併所有國家的年度統計數據
        all_countries_stats = []
        for country, report_df in all_yearly_stats:
            report_df_copy = report_df.copy()
            report_df_copy.insert(0, 'Country', country)
            all_countries_stats.append(report_df_copy)
        
        if all_countries_stats:
            combined_stats = pd.concat(all_countries_stats, ignore_index=True)
            
            # 2. 輸出詳細統計報告（CSV格式）
            detailed_report_path = os.path.join(OUTPUT_DIR, "yearly_statistics_detailed.csv")
            combined_stats.to_csv(detailed_report_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 詳細年度統計報告已儲存至: {detailed_report_path}")
            
            # 3. 輸出摘要統計報告（每個國家的總體指標）
            summary_stats = []
            for country, report_df in all_yearly_stats:
                summary = {
                    'Country': country,
                    'Years Covered': f"{report_df['ISO_YEAR'].min()}-{report_df['ISO_YEAR'].max()}",
                    'Total Cases (All Years)': report_df['Total Cases'].sum(),
                    'Average Annual Cases': report_df['Total Cases'].mean(),
                    'Highest Annual Cases': report_df['Total Cases'].max(),
                    'Most Common Peak Week': report_df['Peak Week'].mode()[0] if len(report_df['Peak Week'].mode()) > 0 else None,
                    'Peak Week Range': f"{report_df['Peak Week'].min():.0f}-{report_df['Peak Week'].max():.0f}"
                }
                summary_stats.append(summary)
            
            summary_df = pd.DataFrame(summary_stats)
            summary_report_path = os.path.join(OUTPUT_DIR, "country_summary_statistics.csv")
            summary_df.to_csv(summary_report_path, index=False, encoding='utf-8-sig')
            print(f"✓ 國家摘要統計報告已儲存至: {summary_report_path}")
            
            # 4. 在終端機印出摘要
            print("\n\n================================================")
            print("          所有國家年度統計報告摘要")
            print("================================================")
            for country, report_df in all_yearly_stats:
                print(f"\n--- {country} ---")
                print(report_df.to_string(index=False))
                print("--------------------\n")
        
        print("\n所有分析已完成！")
        print(f"圖表存放於各國子資料夾，統計數據存放於 '{OUTPUT_DIR}' 主資料夾。")