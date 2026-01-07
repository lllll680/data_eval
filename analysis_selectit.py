import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
INPUT_FILE = "/raid/data/ly/data/dataset/data_eval/selectit_scores.csv"
OUTPUT_STATS_DIR = "/raid/data/ly/data/dataset/data_eval/analysis_results"
os.makedirs(OUTPUT_STATS_DIR, exist_ok=True)

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题（如果是Linux环境可能需要替换字体）
plt.rcParams['axes.unicode_minus'] = False

def run_analysis():
    # 1. 加载数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    
    # 提取文件夹名称 (假设路径格式为 .../文件夹名/文件名.json)
    # 通过 os.path.dirname 获取路径，再用 basename 获取最后一级文件夹名
    df['folder_name'] = df['file_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    print(f"成功加载数据，共 {len(df)} 条记录，涉及 {df['folder_name'].nunique()} 个文件夹。")

    # ================= 2. 全局统计分析 =================
    print("\n--- 全局统计摘要 ---")
    overall_stats = df['selectit_score'].describe()
    print(overall_stats)
    
    # 保存全局统计到文本
    overall_stats.to_csv(os.path.join(OUTPUT_STATS_DIR, "overall_stats.csv"))

    # 可视化：得分分布图
    plt.figure(figsize=(12, 6))
    
    # 子图1：直方图 + 密度图
    plt.subplot(1, 2, 1)
    sns.histplot(df['selectit_score'], kde=True, color='skyblue')
    plt.title('SelectIT Score 全局分布直方图')
    plt.xlabel('Score')
    plt.ylabel('频数')

    # 子图2：箱线图 (查看异常值和四分位)
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['selectit_score'], color='lightgreen')
    plt.title('SelectIT Score 全局箱线图')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_STATS_DIR, "overall_distribution.png"))
    print(f"全局分布图已保存至: {OUTPUT_STATS_DIR}/overall_distribution.png")

    # ================= 3. 每个文件夹的分组分析 =================
    print("\n--- 分文件夹统计分析 ---")
    
    # 计算每个文件夹的 均值、中位数、标准差、最大/最小值
    folder_stats = df.groupby('folder_name')['selectit_score'].agg([
        'count', 'mean', 'std', 'min', 'median', 'max'
    ]).sort_values(by='mean', ascending=False)
    
    print(folder_stats)
    
    # 保存分组统计到 CSV
    folder_stats.to_csv(os.path.join(OUTPUT_STATS_DIR, "folder_group_stats.csv"))
    print(f"分组统计数据已保存至: {OUTPUT_STATS_DIR}/folder_group_stats.csv")

    # 4. (可选额外分析) 各个文件夹的得分对比图
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='folder_name', y='selectit_score', data=df)
    plt.xticks(rotation=45)
    plt.title('不同文件夹 (Query) 的得分分布对比')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_STATS_DIR, "folder_comparison.png"))
    print(f"文件夹对比图已保存至: {OUTPUT_STATS_DIR}/folder_comparison.png")

if __name__ == "__main__":
    run_analysis()
