import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
JSONL_FILE = "/raid/data/ly/data/dataset/ifd_results.jsonl" # 你的结果文件路径
OUTPUT_PLOT = "ifd_distribution.png" # 分布图保存路径
# ===========================================

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    
    # 从 file_path 中提取文件夹名称，方便按文件夹统计
    # 假设路径格式类似 .../dataset/data/folder_name/sample.json
    df['folder'] = df['file_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    return df

def analyze():
    # 1. 加载数据
    if not os.path.exists(JSONL_FILE):
        print(f"错误: 找不到文件 {JSONL_FILE}")
        return
    
    df = load_data(JSONL_FILE)
    
    print("=== 1. 全局统计分析 ===")
    overall_min = df['ifd_score'].min()
    overall_max = df['ifd_score'].max()
    overall_mean = df['ifd_score'].mean()
    overall_avg_steps = df['response_steps'].mean()
    
    print(f"总样本数: {len(df)}")
    print(f"IFD 最小值: {overall_min:.4f}")
    print(f"IFD 最大值: {overall_max:.4f}")
    print(f"IFD 平均值: {overall_mean:.4f}")
    print(f"全局平均 Response Steps: {overall_avg_steps:.2f}")
    print("-" * 30)

    # 2. 绘制全量分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ifd_score'], kde=True, bins=30, color='skyblue')
    plt.axvline(overall_mean, color='red', linestyle='--', label=f'Mean: {overall_mean:.2f}')
    plt.title('Distribution of IFD Scores (Overall)')
    plt.xlabel('IFD Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(OUTPUT_PLOT)
    print(f"分布图已保存至: {OUTPUT_PLOT}")

    print("\n=== 2. 各文件夹统计分析 ===")
    # 按文件夹分组
    grouped = df.groupby('folder')
    
    folder_stats = []
    
    for folder_name, group in grouped:
        # 基础统计
        f_min = group['ifd_score'].min()
        f_max = group['ifd_score'].max()
        f_mean = group['ifd_score'].mean()
        f_avg_steps = group['response_steps'].mean()
        f_count = len(group)
        
        # 找到该文件夹中 IFD 最大的那条数据的 steps
        max_ifd_row = group.loc[group['ifd_score'].idxmax()]
        max_ifd_steps = max_ifd_row['response_steps']
        
        folder_stats.append({
            "Folder": folder_name,
            "Count": f_count,
            "IFD_Min": f_min,
            "IFD_Max": f_max,
            "IFD_Mean": f_mean,
            "Avg_Steps": f_avg_steps,
            "Max_IFD_Steps": max_ifd_steps
        })
    
    # 转换为 DataFrame 打印得更整齐
    stats_df = pd.DataFrame(folder_stats)
    # 按平均 IFD 从高到低排序，让你一眼看到哪个文件夹最“难”
    stats_df = stats_df.sort_values(by="IFD_Mean", ascending=False)
    
    print(stats_df.to_string(index=False, formatters={
        'IFD_Min': '{:,.4f}'.format,
        'IFD_Max': '{:,.4f}'.format,
        'IFD_Mean': '{:,.4f}'.format,
        'Avg_Steps': '{:,.2f}'.format
    }))

    # 3. 补充：简单相关性观察
    correlation = df['ifd_score'].corr(df['response_steps'])
    print(f"\n[额外发现] IFD值 与 步数 的相关系数: {correlation:.4f}")
    print("(注: 接近1表示步数越多IFD通常越高，接近0表示两者无关)")

if __name__ == "__main__":
    analyze()
