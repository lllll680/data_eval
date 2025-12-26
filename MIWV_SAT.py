import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 配置区域 =================
CONFIG = {
    "input_file": "/data2/ly/dataset_eval/miwv_results.json",  # 你的结果文件路径
    "output_dir": "/data2/ly/dataset_eval/analysis_result",     # 分析结果和图表保存的文件夹
    "font_path": None, # 如果图表中文乱码，可以指定中文字体路径，例如 'SimHei.ttf'
}
# ===========================================

def load_data(file_path):
    print(f">>> Loading results from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f">>> Loaded {len(df)} samples.")
    return df

def plot_distribution(df, output_dir):
    """绘制 MIWV 分布直方图"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 绘制直方图和核密度估计
    sns.histplot(df['miwv'], bins=50, kde=True, color='skyblue', edgecolor='black')
    
    # 绘制 0 值线
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (Boundary)')
    
    plt.title('Distribution of MIWV Scores', fontsize=15)
    plt.xlabel('MIWV (Loss_OneShot - Loss_ZeroShot)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    
    save_path = os.path.join(output_dir, 'miwv_distribution.png')
    plt.savefig(save_path)
    print(f"Chart saved: {save_path}")
    plt.close()

def plot_loss_comparison(df, output_dir):
    """绘制 Zero-shot Loss vs One-shot Loss 散点图"""
    plt.figure(figsize=(8, 8))
    
    # 绘制散点
    # Hue 根据 MIWV 正负着色：红色为正(Hard/High Value), 蓝色为负(Easy)
    sns.scatterplot(data=df, x='loss_zero', y='loss_one', 
                    hue=df['miwv'] > 0, palette={True: 'red', False: 'blue'}, 
                    alpha=0.6)
    
    # 绘制 y=x 对角线
    min_val = min(df['loss_zero'].min(), df['loss_one'].min())
    max_val = max(df['loss_zero'].max(), df['loss_one'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x (No Effect)')
    
    plt.title('Zero-shot Loss vs. One-shot Loss', fontsize=15)
    plt.xlabel('Zero-shot Loss (Model by itself)', fontsize=12)
    plt.ylabel('One-shot Loss (Model with example)', fontsize=12)
    plt.legend(title='MIWV > 0 (Valuable)')
    
    save_path = os.path.join(output_dir, 'loss_comparison.png')
    plt.savefig(save_path)
    print(f"Chart saved: {save_path}")
    plt.close()

def save_top_k_subsets(df, original_data, output_dir):
    """根据 MIWV 筛选 Top K% 数据并保存"""
    # 确保按 MIWV 降序排列
    df_sorted = df.sort_values(by='miwv', ascending=False)
    
    ratios = [0.01, 0.05, 0.10, 0.20] # 1%, 5%, 10%, 20%
    
    stats = []
    
    for ratio in ratios:
        top_k = int(len(df) * ratio)
        if top_k == 0: top_k = 1
        
        subset_df = df_sorted.head(top_k)
        
        # 获取阈值
        threshold = subset_df.iloc[-1]['miwv']
        
        # 保存为 JSON
        subset_file = os.path.join(output_dir, f'subset_top_{int(ratio*100)}percent.json')
        
        # 将 DataFrame 转回 list dict 格式保存
        subset_data = subset_df.to_dict(orient='records')
        with open(subset_file, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, ensure_ascii=False, indent=2)
            
        stats.append({
            "Ratio": f"{int(ratio*100)}%",
            "Count": top_k,
            "Min MIWV (Threshold)": round(threshold, 4),
            "Avg MIWV": round(subset_df['miwv'].mean(), 4)
        })
        
    return pd.DataFrame(stats)

def print_case_studies(df):
    """打印典型的 Hard Case 和 Easy Case"""
    df_sorted = df.sort_values(by='miwv', ascending=False)
    
    print("\n" + "="*50)
    print("【Case Study: High MIWV (最有价值/最难的数据)】")
    print("含义：给了例子后，模型反而更困惑 (Loss One > Loss Zero)")
    print("="*50)
    
    top_case = df_sorted.iloc[0]
    print(f"File: {top_case['file']}")
    print(f"MIWV: {top_case['miwv']:.4f} (Zero: {top_case['loss_zero']:.4f} -> One: {top_case['loss_one']:.4f})")
    print(f"Query: {top_case['query']}")
    print(f"Neighbor File (One-shot source): {top_case['neighbor_file']}")
    
    print("\n" + "="*50)
    print("【Case Study: Low MIWV (最简单/冗余的数据)】")
    print("含义：给了例子后，模型表现大幅提升 (Loss One < Loss Zero)，说明模型能很好地类比")
    print("="*50)
    
    bottom_case = df_sorted.iloc[-1]
    print(f"File: {bottom_case['file']}")
    print(f"MIWV: {bottom_case['miwv']:.4f} (Zero: {bottom_case['loss_zero']:.4f} -> One: {bottom_case['loss_one']:.4f})")
    print(f"Query: {bottom_case['query']}")

def main():
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        
    # 1. 加载数据
    df = load_data(CONFIG['input_file'])
    
    # 2. 基础统计
    print("\n>>> Basic Statistics:")
    print(df['miwv'].describe())
    
    # 3. 绘图
    plot_distribution(df, CONFIG['output_dir'])
    plot_loss_comparison(df, CONFIG['output_dir'])
    
    # 4. 筛选子集
    print("\n>>> Selecting Top-K Subsets...")
    stats_df = save_top_k_subsets(df, df, CONFIG['output_dir'])
    print(stats_df.to_string(index=False))
    
    # 5. 案例分析
    print_case_studies(df)
    
    print(f"\n>>> Analysis complete. Results saved in {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
