import json
import os
import pandas as pd
import numpy as np

def analyze_miwv_results(json_file, output_summary_file):
    # 1. 加载数据
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # 2. 提取文件夹名称作为类别
    # 假设你的路径格式是 .../folder_name/file.json
    # 我们取文件所在目录的最后一级文件夹名作为标识符
    df['folder'] = df['file'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    # 3. 全局统计分析 (ALL)
    global_stats = {
        "Group": "ALL",
        "Sample_Count": len(df),
        "MIWV_Mean": df['miwv'].mean(),
        "MIWV_Std": df['miwv'].std(),
        "MIWV_Min": df['miwv'].min(),
        "MIWV_Max": df['miwv'].max(),
        "MIWV_Median": df['miwv'].median(),
        "Loss0_Mean": df['loss_zero'].mean(),
        "Loss1_Mean": df['loss_one'].mean()
    }

    # 4. 分文件夹统计分析 (Per-Folder)
    folder_stats = []
    # 按文件夹分组
    grouped = df.groupby('folder')
    
    for folder_name, group in grouped:
        stats = {
            "Group": folder_name,
            "Sample_Count": len(group),
            "MIWV_Mean": group['miwv'].mean(),
            "MIWV_Std": group['miwv'].std(),
            "MIWV_Min": group['miwv'].min(),
            "MIWV_Max": group['miwv'].max(),
            "MIWV_Median": group['miwv'].median(),
            "Loss0_Mean": group['loss_zero'].mean(),
            "Loss1_Mean": group['loss_one'].mean(),
            # 记录该文件夹对应的 Query (取第一个样本的)
            "Query_Sample": group['query'].iloc[0][:50] + "..." 
        }
        folder_stats.append(stats)

    # 5. 整合结果
    summary = {
        "global": global_stats,
        "per_folder": folder_stats
    }

    # 6. 打印美化后的结果到控制台
    print("\n" + "="*30 + " GLOBAL STATS " + "="*30)
    print(pd.Series(global_stats).to_string())

    print("\n" + "="*30 + " PER FOLDER STATS " + "="*30)
    summary_df = pd.DataFrame(folder_stats)
    # 按照 MIWV 均值降序排列，查看哪组 Query 的 Response 质量更高（或更容易受上下文启发）
    summary_df = summary_df.sort_values(by="MIWV_Mean", ascending=False)
    print(summary_df[["Group", "Sample_Count", "MIWV_Mean", "MIWV_Std", "MIWV_Max"]].to_string(index=False))

    # 7. 保存到 JSON
    with open(output_summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    print(f"\n>>> Summary saved to: {output_summary_file}")

    # 8. (可选) 找出每个文件夹中 MIWV 最高（最有价值/最难）的前 3 个文件
    print("\n" + "="*30 + " TOP 3 SAMPLES PER FOLDER " + "="*30)
    for folder_name, group in grouped:
        top_3 = group.nlargest(3, 'miwv')
        print(f"\nFolder: {folder_name}")
        for _, row in top_3.iterrows():
            print(f"  - MIWV: {row['miwv']:.4f} | File: {os.path.basename(row['file'])}")

if __name__ == "__main__":
    # 修改为你的实际路径
    INPUT_JSON = "/data2/ly/dataset_eval/miwv_results.json"
    OUTPUT_JSON = "/data2/ly/dataset_eval/miwv_summary.json"
    
    analyze_miwv_results(INPUT_JSON, OUTPUT_JSON)
