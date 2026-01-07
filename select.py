import json
import pandas as pd
import os
import glob
from sklearn.preprocessing import MinMaxScaler

# ================= 配置区域 =================
# 请确保这些路径指向你实际的文件
MIWV_FILE = "/data2/ly/dataset_eval/miwv_results.json"
IFD_FILE = "/raid/data/ly/data/dataset/ifd_results.jsonl"
SELECTIT_FILE = "/raid/data/ly/data/dataset/data_eval/selectit_scores.csv"
LLM_EVAL_DIR = "/raid/data/ly/data/dataset/data_eval/reports/" # 包含8个json的文件夹

TOP_N = 4 # 每个文件夹选几条
# ===========================================

def load_data():
    # 1. 加载 MIWV (JSON)
    with open(MIWV_FILE, 'r', encoding='utf-8') as f:
        miwv_data = json.load(f)
    df_miwv = pd.DataFrame(miwv_data)[['file', 'miwv']]
    df_miwv.rename(columns={'file': 'file_path'}, inplace=True)

    # 2. 加载 IFD (JSONL)
    ifd_list = []
    with open(IFD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            ifd_list.append(json.loads(line))
    df_ifd = pd.DataFrame(ifd_list)[['file_path', 'ifd_score']]

    # 3. 加载 SelectIT (CSV)
    df_sit = pd.read_csv(SELECTIT_FILE)[['file_path', 'selectit_score']]

    # 4. 加载 LLM_Eval (多个JSON)
    llm_list = []
    report_files = glob.glob(os.path.join(LLM_EVAL_DIR, "*.json"))
    for rf in report_files:
        with open(rf, 'r', encoding='utf-8') as f:
            report = json.load(f)
            folder_name = report['meta_summary']['dataset_path'].strip('/').split('/')[-1]
            for run_id, detail in report['details'].items():
                llm_list.append({
                    'file_path': detail['file_path'],
                    'folder': folder_name,
                    'logic': detail['individual_scores']['logical_coherence'],
                    'tool': detail['individual_scores']['tool_usage_validity'],
                    'efficiency': detail['individual_scores']['goal_efficiency'],
                    # 提取工具链用于去重
                    'tool_chain': " -> ".join([run_id]) # 简化处理，实际可以从detail中获取更详细的path
                })
    df_llm = pd.DataFrame(llm_list)

    # 5. 合并所有 DataFrame
    # 注意：file_path 可能需要统一为绝对路径以防对不齐
    df_llm['file_path'] = df_llm['file_path'].apply(os.path.abspath)
    df_miwv['file_path'] = df_miwv['file_path'].apply(os.path.abspath)
    df_ifd['file_path'] = df_ifd['file_path'].apply(os.path.abspath)
    df_sit['file_path'] = df_sit['file_path'].apply(os.path.abspath)

    df_final = df_llm.merge(df_miwv, on='file_path', how='inner')
    df_final = df_final.merge(df_ifd, on='file_path', how='inner')
    df_final = df_final.merge(df_sit, on='file_path', how='inner')
    
    return df_final

def rank_and_select(df):
    # 1. 硬性过滤：逻辑和工具使用必须及格
    df = df[(df['logic'] >= 7.0) & (df['tool'] >= 7.0)].copy()

    # 2. 标准化指标 (0-1)
    scaler = MinMaxScaler()
    # 注意：MIWV 越大越好（负值越接近0越好），Scaler会自动处理
    cols_to_norm = ['logic', 'ifd_score', 'miwv', 'selectit_score']
    df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])

    # 3. 计算综合得分 (自定义权重)
    df['final_score'] = (
        df['logic'] * 0.40 + 
        df['ifd_score'] * 0.25 + 
        df['miwv'] * 0.20 + 
        df['selectit_score'] * 0.15
    )

    # 4. 分文件夹排序并选择 Top N
    # 我们不仅要分高，还要保证同一个文件夹下 tool_chain 尽量不重复
    selected_samples = []
    
    for folder, group in df.groupby('folder'):
        # 先按得分降序排
        sorted_group = group.sort_values(by='final_score', ascending=False)
        
        picked = []
        seen_paths = set()
        
        for _, row in sorted_group.iterrows():
            if len(picked) >= TOP_N:
                break
            
            # 简单的多样性检查：这里假设 tool_chain 是我们在 LLM_Eval 中提取的
            # 如果你有更精确的工具链字段，请替换 'tool_chain'
            t_path = row['tool_chain']
            if t_path not in seen_paths:
                picked.append(row)
                seen_paths.add(t_path)
        
        # 如果去重后不够N条，再从未选中的里面按分数补齐
        if len(picked) < TOP_N:
            remaining = sorted_group[~sorted_group['file_path'].isin([p['file_path'] for p in picked])]
            for _, row in remaining.head(TOP_N - len(picked)).iterrows():
                picked.append(row)
                
        selected_samples.extend(picked)

    return pd.DataFrame(selected_samples), df

# 运行
try:
    merged_df = load_data()
    print(f">>> 成功对齐 {len(merged_df)} 条数据。")
    
    top_samples, full_ranked_df = rank_and_select(merged_df)
    
    # 保存结果
    top_samples.to_csv("top_test_set_samples.csv", index=False)
    full_ranked_df.to_csv("all_samples_ranked.csv", index=False)
    
    print(f"\n>>> 筛选完成！已选出 {len(top_samples)} 条优秀样本。")
    print(f">>> 结果已保存至 top_test_set_samples.csv")
    
    # 打印每个文件夹选出的平均分
    print("\n--- 每个文件夹筛选样本的平均逻辑得分 ---")
    print(top_samples.groupby('folder')['logic'].mean())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
