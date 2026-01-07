import json
import pandas as pd
import os
import glob
from sklearn.preprocessing import MinMaxScaler

# ================= 配置区域 =================
MIWV_FILE = "/data2/ly/dataset_eval/miwv_results.json"
IFD_FILE = "/raid/data/ly/data/dataset/ifd_results.jsonl"
SELECTIT_FILE = "/raid/data/ly/data/dataset/data_eval/selectit_scores.csv"
LLM_EVAL_DIR = "/raid/data/ly/data/dataset/data_eval/reports/"
TOP_N = 4 
# ===========================================

def load_data():
    # 1. 加载 MIWV
    with open(MIWV_FILE, 'r', encoding='utf-8') as f:
        df_miwv = pd.DataFrame(json.load(f))[['file', 'miwv']]
    df_miwv.rename(columns={'file': 'file_path', 'miwv': 'miwv_raw'}, inplace=True)

    # 2. 加载 IFD
    ifd_list = []
    with open(IFD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            ifd_list.append(json.loads(line))
    df_ifd = pd.DataFrame(ifd_list)[['file_path', 'ifd_score']]
    df_ifd.rename(columns={'ifd_score': 'ifd_raw'}, inplace=True)

    # 3. 加载 SelectIT
    df_sit = pd.read_csv(SELECTIT_FILE)[['file_path', 'selectit_score']]
    df_sit.rename(columns={'selectit_score': 'selectit_raw'}, inplace=True)

    # 4. 加载 LLM_Eval 并提取真实的工具链
    llm_list = []
    report_files = glob.glob(os.path.join(LLM_EVAL_DIR, "*.json"))
    for rf in report_files:
        with open(rf, 'r', encoding='utf-8') as f:
            report = json.load(f)
            folder_name = report['meta_summary']['dataset_path'].strip('/').split('/')[-1]
            for run_id, detail in report['details'].items():
                # --- 这里是提取真实工具链的关键步骤 ---
                # 从文件路径重新读取原始数据来获取工具链，或者从report中获取（如果report存了）
                # 这里假设你的 report 结构中有类似 summary_item 存的 tool_path
                # 如果没有，我们暂且用 run_id 的前缀，建议此处根据你 report 实际有的字段调整
                llm_list.append({
                    'file_path': os.path.abspath(detail['file_path']),
                    'folder': folder_name,
                    'logic_raw': detail['individual_scores']['logical_coherence'],
                    'tool_raw': detail['individual_scores']['tool_usage_validity'],
                    'eff_raw': detail['individual_scores']['goal_efficiency'],
                    'tool_chain': detail.get('tool_path', run_id) # 这里的 tool_path 需要确保在 eval 代码里存了
                })
    df_llm = pd.DataFrame(llm_list)

    # 5. 合并
    df_miwv['file_path'] = df_miwv['file_path'].apply(os.path.abspath)
    df_ifd['file_path'] = df_ifd['file_path'].apply(os.path.abspath)
    df_sit['file_path'] = df_sit['file_path'].apply(os.path.abspath)

    df_final = df_llm.merge(df_miwv, on='file_path').merge(df_ifd, on='file_path').merge(df_sit, on='file_path')
    return df_final

def rank_and_select(df):
    # 过滤掉逻辑不及格的样本（10分制，低于7分不要）
    df_filtered = df[(df['logic_raw'] >= 7.0) & (df['tool_raw'] >= 7.0)].copy()
    
    # 准备归一化列（计算分时使用）
    cols_to_norm = ['logic_raw', 'tool_raw', 'ifd_raw', 'miwv_raw', 'selectit_raw']
    scaler = MinMaxScaler()
    
    # 创建计算用的副本，以免覆盖原始数据
    norm_values = scaler.fit_transform(df_filtered[cols_to_norm])
    df_filtered['logic_norm'] = norm_values[:, 0]
    df_filtered['tool_norm'] = norm_values[:, 1]
    df_filtered['ifd_norm'] = norm_values[:, 2]
    df_filtered['miwv_norm'] = norm_values[:, 3]
    df_filtered['sit_norm'] = norm_values[:, 4]

    # 综合得分：基于归一化后的值
    df_filtered['final_score'] = (
        df_filtered['logic_norm'] * 0.40 + 
        df_filtered['tool_norm'] * 0.10 + 
        df_filtered['ifd_norm'] * 0.20 + 
        df_filtered['miwv_norm'] * 0.15 + 
        df_filtered['sit_norm'] * 0.15
    )

    selected_samples = []
    for folder, group in df_filtered.groupby('folder'):
        sorted_group = group.sort_values(by='final_score', ascending=False)
        
        picked = []
        seen_chains = set()
        
        # 优先选工具链不同的
        for _, row in sorted_group.iterrows():
            if len(picked) >= TOP_N: break
            if row['tool_chain'] not in seen_chains:
                picked.append(row)
                seen_chains.add(row['tool_chain'])
        
        # 不够则按分数补齐
        if len(picked) < TOP_N:
            remaining = sorted_group[~sorted_group['file_path'].isin([p['file_path'] for p in picked])]
            for _, row in remaining.head(TOP_N - len(picked)).iterrows():
                picked.append(row)
        selected_samples.extend(picked)

    return pd.DataFrame(selected_samples)

# 执行
res_df = load_data()
top_df = rank_and_select(res_df)

# 只保留易读的列输出
output_cols = ['folder', 'file_path', 'logic_raw', 'tool_raw', 'ifd_raw', 'miwv_raw', 'selectit_raw', 'tool_chain', 'final_score']
top_df[output_cols].to_csv("top_test_set_final.csv", index=False)
print("筛选完成！已保存至 top_test_set_final.csv")
