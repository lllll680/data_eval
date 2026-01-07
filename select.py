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

TOP_N = 4 # 每个文件夹选出的样本数量
# ===========================================

def load_data():
    print(">>> 正在加载并对齐各项指标数据...")
    
    # 1. 加载 MIWV (JSON)
    with open(MIWV_FILE, 'r', encoding='utf-8') as f:
        df_miwv = pd.DataFrame(json.load(f))[['file', 'miwv']]
    df_miwv.rename(columns={'file': 'file_path', 'miwv': 'miwv_raw'}, inplace=True)

    # 2. 加载 IFD (JSONL) - 包含 response_steps
    ifd_list = []
    with open(IFD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            ifd_list.append(json.loads(line))
    # 提取 ifd_score 和 response_steps
    df_ifd = pd.DataFrame(ifd_list)[['file_path', 'ifd_score', 'response_steps']]
    df_ifd.rename(columns={'ifd_score': 'ifd_raw'}, inplace=True)

    # 3. 加载 SelectIT (CSV)
    df_sit = pd.read_csv(SELECTIT_FILE)[['file_path', 'selectit_score']]
    df_sit.rename(columns={'selectit_score': 'selectit_raw'}, inplace=True)

    # 4. 加载 LLM_Eval (多个文件夹下的 JSON 报告)
    llm_list = []
    report_files = glob.glob(os.path.join(LLM_EVAL_DIR, "*.json"))
    for rf in report_files:
        with open(rf, 'r', encoding='utf-8') as f:
            report = json.load(f)
            # 提取文件夹名
            folder_name = report['meta_summary']['dataset_path'].strip('/').split('/')[-1]
            for run_id, detail in report['details'].items():
                llm_list.append({
                    'file_path': os.path.abspath(detail['file_path']),
                    'folder': folder_name,
                    'logic_raw': detail['individual_scores']['logical_coherence'],
                    'tool_raw': detail['individual_scores']['tool_usage_validity'],
                    'eff_raw': detail['individual_scores']['goal_efficiency']
                })
    df_llm = pd.DataFrame(llm_list)

    # 5. 统一路径格式并合并
    df_miwv['file_path'] = df_miwv['file_path'].apply(os.path.abspath)
    df_ifd['file_path'] = df_ifd['file_path'].apply(os.path.abspath)
    df_sit['file_path'] = df_sit['file_path'].apply(os.path.abspath)

    # 内连接合并所有指标
    df_final = df_llm.merge(df_miwv, on='file_path', how='inner')
    df_final = df_final.merge(df_ifd, on='file_path', how='inner')
    df_final = df_final.merge(df_sit, on='file_path', how='inner')
    
    return df_final

def rank_and_select(df):
    # 1. 硬性过滤：只考虑逻辑和工具调用表现较好的数据 (例如 >= 7.0)
    # 如果 Question 4 表现普遍较差导致被过滤光了，可以适当调低此阈值
    df_filtered = df[(df['logic_raw'] >= 7.0) & (df['tool_raw'] >= 7.0)].copy()
    
    if df_filtered.empty:
        print("Warning: 所有数据均未通过 logic/tool >= 7.0 的硬性过滤，请检查阈值设置。")
        return pd.DataFrame()

    # 2. 准备归一化指标（用于计算加权得分）
    # 注意：response_steps 不参与打分，只作为参考列
    cols_to_norm = ['logic_raw', 'tool_raw', 'ifd_raw', 'miwv_raw', 'selectit_raw']
    scaler = MinMaxScaler()
    
    # 执行归一化
    norm_values = scaler.fit_transform(df_filtered[cols_to_norm])
    df_filtered['logic_norm'] = norm_values[:, 0]
    df_filtered['tool_norm'] = norm_values[:, 1]
    df_filtered['ifd_norm'] = norm_values[:, 2]
    df_filtered['miwv_norm'] = norm_values[:, 3]
    df_filtered['sit_norm'] = norm_values[:, 4]

    # 3. 计算综合得分 (Final Score)
    # 权重分配：逻辑最高(0.4)，IFD次之(0.25)，MIWV与SelectIT平分剩余
    df_filtered['final_score'] = (
        df_filtered['logic_norm'] * 0.40 + 
        df_filtered['tool_norm'] * 0.10 + 
        df_filtered['ifd_norm'] * 0.25 + 
        df_filtered['miwv_norm'] * 0.15 + 
        df_filtered['sit_norm'] * 0.10
    )

    # 4. 每个文件夹（Query）内部排序并挑选 Top N
    selected_samples = []
    for folder, group in df_filtered.groupby('folder'):
        # 按综合得分从高到低排序
        top_group = group.sort_values(by='final_score', ascending=False).head(TOP_N)
        selected_samples.append(top_group)

    return pd.concat(selected_samples)

# ================= 主程序 =================
try:
    # 加载数据
    merged_df = load_data()
    print(f">>> 成功合并数据，共计 {len(merged_df)} 条记录。")

    # 执行筛选
    top_df = rank_and_select(merged_df)

    if not top_df.empty:
        # 定义输出列的顺序
        # 包含原始分数、response_steps 以及最终计算出的分值
        output_cols = [
            'folder', 
            'file_path', 
            'response_steps',  # 新增的参考指标
            'logic_raw', 
            'tool_raw', 
            'eff_raw',
            'ifd_raw', 
            'miwv_raw', 
            'selectit_raw', 
            'final_score'
        ]
        
        # 结果输出
        top_df[output_cols].to_csv("top_test_set_selection.csv", index=False)
        print(f"\n>>> 筛选完成！已从每个文件夹选出 Top {TOP_N} 的样本。")
        print(f">>> 结果已保存至: top_test_set_selection.csv")
        
        # 打印简单摘要
        summary = top_df.groupby('folder')['logic_raw'].mean()
        print("\n各文件夹入选样本的平均逻辑得分：")
        print(summary)
    else:
        print("未生成任何结果，请检查输入文件。")

except Exception as e:
    print(f"程序运行出错: {e}")
