import os
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置区域 =================
DATA_DIRS = [
    "/data2/ly/dataset_eval/code_apply/", 
    "/data2/ly/dataset_eval/code_apply_2/",
]

# 评估模型路径
MODEL_PATH = "/data2/Qwen/Qwen2.5-72B-Instruct"

# 输出报告目录
OUTPUT_DIR = "/raid/data/ly/data/dataset/data_eval/reports"

# 报告文件后缀名
OUTPUT_FILENAME = "evaluation_report.json"

# 需要跳过的特定文件名
SKIP_FILENAMES = ["question_info.json", "batch_summary.json"]
# ===========================================

class QwenJudge:
    def __init__(self, model_path):
        print(f"正在加载模型: {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.model.eval()

    def _call_model(self, prompt):
        """通用的大模型调用函数"""
        messages = [
            {"role": "system", "content": "You are an expert AI Assistant Evaluator."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.1, 
                top_p=0.9
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return self._parse_json(response)

    def _parse_json(self, response):
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # 兼容列表格式的返回
                start_list = response.find('[')
                end_list = response.rfind(']') + 1
                if start_list != -1 and end_list != -1:
                    json_str = response[start_list:end_list]
                    return json.loads(json_str)
        except Exception as e:
            print(f"JSON Parse Error: {e}\nRaw Content: {response[:100]}...")
        return {}

    def evaluate_single_trajectory(self, data):
        """
        维度 1, 2, 3: 针对单条数据的个体评估
        适配数据格式: query + response -> stepX -> cot/coa
        """
        query = data.get("query", "")
        response_steps = data.get("response", [])
        
        trajectory_text = ""
        # 遍历 response 中的每一个 step 字典
        for idx, step_item in enumerate(response_steps):
            # 获取 step1, step2 这种动态 key
            step_key = list(step_item.keys())[0]
            content = step_item[step_key]
            
            cot = content.get("cot", "")
            trajectory_text += f"Step {idx + 1} (Analysis): {cot}\n"
            
            # 遍历 coa 列表
            if 'coa' in content:
                for coa_idx, coa in enumerate(content['coa']):
                    action = coa.get("action", {})
                    action_name = action.get("name", "Unknown")
                    args = json.dumps(action.get("args", {}), ensure_ascii=False)
                    
                    # observation 处理
                    obs = coa.get("observation", "")
                    if isinstance(obs, (dict, list)):
                        obs_str = json.dumps(obs, ensure_ascii=False)[:300]
                    else:
                        obs_str = str(obs)[:300]

                    trajectory_text += f"  Action {coa_idx+1}: {action_name}({args})\n"
                    trajectory_text += f"  Observation {coa_idx+1}: {obs_str}...\n"
            trajectory_text += "-"*20 + "\n"

        prompt = f"""
### Task
Evaluate the AI Agent's execution trajectory for the given User Question based on 3 dimensions.

### Data
**User Question:** {query}
**Trajectory:**
{trajectory_text}

### Dimensions & Criteria (Score 1-10)
1. **Logical Coherence**: Does the reasoning (CoT) make sense? Is the plan consistent? No hallucinations?
2. **Tool Usage Validity**: Are the chosen tools (Action) correct for the sub-tasks? Are parameters accurate?
3. **Goal Efficiency**: Did it solve the problem directly without loops or redundant steps?

### Output
Return a strictly valid JSON object:
{{
    "scores": {{
        "logical_coherence": <float>,
        "tool_usage_validity": <float>,
        "goal_efficiency": <float>
    }},
    "reason": "Short explanation within 50 words."
}}
"""
        return self._call_model(prompt)

    def evaluate_diversity_batch(self, batch_summaries):
        """维度 4: 多样性评估 (保持原样)"""
        if not batch_summaries:
            return {}
        input_list_str = json.dumps(batch_summaries, indent=2, ensure_ascii=False)
        prompt = f"""
### Task
Identify groups of runs that are semantically similar from {len(batch_summaries)} summaries.
Similar means: Same User Question + Same Tool Sequence.

### Input Data
{input_list_str}

### Output
Return JSON: {{"similar_group_1": ["run_id_A", "run_id_B"], ...}}. If none, return {{}}.
"""
        return self._call_model(prompt)

def process_evaluation():
    judge = QwenJudge(MODEL_PATH)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"跳过不存在的路径: {data_dir}")
            continue
        
        # 查找所有 JSON 文件
        all_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
        
        json_files = []
        for f in all_files:
            fname = os.path.basename(f)
            # 过滤跳过列表及报告文件
            if fname not in SKIP_FILENAMES and OUTPUT_FILENAME not in fname:
                json_files.append(f)
        
        print(f"\n" + "="*50)
        print(f"处理目录: {data_dir}")
        print(f"有效样本数: {len(json_files)}")
        print("="*50)
        
        dir_results = {}   
        dir_summaries = [] 

        for file_path in tqdm(json_files, desc=f"Eval {os.path.basename(data_dir)}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 【关键点】适配新格式校验：检查是否有 query 和 response
                if "query" not in data or "response" not in data:
                    continue

                # 生成 Run ID
                run_id = data.get("run_id") or os.path.basename(file_path).replace(".json", "")
                
                # 1.1 个体评估
                eval_res = judge.evaluate_single_trajectory(data)
                
                # 1.2 收集路径摘要（提取所有 action name 连成串）
                all_tools = []
                for step_item in data.get("response", []):
                    step_key = list(step_item.keys())[0]
                    content = step_item[step_key]
                    if 'coa' in content:
                        for coa in content['coa']:
                            all_tools.append(coa.get("action", {}).get("name", ""))
                
                summary_item = {
                    "run_id": run_id,
                    "question": data.get("query", "")[:100],
                    "tool_path": " -> ".join(filter(None, all_tools))
                }
                dir_summaries.append(summary_item)
                
                # 1.3 存储结果
                dir_results[run_id] = {
                    "file_path": file_path,
                    "individual_scores": eval_res.get("scores", {
                        "logical_coherence": 0, "tool_usage_validity": 0, "goal_efficiency": 0
                    }),
                    "eval_reason": eval_res.get("reason", ""),
                    "similar_run_ids": [] 
                }
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # === 多样性分析 ===
        if dir_summaries:
            print(f"正在分析该目录的多样性...")
            similarity_groups = judge.evaluate_diversity_batch(dir_summaries)
            if similarity_groups:
                for group_name, ids in similarity_groups.items():
                    if isinstance(ids, list) and len(ids) > 1:
                        for r_id in ids:
                            if r_id in dir_results:
                                others = [x for x in ids if x != r_id]
                                dir_results[r_id]["similar_run_ids"] = others

        # === 保存当前目录的结果 ===
        total_runs = len(dir_results)
        if total_runs > 0:
            avg_logic = np.mean([r["individual_scores"].get("logical_coherence", 0) for r in dir_results.values()])
            avg_tool = np.mean([r["individual_scores"].get("tool_usage_validity", 0) for r in dir_results.values()])
        else:
            avg_logic, avg_tool = 0, 0
        
        final_output = {
            "meta_summary": {
                "dataset_path": data_dir,
                "total_count": total_runs,
                "avg_logical_coherence": round(float(avg_logic), 2),
                "avg_tool_usage_validity": round(float(avg_tool), 2),
            },
            "details": dir_results
        }
        
        # 生成输出文件名：目录名_evaluation_report.json
        dir_clean_name = data_dir.strip('/').split('/')[-1]
        output_path = os.path.join(OUTPUT_DIR, f"{dir_clean_name}_{OUTPUT_FILENAME}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        print(f"✅ 目录 {dir_clean_name} 评估完成！Total Count: {total_runs}, 保存至: {output_path}")

if __name__ == "__main__":
    process_evaluation()
