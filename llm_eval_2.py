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
    # "/data2/ly/dataset_eval/code_apply_3/"
]

MODEL_PATH = "/data2/Qwen/Qwen2.5-72B-Instruct"

# 输出文件的名称（将保存在每个对应的 data_dir 下）
OUTPUT_FILENAME = "evaluation_report.json"
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
                start_list = response.find('[')
                end_list = response.rfind(']') + 1
                if start_list != -1 and end_list != -1:
                    json_str = response[start_list:end_list]
                    return json.loads(json_str)
        except Exception as e:
            print(f"JSON Parse Error: {e}\nRaw Content: {response[:100]}...")
        return {}

    def evaluate_single_trajectory(self, data):
        """维度 1, 2, 3: 针对单条数据的个体评估"""
        question = data.get("question", "")
        records = data.get("execution_records", [])
        
        trajectory_text = ""
        for step in records:
            step_num = step.get("step", "?")
            tool = step.get("tool_name", "Unknown")
            reasoning = step.get("reasoning", "")
            req = json.dumps(step.get("tool_request", {}), ensure_ascii=False)
            res = str(step.get("tool_response", ""))[:300] 
            
            trajectory_text += f"Step {step_num}:\n"
            trajectory_text += f"  Reasoning: {reasoning}\n"
            trajectory_text += f"  Action: {tool} -> {req}\n"
            trajectory_text += f"  Observation: {res}...\n\n"

        prompt = f"""
### Task
Evaluate the AI Agent's execution trajectory for the given User Question based on 3 dimensions.

### Data
**User Question:** {question}
**Trajectory:**
{trajectory_text}

### Dimensions & Criteria (Score 1-10)
1. **Logical Coherence**: Does the reasoning make sense? Is the plan consistent? No hallucinations?
2. **Tool Usage Validity**: Are the chosen tools correct for the sub-tasks? Are parameters accurate?
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
        """维度 4: 多样性/重复性评估 (Batch Level)"""
        if not batch_summaries:
            return {}

        input_list_str = json.dumps(batch_summaries, indent=2, ensure_ascii=False)

        prompt = f"""
### Task
You are a Data Diversity Analyst. I will provide a list of {len(batch_summaries)} agent execution summaries.
Your goal is to identify **groups of runs that are semantically similar**.

Two runs are "similar" if:
1. They address the exact same or very similar question.
2. They use the same sequence of tools to solve it (Logic path is identical).

### Input Data
{input_list_str}

### Output
Return a JSON object where keys are "group_1", "group_2", etc., and values are lists of `run_id`s that are similar.
Runs that are unique should NOT be included in the output.

Example Output format:
{{
    "similar_group_1": ["run_id_A", "run_id_B"],
    "similar_group_2": ["run_id_X", "run_id_Y", "run_id_Z"]
}}
If no similarities are found, return empty {{}}.
"""
        return self._call_model(prompt)

def process_evaluation():
    # 1. 加载模型 (只需加载一次)
    judge = QwenJudge(MODEL_PATH)
    
    # 2. 遍历每个目录，独立处理
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"跳过不存在的路径: {data_dir}")
            continue
        
        # 获取该目录下的所有 json 文件
        # 注意：排除掉我们要生成的报告文件本身，防止重复读取
        all_files = glob.glob(os.path.join(data_dir, "*.json"))
        json_files = [f for f in all_files if OUTPUT_FILENAME not in f]
        
        print(f"\n" + "="*50)
        print(f"处理目录: {data_dir}")
        print(f"发现文件: {len(json_files)} 个")
        print("="*50)
        
        # === 核心修改点：变量初始化移到循环内部 ===
        dir_results = {}   # 当前目录的评估结果
        dir_summaries = [] # 当前目录的多样性摘要
        # ======================================

        # === 阶段 1: 逐个文件进行个体打分 ===
        for file_path in tqdm(json_files, desc=f"Eval {os.path.basename(data_dir)}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                run_id = data.get("run_id")
                if not run_id:
                    run_id = os.path.basename(file_path).replace(".json", "")
                
                # 1.1 个体评估
                eval_res = judge.evaluate_single_trajectory(data)
                
                # 1.2 收集摘要
                tool_chain = [r.get("tool_name", "") for r in data.get("execution_records", [])]
                tool_chain_str = "->".join(filter(None, tool_chain))
                
                summary_item = {
                    "run_id": run_id,
                    "question": data.get("question", "")[:100],
                    "tool_path": tool_chain_str
                }
                dir_summaries.append(summary_item)
                
                # 1.3 存入当前目录结果集
                dir_results[run_id] = {
                    "file_path": file_path,
                    "individual_scores": eval_res.get("scores", {"logical_coherence": 0, "tool_usage_validity": 0, "goal_efficiency": 0}),
                    "eval_reason": eval_res.get("reason", ""),
                    "similar_run_ids": [] 
                }
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # === 阶段 2: 批量进行多样性评估 ===
        if dir_summaries:
            print(f"正在分析 {data_dir} 的路径多样性...")
            similarity_groups = judge.evaluate_diversity_batch(dir_summaries)
            
            if similarity_groups:
                for group_name, ids in similarity_groups.items():
                    if isinstance(ids, list) and len(ids) > 1:
                        for r_id in ids:
                            if r_id in dir_results:
                                others = [x for x in ids if x != r_id]
                                dir_results[r_id]["similar_run_ids"] = others

        # === 阶段 3: 计算统计并保存 ===
        # 这里的统计仅针对当前目录
        total_runs = len(dir_results)
        if total_runs > 0:
            avg_logic = np.mean([r["individual_scores"].get("logical_coherence", 0) for r in dir_results.values()])
        else:
            avg_logic = 0
        
        final_output = {
            "meta_summary": {
                "dataset_path": data_dir,
                "total_count": total_runs,
                "avg_logical_coherence": round(avg_logic, 2),
            },
            "details": dir_results
        }
        
        # 拼接输出路径：直接保存在当前处理的文件夹下
        output_path = os.path.join(data_dir, OUTPUT_FILENAME)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        print(f"✅ 目录评估完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    process_evaluation()
