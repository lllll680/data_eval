import os
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置区域 =================
# 你的数据文件夹列表
DATA_DIRS = [
    "/data2/ly/dataset_eval/code_apply/", 
    # "/data2/ly/dataset_eval/code_apply_2/"
]
# 结果保存文件名
OUTPUT_FILE = "evaluation_summary_report.json"
MODEL_PATH = "/data2/Qwen/Qwen2.5-72B-Instruct"

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
                temperature=0.1, # 保持低温以获得稳定的JSON
                top_p=0.9
            )
            # 裁剪掉输入的prompt部分
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return self._parse_json(response)

    def _parse_json(self, response):
        """解析大模型返回的JSON，处理可能的格式杂质"""
        try:
            # 尝试找到第一个 { 和最后一个 }
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # 尝试解析列表（针对多样性评估）
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
        """
        question = data.get("question", "")
        records = data.get("execution_records", [])
        
        # 构建轨迹文本
        trajectory_text = ""
        for step in records:
            step_num = step.get("step", "?")
            tool = step.get("tool_name", "Unknown")
            reasoning = step.get("reasoning", "")
            req = json.dumps(step.get("tool_request", {}), ensure_ascii=False)
            res = str(step.get("tool_response", ""))[:300] # 截断过长输出
            
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
        """
        维度 4: 多样性/重复性评估 (Batch Level)
        一次性输入这一批数据的信息，让模型找出相似组。
        """
        if not batch_summaries:
            return {}

        # 构建简化的输入列表，节省token
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
    # 1. 准备工作
    judge = QwenJudge(MODEL_PATH)
    all_results = {} # 存储最终结果
    
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir): continue
        
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        print(f"\n处理目录: {data_dir} (共 {len(json_files)} 文件)")
        
        dir_summaries = [] # 用于多样性评估的摘要列表
        dir_file_map = {}  # 映射 run_id -> file_content
        
        # === 阶段 1: 逐个文件进行个体打分 ===
        for file_path in tqdm(json_files, desc="Individual Eval"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 获取或生成 run_id
                run_id = data.get("run_id")
                if not run_id:
                    run_id = os.path.basename(file_path).replace(".json", "")
                
                # 1. 调用大模型进行三维打分
                eval_res = judge.evaluate_single_trajectory(data)
                
                # 2. 准备多样性评估的数据摘要
                # 提取工具链字符串，例如 "Search->Calc"
                tool_chain = [r.get("tool_name", "") for r in data.get("execution_records", [])]
                tool_chain_str = "->".join(filter(None, tool_chain))
                
                summary_item = {
                    "run_id": run_id,
                    "question": data.get("question", "")[:100], # 截断问题避免过长
                    "tool_path": tool_chain_str
                }
                dir_summaries.append(summary_item)
                
                # 3. 暂存结果
                all_results[run_id] = {
                    "file_path": file_path,
                    "individual_scores": eval_res.get("scores", {"logical_coherence": 0, "tool_usage_validity": 0, "goal_efficiency": 0}),
                    "eval_reason": eval_res.get("reason", ""),
                    "similar_run_ids": [] # 稍后填充
                }
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # === 阶段 2: 批量进行多样性/重复性评估 ===
        if dir_summaries:
            print(f"正在分析 {len(dir_summaries)} 条数据的路径多样性...")
            similarity_groups = judge.evaluate_diversity_batch(dir_summaries)
            
            # 解析结果并回填到 all_results
            # similarity_groups 类似: {"g1": ["id1", "id2"], "g2": ["id3", "id4", "id5"]}
            if similarity_groups:
                for group_name, ids in similarity_groups.items():
                    if isinstance(ids, list) and len(ids) > 1:
                        for r_id in ids:
                            if r_id in all_results:
                                # 将同组的其他ID加入列表 (排除自己)
                                others = [x for x in ids if x != r_id]
                                all_results[r_id]["similar_run_ids"] = others
            else:
                print("未发现明显的重复路径。")

    # === 阶段 3: 保存汇总报告 ===
    output_path = os.path.join(os.getcwd(), OUTPUT_FILE)
    
    # 计算整体统计信息
    total_runs = len(all_results)
    avg_logic = np.mean([r["individual_scores"].get("logical_coherence", 0) for r in all_results.values()]) if total_runs else 0
    
    final_output = {
        "meta_summary": {
            "total_count": total_runs,
            "avg_logical_coherence": round(avg_logic, 2),
            "note": "similar_run_ids populated by batch LLM comparison."
        },
        "details": all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 评估完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    process_evaluation()
