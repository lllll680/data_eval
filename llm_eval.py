import os
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =================配置区域=================
DATA_DIRS = [
    "/data2/ly/dataset_eval/code_apply/",
#     "/data2/ly/dataset_eval/code_apply_2/",
#     "/data2/ly/dataset_eval/code_apply_3/"
]
MODEL_PATH = "/data2/Qwen/Qwen2.5-72B-Instruct"

# 为了演示，如果数据量巨大，可以设置 SAMPLE_NUM 只评估前N条进行验证
# 设置为 None 则评估所有数据
SAMPLE_NUM = 50 
# =========================================

class AgentDataEvaluator:
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs
        self.files = self._load_files()
        print(f"共找到 {len(self.files)} 个数据文件。")

    def _load_files(self):
        files = []
        for d in self.data_dirs:
            # 递归查找所有 .json 文件
            files.extend(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
        return files

    def get_data_content(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    # --- 指标 1: 平均推理步数 ---
    def calculate_average_steps(self):
        step_counts = []
        for file_path in tqdm(self.files, desc="计算平均步数"):
            data = self.get_data_content(file_path)
            if data and "execution_records" in data:
                # 过滤掉空的 record，只计算实际步数
                records = data["execution_records"]
                if isinstance(records, list):
                    step_counts.append(len(records))
        
        if not step_counts:
            return 0, 0
            
        avg_steps = np.mean(step_counts)
        max_steps = np.max(step_counts)
        return avg_steps, max_steps

class QwenJudge:
    def __init__(self, model_path):
        print(f"正在加载模型: {model_path} ... (可能需要几分钟)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 使用 auto device map 自动分配显存，需保证显存足够
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()

    def construct_evaluation_prompt(self, data):
        """
        构建 LLM-as-a-Judge 的 Prompt。
        这是评估逻辑连贯性和数据质量的核心。
        """
        question = data.get("question", "")
        records = data.get("execution_records", [])
        
        # 将轨迹格式化为易读的文本
        trajectory_text = ""
        for i, rec in enumerate(records):
            trajectory_text += f"Step {i+1}:\n"
            trajectory_text += f"  Reasoning: {rec.get('reasoning', '')}\n"
            trajectory_text += f"  Tool Call: {rec.get('tool_name', '')} -> {json.dumps(rec.get('tool_request', {}), ensure_ascii=False)}\n"
            trajectory_text += f"  Tool Output: {str(rec.get('tool_response', ''))[:200]}...\n" # 截断过长的输出
            trajectory_text += "-" * 20 + "\n"

        # === 核心 Prompt 设计 ===
        # 参考了 G-Eval 和 AgentBench 的打分逻辑
        prompt = f"""
### Role
You are an expert AI Assistant Evaluator. Your task is to evaluate the quality of an Agent's execution trajectory based on a specific User Question.

### Input Data
**User Question:** 
{question}

**Agent Execution Trajectory:**
{trajectory_text}

### Evaluation Criteria
Please score the trajectory on a scale of 1 to 10 for the following three dimensions. Then calculate a Weighted Final Score.

1. **Logical Coherence (Weight: 0.4)**
   - Does the reasoning in each step logically follow from the previous step and tool outputs?
   - Is the plan clear, or is the agent randomly trying tools?
   - Are there any contradictions between the reasoning and the action?

2. **Tool Usage Validity (Weight: 0.3)**
   - Are the selected tools appropriate for the current sub-goal?
   - Are the parameters generated for the tools correct and reasonable?

3. **Goal Efficiency (Weight: 0.3)**
   - Did the agent make progress towards solving the user's question?
   - Is the trajectory concise, or does it contain unnecessary redundant steps?

### Output Format
You must output a valid JSON object strictly following this format, with no extra text:
{{
    "analysis": "Brief justification for the scores...",
    "scores": {{
        "logical_coherence": <float 1-10>,
        "tool_usage_validity": <float 1-10>,
        "goal_efficiency": <float 1-10>
    }},
    "weighted_final_score": <float 1-10>
}}
"""
        return prompt

    def evaluate_one(self, data):
        prompt = self.construct_evaluation_prompt(data)
        messages = [
            {"role": "system", "content": "You are a helpful and rigorous AI evaluator."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.2, # 低温度保证评分稳定性
                top_p=0.9
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return self._parse_json(response)

    def _parse_json(self, response):
        # 简单的清理和解析逻辑，防止模型输出 ```json ... ```
        clean_str = response.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_str)
        except:
            print(f"Warning: Failed to parse JSON. Response was: {clean_str[:50]}...")
            return None

# ================= 主执行逻辑 =================
if __name__ == "__main__":
    # 1. 统计计算
    evaluator = AgentDataEvaluator(DATA_DIRS)
    avg_steps, max_steps = evaluator.calculate_average_steps()
    
    print("\n" + "="*40)
    print(f"📊 基础统计 (Basic Statistics)")
    print("="*40)
    print(f"平均推理步数 (Avg Steps): {avg_steps:.2f}")
    print(f"最大推理步数 (Max Steps): {max_steps}")
    print("="*40 + "\n")

    # 2. 模型打分 (如果显存不够，请注释掉这部分)
    # 只有当样本量 > 0 时才运行
    if evaluator.files:
        judge = QwenJudge(MODEL_PATH)
        
        scores_log = []
        files_to_eval = evaluator.files[:SAMPLE_NUM] if SAMPLE_NUM else evaluator.files
        
        print(f"开始使用 Qwen-72B 进行打分，共 {len(files_to_eval)} 条数据...")
        
        for file_path in tqdm(files_to_eval):
            data = evaluator.get_data_content(file_path)
            if not data: continue
            
            result = judge.evaluate_one(data)
            if result:
                scores_log.append(result["weighted_final_score"])
        
        if scores_log:
            avg_score = np.mean(scores_log)
            print("\n" + "="*40)
            print(f"🧠 模型评分结果 (LLM-as-a-Judge Evaluation)")
            print("="*40)
            print(f"评估模型: Qwen2.5-72B-Instruct")
            print(f"评估样本数: {len(scores_log)}")
            print(f"加权综合得分 (Weighted Final Score): {avg_score:.2f} / 10.0")
            print("="*40)
        else:
            print("未生成有效评分。")
