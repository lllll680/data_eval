import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= 配置区域 =================
CONFIG = {
    # 数据文件夹列表
    "data_dirs": [
        "/data2/ly/dataset_eval/data",
    ],
    # 结果保存路径
    "output_file": "/data2/ly/dataset_eval/miwv_results.json",
    
    # 模型路径
    "llm_model_path": "/data2/Qwen/Qwen2.5-7B-Instruct", 
    "embedding_model_path": "/data2/Qwen/Qwen3-Embedding-4B", 
    
    # 硬件设置
    "device_map": "auto", 
    "dtype": torch.bfloat16, 
}
# ===========================================

class DataSample:
    def __init__(self, file_path, data):
        self.file_path = file_path
        self.query = data.get('query', '')
        self.raw_response = data.get('response', [])
        # 序列化后的完整文本
        self.full_text = "" 
        # 需要 mask 掉的字符区间列表 [(start, end), ...]
        self.mask_ranges = [] 
        
        self._serialize()

    def _serialize(self):
        """
        将 JSON 结构序列化为文本，兼容处理 dict 和 str 类型的 args/observation
        """
        # 1. 构建 Prompt 部分
        prompt_text = f"User: {self.query}\n\nAssistant:\n"
        self.full_text += prompt_text
        
        # 记录 Prompt 部分mask掉
        self.mask_ranges.append((0, len(prompt_text)))

        # 2. 构建 Response 部分
        if not isinstance(self.raw_response, list):
            # 如果 response 不是列表，尝试转为 str 直接处理（防御性编程）
            self.full_text += str(self.raw_response)
            return

        for step_idx, step_item in enumerate(self.raw_response):
            if not isinstance(step_item, dict):
                continue
            
            # 这里的 key 可能是 "step1", "step2" 等
            keys = list(step_item.keys())
            if not keys:
                continue
            key = keys[0]
            content = step_item[key]
            
            if not isinstance(content, dict):
                continue

            # --- Cot ---
            cot_content = content.get('cot', '')
            # 兼容 Cot 是列表的情况
            if isinstance(cot_content, list):
                cot_content = " ".join([str(c) for c in cot_content])
            
            cot_text = f"Step {step_idx+1} Cot: {cot_content}\n"
            self.full_text += cot_text
            
            # --- CoAs ---
            if 'coa' in content and isinstance(content['coa'], list):
                for coa in content['coa']:
                    if not isinstance(coa, dict):
                        continue
                        
                    # --- Action ---
                    action_obj = coa.get('action', {})
                    if isinstance(action_obj, dict):
                        action_name = action_obj.get('name', 'unknown')
                        raw_args = action_obj.get('args', {})
                        
                        # === 修改点1：兼容 args 是字符串或字典 ===
                        if isinstance(raw_args, dict):
                            args_str = ", ".join([f'{k}="{v}"' for k,v in raw_args.items()])
                        else:
                            # 如果 args 是字符串，直接用
                            args_str = str(raw_args)
                            
                        action_text = f"Step {step_idx+1} Action: {action_name}({args_str})\n"
                    else:
                        action_text = f"Step {step_idx+1} Action: {str(action_obj)}\n"
                        
                    self.full_text += action_text

                    # --- Observation ---
                    obs_data = coa.get('observation', [])
                    
                    # === 修改点2：兼容 observation 格式 ===
                    # 确保它是列表
                    if not isinstance(obs_data, list):
                        obs_data = [obs_data]
                    
                    obs_str_parts = []
                    for obs_item in obs_data:
                        if isinstance(obs_item, dict):
                            # 原来的逻辑：拼接 key: value
                            item_str = ", ".join([f"{k}: {v}" for k,v in obs_item.items()])
                        else:
                            # 如果是字符串，直接添加
                            item_str = str(obs_item)
                        obs_str_parts.append(f"- {item_str}")
                    
                    if obs_str_parts:
                        full_obs_str = "Step {} Observation:\n{}\n".format(step_idx+1, "\n".join(obs_str_parts))
                        
                        # === 记录 Mask ===
                        start_idx = len(self.full_text)
                        self.full_text += full_obs_str
                        end_idx = len(self.full_text)
                        
                        self.mask_ranges.append((start_idx, end_idx))

class MIWVCalculator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        
    def load_data(self):
        print(">>> Loading Data...")
        samples = []
        for d in self.config['data_dirs']:
            files = glob.glob(os.path.join(d, "**/*.json"), recursive=True)
            for f in files:
                # 跳过 question_info.json
                if os.path.basename(f) == "question_info.json":
                    continue
                try:
                    with open(f, 'r', encoding='utf-8') as fr:
                        data = json.load(fr)
                        # 检查必要字段是否存在，不存在则报错跳过
                        if 'query' not in data or 'response' not in data:
                            print(f"Skipping {f}: Missing query or response field")
                            continue
                        samples.append(DataSample(f, data))
                except Exception as e:
                    # 打印具体的错误文件和错误信息，方便排查
                    print(f"Error loading {f}: {e}")
        
        print(f">>> Loaded {len(samples)} samples.")
        return samples

    def get_embeddings(self, samples):
        print(">>> Loading Embedding Model...")
        tokenizer = AutoTokenizer.from_pretrained(self.config['embedding_model_path'], trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.config['embedding_model_path'], 
            trust_remote_code=True,
            device_map=self.config['device_map'],
            torch_dtype=self.config['dtype']
        ).eval()

        queries = [s.query for s in samples]
        embeddings = []
        
        batch_size = 32
        print(">>> Computing Embeddings...")
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size)):
                batch_texts = queries[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
                
                outputs = model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                else:
                    batch_emb = outputs[0]
                
                batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
                embeddings.append(batch_emb.cpu().numpy())
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return np.concatenate(embeddings, axis=0)

    def calculate_token_loss(self, text, mask_ranges, prefix_context=""):
        full_input_str = prefix_context + text
        
        inputs = self.tokenizer(
            full_input_str, 
            return_tensors="pt", 
            return_offsets_mapping=True,
            add_special_tokens=True
        ).to(self.model.device)
        
        input_ids = inputs.input_ids
        labels = input_ids.clone()
        offsets = inputs.offset_mapping[0].cpu().numpy()
        
        prefix_len = len(prefix_context)
        
        for idx, (start, end) in enumerate(offsets):
            if start < prefix_len:
                labels[0, idx] = -100
                continue
                
            rel_start = start - prefix_len
            
            is_masked = False
            for (m_start, m_end) in mask_ranges:
                if m_start <= rel_start < m_end:
                    is_masked = True
                    break
            
            if is_masked:
                labels[0, idx] = -100
                
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=labels)
        
        return outputs.loss.item()

    def run(self):
        samples = self.load_data()
        if not samples:
            print("ERROR: No valid samples found. Please check your data paths and JSON format.")
            return

        embeddings = self.get_embeddings(samples)
        
        print(">>> Computing Similarity Matrix...")
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, -1)
        most_sim_indices = np.argmax(sim_matrix, axis=1)

        print(f">>> Loading LLM: {self.config['llm_model_path']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['llm_model_path'], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['llm_model_path'],
            device_map=self.config['device_map'],
            torch_dtype=self.config['dtype'],
            trust_remote_code=True
        ).eval()

        results = []
        
        print(">>> Calculating MIWV Scores...")
        for i, sample in enumerate(tqdm(samples)):
            try:
                loss_zero = self.calculate_token_loss(sample.full_text, sample.mask_ranges, prefix_context="")
                
                neighbor_idx = most_sim_indices[i]
                neighbor = samples[neighbor_idx]
                one_shot_context = neighbor.full_text + "\n\n"
                
                loss_one = self.calculate_token_loss(sample.full_text, sample.mask_ranges, prefix_context=one_shot_context)
                
                miwv = loss_one - loss_zero
                
                results.append({
                    "file": sample.file_path,
                    "query": sample.query,
                    "loss_zero": loss_zero,
                    "loss_one": loss_one,
                    "miwv": miwv,
                    "neighbor_file": neighbor.file_path
                })
                
            except Exception as e:
                print(f"Error processing sample {i} ({sample.file_path}): {e}")
                torch.cuda.empty_cache()

        results.sort(key=lambda x: x['miwv'], reverse=True)
        
        with open(self.config['output_file'], 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f">>> Done! Results saved to {self.config['output_file']}")

if __name__ == "__main__":
    calculator = MIWVCalculator(CONFIG)
    calculator.run()
