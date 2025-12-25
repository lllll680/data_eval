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
        # "/raid/data/ly/data2",
        # "/raid/data/ly/data3"
    ],
    # 结果保存路径
    "output_file": "/data2/ly/dataset_eval/miwv_results.json",
    
    # 模型路径
    "llm_model_path": "/data2/Qwen/Qwen2.5-7B-Instruct", # 替换为你本地的 Qwen2.5-72B 路径
    "embedding_model_path": "/data2/Qwen/Qwen3-Embedding-4B", # 替换为你本地的 Embedding 模型路径
    
    # 硬件设置
    "device_map": "auto",  # 自动分配到 4张 A100
    "dtype": torch.bfloat16, # A100 推荐使用 bf16
}
# ===========================================

class DataSample:
    def __init__(self, file_path, data):
        self.file_path = file_path
        self.query = data['query']
        self.raw_response = data['response']
        # 序列化后的完整文本
        self.full_text = "" 
        # 需要 mask 掉的字符区间列表 [(start, end), ...]
        self.mask_ranges = [] 
        
        self._serialize()

    def _serialize(self):
        """
        将 JSON 结构序列化为文本，并记录 Observation 的位置用于 Mask。
        """
        # 1. 构建 Prompt 部分
        prompt_text = f"User: {self.query}\n\nAssistant:\n"
        self.full_text += prompt_text
        
        # 记录 Prompt 部分也许 mask 掉 (通常计算 Loss 只计算 Response，这里根据你的需求，我们mask掉Prompt)
        self.mask_ranges.append((0, len(prompt_text)))

        # 2. 构建 Response 部分
        for step_idx, step_item in enumerate(self.raw_response):
            # 这里的 key 可能是 "step1", "step2" 等
            key = list(step_item.keys())[0]
            content = step_item[key]
            
            # --- Cot ---
            cot_text = f"Step {step_idx+1} Cot: {content.get('cot', '')}\n"
            self.full_text += cot_text
            # Cot 需要计算 Loss，所以不添加 mask_ranges
            
            # --- CoAs ---
            if 'coa' in content:
                for coa in content['coa']:
                    # --- Action ---
                    action_obj = coa.get('action', {})
                    # 格式化 Action 参数
                    args_str = ", ".join([f'{k}="{v}"' for k,v in action_obj.get('args', {}).items()])
                    action_text = f"Step {step_idx+1} Action: {action_obj.get('name', 'unknown')}({args_str})\n"
                    self.full_text += action_text
                    # Action 需要计算 Loss，不 mask

                    # --- Observation ---
                    obs_list = coa.get('observation', [])
                    # 将 Observation 列表转为字符串
                    obs_str_parts = []
                    for obs_item in obs_list:
                        # 简单的 dict 转 str
                        item_str = ", ".join([f"{k}: {v}" for k,v in obs_item.items()])
                        obs_str_parts.append(f"- {item_str}")
                    
                    full_obs_str = "Step {} Observation:\n{}\n".format(step_idx+1, "\n".join(obs_str_parts))
                    
                    # === 关键步骤：记录 Observation 的起止位置 ===
                    start_idx = len(self.full_text)
                    self.full_text += full_obs_str
                    end_idx = len(self.full_text)
                    
                    # 添加到 mask 列表
                    self.mask_ranges.append((start_idx, end_idx))

class MIWVCalculator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.embed_tokenizer = None
        self.embed_model = None
        
    def load_data(self):
        print(">>> Loading Data...")
        samples = []
        for d in self.config['data_dirs']:
            # 递归查找所有 json 文件
            files = glob.glob(os.path.join(d, "**/*.json"), recursive=True)
            for f in files:
                # ================= 修改处 =================
                # 如果文件名是 question_info.json，则跳过
                if os.path.basename(f) == "question_info.json":
                    continue
                # ========================================

                try:
                    with open(f, 'r', encoding='utf-8') as fr:
                        data = json.load(fr)
                        samples.append(DataSample(f, data))
                except Exception as e:
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
        """
        计算单条数据的 Loss
        """
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
            # rel_end = end - prefix_len
            
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
            print("No valid samples found.")
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
                print(f"Error processing sample {i}: {e}")
                torch.cuda.empty_cache()

        results.sort(key=lambda x: x['miwv'], reverse=True)
        
        with open(self.config['output_file'], 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f">>> Done! Results saved to {self.config['output_file']}")

if __name__ == "__main__":
    calculator = MIWVCalculator(CONFIG)
    calculator.run()
