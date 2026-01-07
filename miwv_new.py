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
        # 在这里检查是否还有重复路径
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
        self.query = data['query']
        self.raw_response = data['response']
        self.full_text = "" 
        self.mask_ranges = [] 
        self._serialize()

    def _serialize(self):
        prompt_text = f"User: {self.query}\n\nAssistant:\n"
        self.full_text += prompt_text
        self.mask_ranges.append((0, len(prompt_text)))

        for step_idx, step_item in enumerate(self.raw_response):
            key = list(step_item.keys())[0]
            content = step_item[key]
            
            # --- Cot ---
            cot_text = f"Step {step_idx+1} Cot: {content.get('cot', '')}\n"
            self.full_text += cot_text
            
            # --- CoAs ---
            if 'coa' in content:
                for coa in content['coa']:
                    # --- Action ---
                    action_obj = coa.get('action', {})
                    args_str = ", ".join([f'{k}="{v}"' for k,v in action_obj.get('args', {}).items()])
                    action_text = f"Step {step_idx+1} Action: {action_obj.get('name', 'unknown')}({args_str})\n"
                    self.full_text += action_text

                    # --- Observation ---
                    obs_list = coa.get('observation', [])
                    obs_str_parts = []
                    for obs_item in obs_list:
                        item_str = ", ".join([f"{k}: {v}" for k,v in obs_item.items()])
                        obs_str_parts.append(f"- {item_str}")
                    
                    full_obs_str = "Step {} Observation:\n{}\n".format(step_idx+1, "\n".join(obs_str_parts))
                    
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
        # 定义需要跳过的文件名
        skip_filenames = {"question_info.json", "batch_summary.json"}
        # 用于物理路径去重，防止 CONFIG 中重复配置了文件夹
        seen_file_paths = set()

        for d in self.config['data_dirs']:
            if not os.path.exists(d):
                print(f"Warning: Directory not found: {d}")
                continue
                
            files = glob.glob(os.path.join(d, "**/*.json"), recursive=True)
            for f in files:
                # 1. 获取文件名进行过滤
                basename = os.path.basename(f)
                if basename in skip_filenames:
                    continue
                
                # 2. 物理路径去重 (处理不小心写错文件夹路径的情况)
                abs_f = os.path.abspath(f)
                if abs_f in seen_file_paths:
                    continue
                seen_file_paths.add(abs_f)

                try:
                    with open(f, 'r', encoding='utf-8') as fr:
                        data = json.load(fr)
                        # 简单的格式校验
                        if 'query' in data and 'response' in data:
                            samples.append(DataSample(f, data))
                except Exception as e:
                    print(f"Error loading {f}: {e}")
                    
        print(f">>> Loaded {len(samples)} unique samples.")
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
        # 1. 加载数据
        samples = self.load_data()
        if not samples:
            print("No valid samples found.")
            return

        # 2. 计算相似度并寻找邻居
        embeddings = self.get_embeddings(samples)
        print(">>> Computing Similarity Matrix...")
        sim_matrix = cosine_similarity(embeddings)
        
        # 额外保险：如果 Query 完全一致，则不互为邻居（防止同一 Query 的不同 Response 互相泄露）
        for i in range(len(samples)):
            for j in range(len(samples)):
                if i == j or samples[i].query == samples[j].query:
                    sim_matrix[i][j] = -1
                    
        most_sim_indices = np.argmax(sim_matrix, axis=1)

        # 3. 加载推理模型
        print(f">>> Loading LLM: {self.config['llm_model_path']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['llm_model_path'], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['llm_model_path'],
            device_map=self.config['device_map'],
            torch_dtype=self.config['dtype'],
            trust_remote_code=True
        ).eval()

        # 4. 计算 MIWV
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

        # 5. 保存结果
        results.sort(key=lambda x: x['miwv'], reverse=True)
        with open(self.config['output_file'], 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f">>> Done! Results saved to {self.config['output_file']}")

if __name__ == "__main__":
    calculator = MIWVCalculator(CONFIG)
    calculator.run()
