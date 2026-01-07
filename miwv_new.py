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
        self.query = data['query']
        self.raw_response = data['response']
        self.full_text = "" 
        self.mask_ranges = [] 
        self._serialize()

    def _serialize(self):
        """
        将 JSON 结构序列化为文本，并记录需要 Mask 的位置。
        """
        # 1. 构建 Prompt 部分
        prompt_text = f"User: {self.query}\n\nAssistant:\n"
        self.full_text += prompt_text
        # Mask 掉 Prompt 区域
        self.mask_ranges.append((0, len(prompt_text)))

        # 2. 构建 Response 部分
        for step_idx, step_item in enumerate(self.raw_response):
            # 获取当前步的 key (如 "step1") 和内容
            key = list(step_item.keys())[0]
            content = step_item[key]
            
            # --- Cot (计算 Loss) ---
            cot_text = f"Step {step_idx+1} Cot: {content.get('cot', '')}\n"
            self.full_text += cot_text
            
            # --- CoAs ---
            if 'coa' in content:
                for coa in content['coa']:
                    # --- Action (计算 Loss) ---
                    action_obj = coa.get('action', {})
                    args_dict = action_obj.get('args', {})
                    args_str = ", ".join([f'{k}="{v}"' for k, v in args_dict.items()])
                    action_text = f"Step {step_idx+1} Action: {action_obj.get('name', 'unknown')}({args_str})\n"
                    self.full_text += action_text

                    # --- Observation (修复此处：适配字典或列表格式，且需要 Mask) ---
                    obs_data = coa.get('observation', {})
                    obs_str_parts = []
                    
                    # 关键修复：判断 observation 的数据类型
                    if isinstance(obs_data, dict):
                        # 如果是字典: {"接口": "xxx", "IP": "xxx"}
                        item_str = ", ".join([f"{k}: {v}" for k, v in obs_data.items()])
                        obs_str_parts.append(f"- {item_str}")
                    elif isinstance(obs_data, list):
                        # 如果是列表: [{"k1": "v1"}, {"k2": "v2"}]
                        for obs_item in obs_data:
                            if isinstance(obs_item, dict):
                                item_str = ", ".join([f"{k}: {v}" for k, v in obs_item.items()])
                                obs_str_parts.append(f"- {item_str}")
                            else:
                                obs_str_parts.append(f"- {obs_item}")
                    
                    full_obs_str = "Step {} Observation:\n{}\n".format(step_idx+1, "\n".join(obs_str_parts))
                    
                    # 记录 Observation 的起止位置以便 Mask
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
        skip_filenames = {"question_info.json", "batch_summary.json"}
        seen_file_paths = set()

        for d in self.config['data_dirs']:
            if not os.path.exists(d):
                print(f"Warning: Directory not found: {d}")
                continue
            files = glob.glob(os.path.join(d, "**/*.json"), recursive=True)
            for f in files:
                basename = os.path.basename(f)
                if basename in skip_filenames: continue
                
                abs_f = os.path.abspath(f)
                if abs_f in seen_file_paths: continue
                seen_file_paths.add(abs_f)

                try:
                    with open(f, 'r', encoding='utf-8') as fr:
                        data = json.load(fr)
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
                    mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_emb = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
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
            # 1. Mask 掉前缀(One-shot context)
            if start < prefix_len:
                labels[0, idx] = -100
                continue
            
            # 2. Mask 掉 Response 内部被标记为 Observation 或 Prompt 的部分
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
        if not samples: return

        # 1. 计算相似度并选择邻居
        embeddings = self.get_embeddings(samples)
        print(">>> Computing Similarity Matrix...")
        sim_matrix = cosine_similarity(embeddings)
        
        # 排除自身，且排除 Query 完全相同的样本（防止泄露）
        for i in range(len(samples)):
            for j in range(len(samples)):
                if i == j or samples[i].query == samples[j].query:
                    sim_matrix[i][j] = -1
        most_sim_indices = np.argmax(sim_matrix, axis=1)

        # 2. 加载推理模型
        print(f">>> Loading LLM: {self.config['llm_model_path']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['llm_model_path'], trust_remote_code=True)
        # 确保 pad_token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['llm_model_path'],
            device_map=self.config['device_map'],
            torch_dtype=self.config['dtype'],
            trust_remote_code=True
        ).eval()

        # 3. 计算 MIWV
        results = []
        print(">>> Calculating MIWV Scores...")
        for i, sample in enumerate(tqdm(samples)):
            try:
                # Loss Zero: 直接计算
                loss_zero = self.calculate_token_loss(sample.full_text, sample.mask_ranges, prefix_context="")
                
                # Loss One: 以最相似样本作为上下文
                neighbor = samples[most_sim_indices[i]]
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

        # 4. 排序并保存
        results.sort(key=lambda x: x['miwv'], reverse=True)
        with open(self.config['output_file'], 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f">>> Done! Results saved to {self.config['output_file']}")

if __name__ == "__main__":
    calculator = MIWVCalculator(CONFIG)
    calculator.run()
