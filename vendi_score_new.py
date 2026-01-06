import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Callable
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import Levenshtein  # 需安装: pip install python-Levenshtein

# ------------------------------------------------------------------------
# 1. 路径配置
# ------------------------------------------------------------------------
VENDI_REPO_PATH = "/data2/ly/dataset_eval/Vendi-Score"
sys.path.append(VENDI_REPO_PATH)
from vendi_score import vendi

# ------------------------------------------------------------------------
# 2. 特征提取函数
# ------------------------------------------------------------------------

def extract_reasoning_text(dataset: List[Dict]) -> List[str]:
    """
    提取 reasoning：将所有步骤的 reasoning 拼接成一段长文本。
    反映“思维链”的语义多样性。
    """
    texts = []
    for item in dataset:
        records = item.get("execution_records", []) or []
        # 过滤掉空的 reasoning，用换行符或特殊标记连接
        steps_reasoning = [
            rec.get("reasoning", "").strip() 
            for rec in records 
            if rec.get("reasoning")
        ]
        # 如果没有 reasoning，给一个占位符，避免空字符串
        full_text = " -> ".join(steps_reasoning) if steps_reasoning else "empty_reasoning"
        texts.append(full_text)
    return texts

def extract_tool_sequence(dataset: List[Dict]) -> List[str]:
    """
    提取 tool_name 序列：仅提取工具名，拼接成字符串。
    反映“执行路径”的结构多样性。
    """
    sequences = []
    for item in dataset:
        records = item.get("execution_records", []) or []
        # 仅提取 tool_name
        tools = [
            rec.get("tool_name", "").strip() 
            for rec in records 
            if rec.get("tool_name")
        ]
        # 用空格连接，形成类似 "login check_cpu restart" 的序列字符串
        seq_str = " ".join(tools)
        sequences.append(seq_str)
    return sequences

# ------------------------------------------------------------------------
# 3. 相似度计算核心
# ------------------------------------------------------------------------

# A. Embedding 方式 (用于 Reasoning)
class QwenEmbedder:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading Qwen model from: {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.device = device
        self.model.eval()

    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        # 添加进度提示
        total = len(texts)
        print(f"Embedding encoding progress: 0/{total}", end="\r")
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096, # reasoning 可能较长
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                all_embeddings.append(batch_embeddings.cpu().numpy())
            
            print(f"Embedding encoding progress: {min(i + batch_size, total)}/{total}", end="\r")
        
        print("")
        if not all_embeddings: return np.array([])
        return np.concatenate(all_embeddings, axis=0)

# B. 编辑距离方式 (用于 Tool Sequence)
def levenshtein_similarity(seqs: List[str]):
    """
    计算基于 Levenshtein Ratio 的相似度矩阵 K。
    Ratio = (len(s1) + len(s2) - dist) / (len(s1) + len(s2))
    完全相同为 1.0，完全不同为 0.0。
    """
    n = len(seqs)
    print(f"Computing Levenshtein Kernel for {n} samples (O(N^2))...")
    
    # 注意：如果数据量 > 5000，这里会比较慢。
    # 为了演示清晰，使用双重循环。生产环境可用 joblib 并行。
    K = np.zeros((n, n))
    
    # 预先处理为 list 以加速索引
    seq_list = list(seqs)
    
    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            # Levenshtein.ratio 返回 0-1 之间的相似度
            sim = Levenshtein.ratio(seq_list[i], seq_list[j])
            K[i, j] = sim
            K[j, i] = sim
            
        if i % 100 == 0:
            print(f"Kernel progress: {i}/{n}", end="\r")
    
    print(f"Kernel computation done.        ")
    return K

# ------------------------------------------------------------------------
# 4. 主程序
# ------------------------------------------------------------------------

def load_data(paths: List[str]) -> List[Dict]:
    # ... (使用你之前提供的 load_data 逻辑，或者简单的 glob) ...
    json_files = []
    for path in paths:
        if os.path.isfile(path): json_files.append(path)
        else: json_files.extend(glob.glob(os.path.join(path, "**/*.json"), recursive=True))
    
    data = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list): data.extend(content)
                else: data.append(content)
        except: pass
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--model-path", default="/data2/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # 1. 加载数据
    dataset = load_data(args.data_roots)
    N = len(dataset)
    print(f"Total samples: {N}")
    if N == 0: return

    print("\n" + "="*60)
    print("PART 1: 思维链多样性 (Reasoning Diversity)")
    print("Method: Qwen3 Embeddings + Cosine Similarity")
    print("-" * 60)
    
    # 提取 reasoning
    reasoning_texts = extract_reasoning_text(dataset)
    # 加载模型并 Embedding
    embedder = QwenEmbedder(args.model_path)
    embeddings = embedder.get_embeddings(reasoning_texts, args.batch_size)
    
    # 计算 Vendi Score (Reasoning)
    vs_reasoning = vendi.score_X(embeddings, q=1, normalize=True)
    print(f"\n>> Reasoning Vendi Score: {vs_reasoning:.4f}")
    print(f">> Reasoning Ratio (VS/N): {vs_reasoning/N:.4f}")


    print("\n" + "="*60)
    print("PART 2: 工具路径多样性 (Tool Sequence Diversity)")
    print("Method: Tool Names + Levenshtein Ratio (Edit Distance)")
    print("-" * 60)

    if N > 10000:
        print("Warning: Sample size > 10000, Levenshtein calculation might be slow.")

    # 提取 tool sequences
    tool_seqs = extract_tool_sequence(dataset)
    
    # 计算自定义 Kernel
    K_tools = levenshtein_similarity(tool_seqs)
    
    # 计算 Vendi Score (Tools) - 使用 score_K 直接传入相似度矩阵
    vs_tools = vendi.score_K(K_tools, q=1)
    print(f"\n>> Tool Sequence Vendi Score: {vs_tools:.4f}")
    print(f">> Tool Sequence Ratio (VS/N): {vs_tools/N:.4f}")

    print("="*60)

if __name__ == "__main__":
    main()
