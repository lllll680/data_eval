"""
Vendi-Score 计算原理分析与诊断工具

1. Vendi-Score 计算流程：
   - 输入：embedding 矩阵 X [N, D]，N 是样本数，D 是 embedding 维度
   - 步骤1：对每行做 L2 归一化（normalize=True）
   - 步骤2：计算相似度矩阵 K = X @ X.T [N, N]（余弦相似度）
   - 步骤3：对 K 进行特征值分解，得到特征值 w
   - 步骤4：计算 Shannon 熵：H = -sum(w * log(w))，其中 w > 0
   - 步骤5：Vendi-Score = exp(H)
   - 步骤6：Ratio = Vendi-Score / N

2. Ratio 的含义：
   - Ratio = 1.0：完全多样性，每个样本都不同
   - Ratio < 1.0：存在重复或相似样本
   - Ratio 接近 0：大量重复，多样性极低

3. tools_usage 结构：
   - tools_usage 是一个字符串列表，每个元素代表一条数据的工具使用序列
   - 例如：["tool1(params:['key1']) -> tool2(params:['key2'])", ...]
   - 每个字符串被转换为相同维度的 embedding（通过 Qwen3）
   - 最终得到 [N, D] 的 embedding 矩阵
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

VENDI_REPO_PATH = "/data2/ly/dataset_eval/Vendi-Score"
sys.path.append(VENDI_REPO_PATH)

from vendi_score import vendi  # noqa: E402


def find_json_files(paths: List[str]) -> List[str]:
    """在给定的路径列表中递归收集所有 .json 文件"""
    json_files: List[str] = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".json"):
            json_files.append(os.path.abspath(path))
        elif os.path.isdir(path):
            json_files.extend(
                glob.glob(os.path.join(os.path.abspath(path), "**", "*.json"), recursive=True)
            )
    return sorted(set(json_files))


def load_data(json_files: List[str]) -> List[Dict]:
    """加载多个 json 文件，每个文件代表一条数据"""
    data = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
    return data


def stringify_request(req: Dict, include_values: bool = False) -> str:
    """将 tool_request 转为字符串"""
    if not isinstance(req, dict):
        return str(req)
    keys = sorted(req.keys())
    if include_values:
        # 包含值，但限制长度避免过长
        parts = []
        for k in keys:
            v = str(req[k])[:50]  # 限制值长度
            parts.append(f"{k}:{v}")
        return f"params:{','.join(parts)}"
    return f"params:{keys}"


def extract_tool_usage(dataset: List[Dict], include_values: bool = False) -> List[str]:
    """从 execution_records 中提取工具调用字符串"""
    tools_usage = []
    for item in dataset:
        records = item.get("execution_records", []) or []
        tool_strs = []
        for rec in records:
            tool_name = rec.get("tool_name", "") or "unknown_tool"
            request = stringify_request(rec.get("tool_request", {}), include_values=include_values)
            tool_strs.append(f"{tool_name}({request})")
        tools_usage.append(" -> ".join(tool_strs))
    return tools_usage


class QwenEmbedder:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading Qwen model from: {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.device = device
        self.model.eval()

    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """分批次计算 Embeddings"""
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs["attention_mask"]
                last_hidden = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                all_embeddings.append(batch_embeddings.cpu().numpy())

        if not all_embeddings:
            return np.array([])
        return np.concatenate(all_embeddings, axis=0)


def analyze_diversity(embeddings: np.ndarray, tools_usage: List[str]):
    """详细分析多样性，诊断问题"""
    N = len(tools_usage)
    D = embeddings.shape[1] if embeddings.size > 0 else 0

    print("\n" + "=" * 70)
    print("多样性诊断分析")
    print("=" * 70)

    # 1. 基础统计
    print(f"\n[1] 基础统计:")
    print(f"    样本数 N = {N}")
    print(f"    Embedding 维度 D = {D}")

    if embeddings.size == 0:
        print("    错误：Embeddings 为空")
        return

    # 2. 计算相似度矩阵
    # 先归一化
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    similarity_matrix = embeddings_norm @ embeddings_norm.T

    # 3. 相似度分布统计
    # 取上三角（不包括对角线）
    triu_indices = np.triu_indices(N, k=1)
    similarities = similarity_matrix[triu_indices]

    print(f"\n[2] 相似度矩阵统计:")
    print(f"    平均相似度: {similarities.mean():.4f}")
    print(f"    中位数相似度: {np.median(similarities):.4f}")
    print(f"    最大相似度: {similarities.max():.4f}")
    print(f"    最小相似度: {similarities.min():.4f}")
    print(f"    标准差: {similarities.std():.4f}")

    # 统计高相似度对的数量
    high_sim_threshold = 0.95
    high_sim_count = (similarities >= high_sim_threshold).sum()
    print(f"    相似度 >= {high_sim_threshold} 的样本对数: {high_sim_count} / {len(similarities)} ({100*high_sim_count/len(similarities):.2f}%)")

    # 4. 特征值分析
    K = similarity_matrix
    eigenvals = np.linalg.eigvalsh(K)
    eigenvals = eigenvals[eigenvals > 1e-10]  # 过滤数值误差
    eigenvals = eigenvals / eigenvals.sum()  # 归一化

    print(f"\n[3] 特征值分布分析:")
    print(f"    非零特征值数量: {len(eigenvals)}")
    print(f"    最大特征值: {eigenvals.max():.4f}")
    print(f"    前5大特征值: {eigenvals[-5:][::-1]}")
    print(f"    特征值熵 (Shannon): {-np.sum(eigenvals * np.log(eigenvals + 1e-10)):.4f}")

    # 5. Vendi-Score
    score = vendi.score_X(embeddings, q=1, normalize=True)
    ratio = score / N

    print(f"\n[4] Vendi-Score 结果:")
    print(f"    Vendi-Score = {score:.4f}")
    print(f"    Ratio = {ratio:.4f}")

    # 6. 文本重复分析
    print(f"\n[5] 文本重复分析:")
    text_counter = Counter(tools_usage)
    unique_count = len(text_counter)
    duplicate_count = N - unique_count
    print(f"    唯一文本数: {unique_count} / {N}")
    print(f"    完全重复的样本数: {duplicate_count}")

    if duplicate_count > 0:
        print(f"    前5个最常见的文本:")
        for text, count in text_counter.most_common(5):
            print(f"      [{count}次] {text[:80]}...")

    # 7. 工具名称分析
    print(f"\n[6] 工具名称分布:")
    tool_names_all = []
    for item in tools_usage:
        # 简单提取 tool_name（在第一个括号之前）
        parts = item.split("(")
        if parts:
            tool_name = parts[0].strip()
            tool_names_all.append(tool_name)

    tool_name_counter = Counter(tool_names_all)
    print(f"    唯一工具名数量: {len(tool_name_counter)}")
    print(f"    前10个最常见的工具名:")
    for name, count in tool_name_counter.most_common(10):
        print(f"      [{count}次] {name[:60]}")

    # 8. 诊断建议
    print(f"\n[7] 诊断建议:")
    if ratio < 0.1:
        print("    ⚠️  Ratio 极低 (< 0.1)，可能存在以下问题:")
        print("       1. 大量样本使用相同的工具序列")
        print("       2. tool_name 虽然不同，但语义非常相似")
        print("       3. 只使用 params keys，丢失了参数值的差异")
        print("       4. Embedding 模型对技术术语区分度不够")
    elif ratio < 0.5:
        print("    ⚠️  Ratio 较低 (< 0.5)，建议:")
        print("       1. 考虑包含 tool_request 的实际值（不仅仅是 keys）")
        print("       2. 检查是否有大量相似的工具调用模式")
        print("       3. 考虑使用更大的 embedding 模型")
    else:
        print("    ✓ Ratio 较高，多样性较好")

    if high_sim_count > len(similarities) * 0.3:
        print(f"    ⚠️  超过30%的样本对相似度极高，可能存在模式重复")

    if duplicate_count > N * 0.1:
        print(f"    ⚠️  超过10%的样本完全重复，建议检查数据质量")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="分析 Vendi-Score 计算，诊断多样性问题")
    parser.add_argument(
        "--data-roots",
        nargs="+",
        required=True,
        help="一个或多个路径，包含 .json 数据文件",
    )
    parser.add_argument(
        "--model-path",
        default="/data2/Qwen/Qwen3-Embedding-0.6B",
        help="Qwen3 embedding 模型路径",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="嵌入计算的 batch size",
    )
    parser.add_argument(
        "--include-values",
        action="store_true",
        help="是否在 tool_request 中包含实际值（不仅仅是 keys）",
    )
    args = parser.parse_args()

    # 1. 加载数据
    json_files = find_json_files(args.data_roots)
    print(f"Discovered {len(json_files)} json files.")
    if not json_files:
        print("Error: No json files found.")
        return

    dataset = load_data(json_files)
    print(f"Loaded {len(dataset)} data items.")
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    # 2. 提取工具使用文本
    print("Extracting tool usage strings...")
    tools_usage = extract_tool_usage(dataset, include_values=args.include_values)
    
    # 显示样例
    if tools_usage:
        print(f"\n[Sample Tool Usage (first 3)]:")
        for i, text in enumerate(tools_usage[:3]):
            print(f"  [{i+1}] {text[:120]}...")

    # 3. 加载模型并计算 embeddings
    try:
        embedder = QwenEmbedder(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nComputing embeddings...")
    embeddings = embedder.get_embeddings(tools_usage, batch_size=args.batch_size)
    if embeddings.size == 0:
        print("Error: Embeddings are empty.")
        return

    # 4. 详细分析
    analyze_diversity(embeddings, tools_usage)


if __name__ == "__main__":
    main()

