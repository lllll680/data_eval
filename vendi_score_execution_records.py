import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ------------------------------------------------------------------------
# 1. 路径配置：将 Vendi-Score 库加入系统路径，以便导入
# ------------------------------------------------------------------------
VENDI_REPO_PATH = "/data2/ly/dataset_eval/Vendi-Score"
sys.path.append(VENDI_REPO_PATH)

from vendi_score import vendi  # noqa: E402


# ------------------------------------------------------------------------
# 2. 数据加载
# ------------------------------------------------------------------------
def find_json_files(paths: List[str]) -> List[str]:
    """在给定的路径列表中递归收集所有 .json 文件"""
    json_files: List[str] = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".json"):
            json_files.append(os.path.abspath(path))
        elif os.path.isdir(path):
            # 使用 glob 递归收集
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
        except Exception as e:  # noqa: BLE001 - 简单脚本直接打印错误
            print(f"[WARN] Failed to load {fp}: {e}")
    return data


# ------------------------------------------------------------------------
# 3. 特征提取（工具多样性）
# ------------------------------------------------------------------------
def stringify_request(req: Dict) -> str:
    """
    将 tool_request 转为简短字符串，主要保留键，避免数值干扰多样性。
    如果需要更细粒度，可以在此展开值。
    """
    if not isinstance(req, dict):
        return str(req)
    keys = sorted(req.keys())
    return f"params:{keys}"


def extract_tool_usage(dataset: List[Dict]) -> List[str]:
    """
    从 execution_records 中提取工具调用字符串："tool_name(params:[...])"
    一个数据条目 -> 组合所有步骤，用 " -> " 连接
    """
    tools_usage = []
    for item in dataset:
        records = item.get("execution_records", []) or []
        tool_strs = []
        for rec in records:
            tool_name = rec.get("tool_name", "") or "unknown_tool"
            request = stringify_request(rec.get("tool_request", {}))
            tool_strs.append(f"{tool_name}({request})")
        tools_usage.append(" -> ".join(tool_strs))
    return tools_usage


# ------------------------------------------------------------------------
# 4. Embedding 模型封装 (Qwen3)
# ------------------------------------------------------------------------
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
                max_length=8192,  # Qwen3 支持长文，根据显存调整
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs["attention_mask"]
                last_hidden = outputs.last_hidden_state
                # Masked Mean Pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                all_embeddings.append(batch_embeddings.cpu().numpy())

        if not all_embeddings:
            return np.array([])
        return np.concatenate(all_embeddings, axis=0)


# ------------------------------------------------------------------------
# 5. 主程序
# ------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Vendi-Score for tool diversity using execution_records."
    )
    parser.add_argument(
        "--data-roots",
        nargs="+",
        required=True,
        help="一个或多个路径，包含 .json 数据文件；支持文件或文件夹，递归读取。",
    )
    parser.add_argument(
        "--model-path",
        default="/data2/Qwen/Qwen3-Embedding-0.6B",
        help="Qwen3 embedding 模型路径。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="嵌入计算的 batch size。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 收集并加载数据
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

    # 2. 提取工具多样性文本
    print("Extracting tool usage strings from execution_records...")
    tools_usage = extract_tool_usage(dataset)
    print(f"[Sample Tool Usage]: {tools_usage[0][:120]}..." if tools_usage else "No tool usage found.")

    # 3. 加载模型
    try:
        embedder = QwenEmbedder(args.model_path)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading model: {e}")
        return

    # 4. 计算 Vendi Score
    embeddings = embedder.get_embeddings(tools_usage, batch_size=args.batch_size)
    if embeddings.size == 0:
        print("Error: Embeddings are empty.")
        return

    score = vendi.score_X(embeddings, q=1, normalize=True)
    ratio = score / len(dataset)

    print("\n" + "=" * 50)
    print(f"{'Metric Name':<25} | {'Vendi Score':<10} | {'Ratio (VS/N)':<10}")
    print("-" * 54)
    print(f"{'工具使用 (Tool Usage)':<25} | {score:.4f}     | {ratio:.4f}")
    print("=" * 50)
    print("说明: Ratio 越接近 1.0，说明工具调用的多样性越高；接近 0 说明重复度高。")


if __name__ == "__main__":
    main()

