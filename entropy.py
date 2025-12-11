import os
import json
import math
import glob
from collections import Counter
from tqdm import tqdm  # 如果没有安装，可以使用 pip install tqdm，或者删除相关代码

def calculate_shannon_entropy(counter):
    """
    计算香农熵 (Shannon Entropy)
    H(X) = - sum(p(x) * log2(p(x)))
    """
    total_count = sum(counter.values())
    if total_count == 0:
        return 0.0
    
    entropy = 0.0
    for count in counter.values():
        p_x = count / total_count
        entropy -= p_x * math.log2(p_x)
    
    return entropy

def analyze_tool_usage(root_path):
    # 1. 初始化计数器
    individual_tool_counter = Counter() # 记录单个工具出现的次数 (例如: "Search": 100)
    tool_chain_counter = Counter()      # 记录工具组合路径的次数 (例如: "Search->Calculator": 50)
    
    # 获取所有json文件路径 (递归查找)
    # 假设路径结构是 /data2/ly/dataset_eval/code_apply/ 下面的任意子文件夹中
    search_pattern = os.path.join(root_path, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)
    
    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")

    valid_files_count = 0

    # 2. 遍历文件并提取数据
    for file_path in tqdm(json_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取 execution_records
            records = data.get("execution_records", [])
            
            # 如果没有执行记录，跳过
            if not records:
                continue
                
            current_chain = []
            
            for step in records:
                # 获取 tool_name，如果为空则标记为 "unknown"
                t_name = step.get("tool_name", "").strip()
                if not t_name:
                    continue
                
                # 统计单个工具
                individual_tool_counter[t_name] += 1
                current_chain.append(t_name)
            
            # 统计工具链 (将列表转换为字符串或元组作为Key)
            if current_chain:
                # 使用 " -> " 连接，形成如 "Search -> Calculator" 的字符串
                chain_signature = " -> ".join(current_chain)
                tool_chain_counter[chain_signature] += 1
                valid_files_count += 1
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    # 3. 计算指标
    
    # --- 指标 A: 唯一工具数量 ---
    unique_tools = list(individual_tool_counter.keys())
    num_unique_tools = len(unique_tools)
    
    # --- 指标 B: 单个工具分布熵 (Tool Distribution Entropy) ---
    tool_distribution_entropy = calculate_shannon_entropy(individual_tool_counter)
    
    # --- 指标 C: 工具链组合熵 (Tool Chain Entropy) --- 
    # 这个指标对应你想要的 "组合的频率分布（熵）"
    chain_entropy = calculate_shannon_entropy(tool_chain_counter)

    # 4. 输出结果
    print("\n" + "="*50)
    print("📊 数据集工具多样性分析报告 (Statistics Report)")
    print("="*50)
    print(f"有效数据条数 (Valid Trajectories): {valid_files_count}")
    print(f"唯一工具名称数量 (Unique Tool Names): {num_unique_tools}")
    print("-" * 30)
    print(f"1. 单个工具分布熵 (Individual Tool Entropy): {tool_distribution_entropy:.4f}")
    print(f"2. 工具组合路径熵 (Tool Chain Entropy):      {chain_entropy:.4f}  <-- 核心指标")
    print("-" * 30)
    
    print("\nTop 5 最常用的工具 (Most Frequent Tools):")
    for tool, count in individual_tool_counter.most_common(5):
        print(f"  - {tool}: {count}")

    print("\nTop 5 最常用的工具组合 (Most Frequent Chains):")
    for chain, count in tool_chain_counter.most_common(5):
        print(f"  - [{chain}]: {count}")
    print("="*50)

if __name__ == "__main__":
    # 配置你的数据路径
    DATA_PATH = "/data2/ly/dataset_eval/code_apply/"
    
    if os.path.exists(DATA_PATH):
        analyze_tool_usage(DATA_PATH)
    else:
        print(f"路径不存在: {DATA_PATH}")
