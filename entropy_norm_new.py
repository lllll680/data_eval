import os
import json
import math
import glob
from collections import Counter
from tqdm import tqdm

def calculate_entropy_metrics(counter):
    """
    计算香农熵及其归一化值
    """
    total_count = sum(counter.values())
    unique_count = len(counter)
    
    # 1. 处理空数据或只有1种情况的边界条件
    if total_count == 0 or unique_count <= 1:
        return 0.0, 0.0, unique_count
    
    # 2. 计算原始香农熵 H(X)
    raw_entropy = 0.0
    for count in counter.values():
        p_x = count / total_count
        raw_entropy -= p_x * math.log2(p_x)
    
    # 3. 计算最大可能的熵 H_max = log2(N)
    max_entropy = math.log2(unique_count)
    
    # 4. 计算归一化熵 (Efficiency) = H(X) / H_max
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    
    return raw_entropy, normalized_entropy, unique_count

def get_diversity_level(norm_entropy):
    """根据归一化熵给出简单的阈值评价"""
    if norm_entropy < 0.3:
        return "低 (Low) - 模式非常固定/单一"
    elif norm_entropy < 0.7:
        return "中 (Moderate) - 存在主要模式，兼顾多样性"
    else:
        return "高 (High) - 分布非常均匀/发散 (或过于杂乱)"

def analyze_tool_usage(root_paths):
    individual_tool_counter = Counter() 
    tool_chain_counter = Counter()      
    
    all_json_files = []

    # --- 1. 聚合多个文件夹下的文件 ---
    print(f"正在扫描 {len(root_paths)} 个数据目录...")
    for path in root_paths:
        if not os.path.exists(path):
            print(f"⚠️ 警告: 路径不存在，已跳过 -> {path}")
            continue
            
        search_pattern = os.path.join(path, "**", "*.json")
        # 递归查找该目录下的所有json
        files = glob.glob(search_pattern, recursive=True)
        print(f"  - 目录 {path}: 发现 {len(files)} 个文件")
        all_json_files.extend(files)

    print(f"总计找到 {len(all_json_files)} 个JSON文件，开始处理...")
    valid_files_count = 0

    # --- 2. 遍历处理 ---
    for file_path in tqdm(all_json_files, desc="Processing files"):
        
        # === 跳过特定文件 ===
        file_name = os.path.basename(file_path)
        if file_name == "batch_summary.json" or file_name == "question_info.json":
            continue
        # ===============================================

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 新格式：从 response 数组中提取工具使用信息
            response_steps = data.get("response", [])
            if not response_steps:
                continue
                
            current_chain = []
            
            # 遍历 response 数组中的每个步骤
            for step_item in response_steps:
                # step_item 是一个字典，key 是 "step1", "step2" 等
                # value 是包含 "cot" 和 "coa" 的字典
                for step_key, step_content in step_item.items():
                    # 获取 coa 数组
                    coa_list = step_content.get("coa", [])
                    if not coa_list:
                        continue
                    
                    # 遍历该步骤中的所有 action-observation 对
                    for coa_item in coa_list:
                        action = coa_item.get("action", {})
                        if not action:
                            continue
                        
                        tool_name = action.get("name", "").strip()
                        if not tool_name:
                            continue
                        
                        # 统计单个工具使用
                        individual_tool_counter[tool_name] += 1
                        # 添加到工具链
                        current_chain.append(tool_name)
            
            if current_chain:
                chain_signature = " -> ".join(current_chain)
                tool_chain_counter[chain_signature] += 1
                valid_files_count += 1
                
        except Exception as e:
            # print(f"Error reading {file_path}: {e}") 
            continue

    # --- 3. 计算指标 ---
    
    # 指标 A: 单个工具分布
    t_raw, t_norm, t_unique = calculate_entropy_metrics(individual_tool_counter)
    
    # 指标 B: 工具链组合分布 (核心)
    c_raw, c_norm, c_unique = calculate_entropy_metrics(tool_chain_counter)

    # --- 4. 输出报告 ---
    print("\n" + "="*60)
    print("📊 多数据集工具多样性分析报告 (Normalized Analysis)")
    print("="*60)
    print(f"包含目录数: {len(root_paths)}")
    print(f"有效轨迹数 (Valid Trajectories): {valid_files_count}")
    print("-" * 60)
    
    print(f"【指标 A】 单个工具使用分布 (Tool Usage Distribution)")
    print(f"  - 唯一工具数 (N):       {t_unique}")
    print(f"  - 原始熵 (Raw Entropy): {t_raw:.4f}")
    print(f"  - 归一化熵 (0~1):       {t_norm:.4f}  [{get_diversity_level(t_norm)}]")
    
    print("-" * 60)
    
    print(f"【指标 B】 工具链路径组合 (Tool Chain Diversity) <--- 核心关注")
    print(f"  - 唯一路径组合数 (N):   {c_unique}")
    print(f"  - 原始熵 (Raw Entropy): {c_raw:.4f}")
    print(f"  - 归一化熵 (0~1):       {c_norm:.4f}  [{get_diversity_level(c_norm)}]")
    
    print("-" * 60)
    print("💡 阈值解读参考:")
    print("   [0.0 - 0.3]: 集中度高。模型总是倾向于使用某1-2种特定的解决路径。")
    print("   [0.3 - 0.7]: 平衡状态。既有主流的解决套路，也有处理长尾问题的变体。")
    print("   [0.7 - 1.0]: 离散度高。几乎没有固定的套路，每条数据的解决路径都不同。")
    print("="*60)
    
    print("\nTop 5 最常用的工具组合 (及其占比):")
    total_chains = sum(tool_chain_counter.values())
    if total_chains > 0:
        for chain, count in tool_chain_counter.most_common(5):
            ratio = (count / total_chains) * 100
            print(f"  {ratio:5.1f}% | [{chain}]")
    else:
        print("  (无数据)")
    print("="*60)

if __name__ == "__main__":
    # 配置你的文件夹列表
    DATA_PATHS = [
        "/raid/data/ly/data/dataset/data",
        # "/raid/data/ly/data2",
        # "/raid/data/ly/data3",
        # 你可以继续添加更多路径...
    ]
    
    analyze_tool_usage(DATA_PATHS)
