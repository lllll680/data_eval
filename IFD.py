import os
import json
import torch
import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple

# ================= 配置区域 =================
MODEL_PATH = "/raid/data/models/Qwen2.5-7B-Instruct"  # 你的模型路径
DATA_DIRS = [
"/raid/data/ly/data/dataset/data", 
# "/raid/data/ly/data/dataset/data2", 
# "/raid/data/ly/data/dataset/data3"
] # 数据文件夹列表
OUTPUT_FILE = "/raid/data/ly/data/dataset/ifd_results.jsonl" # 结果保存路径
BATCH_SIZE = 1 # 72B模型很大，且不同样本长度差异大，建议设为1避免OOM
SKIP_FILES = {"question_info.json", "batch_summary.json"}

# 显存优化配置
# 如果是80G A100 * 4，可以直接用 bf16。如果是40G，建议开启 load_in_4bit=True
USE_4BIT = False 
USE_BF16 = True 

# ===========================================

def load_model_and_tokenizer(model_path):
    print(f"正在加载模型: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 自动分配到4张卡上
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", 
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        load_in_4bit=USE_4BIT,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def find_json_files(directories: List[str]) -> List[str]:
    json_files = []
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".json") and file not in SKIP_FILES:
                    json_files.append(os.path.join(root, file))
    return json_files

def format_response_structure(response_list: List[Dict]) -> List[Tuple[str, str]]:
    """
    将复杂的Response JSON结构解析为 (文本片段, 类型) 的列表。
    类型分为: 'text' (需要计算loss) 和 'ignore' (Observation, 不计算loss)
    """
    segments = []
    
    for step_idx, step_item in enumerate(response_list):
        # 获取 step key (e.g., "step1")
        step_key = list(step_item.keys())[0]
        step_content = step_item[step_key]
        
        # 1. 格式化 COT
        cot_text = f"Step {step_idx+1}:\nThought: {step_content.get('cot', '')}\n"
        segments.append((cot_text, "text"))
        
        # 2. 格式化 COA (Action 和 Observation)
        if "coa" in step_content and isinstance(step_content["coa"], list):
            for coa_item in step_content["coa"]:
                action = coa_item.get("action", {})
                observation = coa_item.get("observation", {})
                
                # Action 部分 (需要计算 Loss)
                # 将 Action 格式化为函数调用风格
                args_str = ", ".join([f"{k}='{v}'" for k, v in action.get("args", {}).items()])
                action_text = f"Action: {action.get('name', 'unknown')}({args_str})\n"
                segments.append((action_text, "text"))
                
                # Observation 部分 (!!! 忽略 Loss !!!)
                # 将 Observation 转为字符串
                obs_str = json.dumps(observation, ensure_ascii=False)
                obs_text = f"Observation: {obs_str}\n"
                segments.append((obs_text, "ignore"))
                
    return segments

def tokenize_and_mask(tokenizer, query: str, response_segments: List[Tuple[str, str]], device, is_conditioned=True):
    """
    修复版：使用 encode 获取特殊字符 ID，防止 NoneType 报错
    """
    
    input_ids_list = []
    labels_list = []
    
    # === 修改开始：更安全的获取特殊 Token ID ===
    # 使用 encode 而不是 convert_tokens_to_ids，因为 \n 直接查字典可能会返回 None
    def get_token_id(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return tokens[0] if tokens else 0  # 如果获取失败返回0或其他默认值，但在Qwen中通常不会失败

    im_start_id = get_token_id("<|im_start|>")
    im_end_id = get_token_id("<|im_end|>")
    nl_id = get_token_id("\n")
    # === 修改结束 ===
    
    # === 1. 处理 Query 部分 (仅在 Conditioned 模式下存在) ===
    if is_conditioned:
        # System Prompt (为了保持简单，我们这里只用 User 格式，如果你训练加了 System 请取消下面注释)
        # system_text = "system\nYou are a helpful assistant.\n"
        # system_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # system_tokens = [im_start_id] + system_ids + [im_end_id, nl_id]
        
        # User Query
        # 构建: <|im_start|>user\n{query}<|im_end|>\n
        query_ids = tokenizer.encode(f"user\n{query}", add_special_tokens=False)
        user_tokens = [im_start_id] + query_ids + [im_end_id, nl_id]
        
        # Assistant Header
        # 构建: <|im_start|>assistant\n
        assist_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
        assist_header = [im_start_id] + assist_ids
        
        prefix_tokens = user_tokens + assist_header
        
        input_ids_list.extend(prefix_tokens)
        # Query 部分全部 Mask 掉 (-100)
        labels_list.extend([-100] * len(prefix_tokens))
    else:
        pass

    # === 2. 处理 Response Segments ===
    for content, seg_type in response_segments:
        # Tokenize 每一小段
        seg_tokens = tokenizer.encode(content, add_special_tokens=False)
        
        input_ids_list.extend(seg_tokens)
        
        if seg_type == "ignore":
            # 如果是 Observation，Mask 掉
            labels_list.extend([-100] * len(seg_tokens))
        else:
            # 如果是 CoT 或 Action，保留 Label
            labels_list.extend(seg_tokens)

    # 添加结尾符
    if is_conditioned:
        input_ids_list.append(im_end_id)
        labels_list.append(im_end_id)

    # 再次检查是否有 None (调试保护)
    if None in input_ids_list:
        print(f"Warning: input_ids_list contains None! Query: {query[:20]}")
        # 简单的回退策略：将 None 替换为 pad_token_id 或 0
        input_ids_list = [x if x is not None else 0 for x in input_ids_list]

    # 转为 Tensor
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
    labels = torch.tensor([labels_list], dtype=torch.long).to(device)
    
    return input_ids, labels




def calculate_loss(model, input_ids, labels):
    """计算 Loss，自动处理 Labels 中的 -100"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss.item()

def main():
    # 1. 初始化
    all_files = find_json_files(DATA_DIRS)
    print(f"找到 {len(all_files)} 个数据文件。")
    
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    device = model.device
    
    results = []
    
    # 2. 遍历处理
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for json_file in tqdm(all_files, desc="计算 IFD"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                query = data.get("query", "")
                response_obj = data.get("response", [])
                
                # 解析结构
                segments = format_response_structure(response_obj)
                
                if not segments:
                    continue

                # === 计算 Conditioned Score s(A|Q) ===
                # 包含 Query，Mask 掉 Query 和 Observation
                input_ids_cond, labels_cond = tokenize_and_mask(
                    tokenizer, query, segments, device, is_conditioned=True
                )
                
                # 如果这一条数据太长超过模型限制，需要截断或跳过
                if input_ids_cond.size(1) > 32768: # Qwen2.5 支持 32k/128k
                    print(f"Skipping {json_file}: too long ({input_ids_cond.size(1)})")
                    continue

                loss_cond = calculate_loss(model, input_ids_cond, labels_cond)
                
                # === 计算 Direct Score s(A) ===
                # 不包含 Query，Mask 掉 Observation
                input_ids_direct, labels_direct = tokenize_and_mask(
                    tokenizer, query, segments, device, is_conditioned=False
                )
                loss_direct = calculate_loss(model, input_ids_direct, labels_direct)
                
                # === 计算 IFD ===
                # 防止除零错误
                if loss_direct == 0:
                    ifd = 0
                else:
                    ifd = loss_cond / loss_direct
                
                # === 记录结果 ===
                result_record = {
                    "file_path": json_file,
                    "loss_conditioned": loss_cond,
                    "loss_direct": loss_direct,
                    "ifd_score": ifd,
                    "query": query, # 可选，方便查看
                    "response_steps": len(response_obj) # 记录步数，方便后续筛选
                }
                
                # 写入文件 (JSONL)
                f_out.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                f_out.flush() # 实时写入，防止程序崩溃丢失数据
                
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {str(e)}")
                continue

    print(f"完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
