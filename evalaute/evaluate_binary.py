import json
import os
from itertools import combinations

def extract_yes_no(ans):
    if not isinstance(ans, str):
        return "unknown"
    ans = ans.strip().lower()
    if ans.startswith("yes"):
        return "yes"
    if ans.startswith("no"):
        return "no"
    if "yes" in ans:
        return "yes"
    if "no" in ans:
        return "no"
    # unknown等情况一律当作错误
    return "unknown"

def safe_get_prediction(qa_item):
    """安全获取预测值，支持多种键名"""
    # 首先尝试 prediction 键
    if "prediction" in qa_item:
        return qa_item["prediction"]
    
    # 如果 prediction 键不存在，尝试在 predicted 内部查找
    if "predicted" in qa_item and isinstance(qa_item["predicted"], dict):
        # 尝试常见的预测键名
        for key in ["prediction", "answer", "result", "output"]:
            if key in qa_item["predicted"]:
                return qa_item["predicted"][key]
    
    return None

def has_paired_videos_clip_level(qa_keys):
    """检查clip-level是否有成对的视频文件（只要有binary_qa_0/1/2且不为空）"""
    # 检查是否有binary_qa_0, binary_qa_1, binary_qa_2等键
    binary_qa_keys = [k for k in qa_keys if k.startswith('binary_qa_') and k[9:].isdigit()]
    return len(binary_qa_keys) >= 2

def has_paired_videos_video_level(file_list):
    """检查video-level是否有成对的视频文件"""
    # 提取视频类型前缀
    prefixes = set()
    for file_name in file_list:
        if file_name.endswith('.mp4'):
            # 提取前缀，如 action_0002_0.mp4 -> action_0002
            parts = file_name.split('_')
            if len(parts) >= 3:
                prefix = '_'.join(parts[:-1])  # 去掉最后的数字部分
                prefixes.add(prefix)
    
    # 检查每个前缀是否有至少两个不同的视频
    for prefix in prefixes:
        matching_files = [f for f in file_list if f.startswith(prefix) and f.endswith('.mp4')]
        if len(matching_files) >= 2:
            return True
    
    return False

def evaluate_binary_qa(json_data):
    # video-level评测：兼容binary_qa_pos/binary_qa_neg结构
    total_pairs = 0
    correct_pairs = 0
    FP = 0
    model_yes_count = 0
    gt_yes_count = 0
    total_qa = 0
    skipped_samples = 0

    for data in json_data:
        # 检查是否有predicted字段
        if "predicted" not in data:
            skipped_samples += 1
            continue
            
        # 检查是否有必要的键
        if "binary_qa_pos" not in data or "binary_qa_neg" not in data:
            skipped_samples += 1
            continue
            
        # 构建GT字典
        gt_pos = {qa["file_name"]: extract_yes_no(qa["answer"]) for qa in data["binary_qa_pos"]}
        gt_neg = {qa["file_name"]: extract_yes_no(qa["answer"]) for qa in data["binary_qa_neg"]}
        
        # 构建预测字典，使用鲁棒的预测值获取
        pred_pos = {}
        pred_neg = {}
        
        for qa in data.get("predicted", {}).get("binary_qa_pos", []):
            if "file_name" in qa:
                pred_val = safe_get_prediction(qa)
                if pred_val is not None:
                    pred_pos[qa["file_name"]] = extract_yes_no(pred_val)
                    
        for qa in data.get("predicted", {}).get("binary_qa_neg", []):
            if "file_name" in qa:
                pred_val = safe_get_prediction(qa)
                if pred_val is not None:
                    pred_neg[qa["file_name"]] = extract_yes_no(pred_val)
        
        # 检查是否有成对的视频
        all_files = list(gt_pos.keys()) + list(gt_neg.keys())
        if not has_paired_videos_video_level(all_files):
            skipped_samples += 1
            continue

        gt_yes_count += sum(1 for v in gt_pos.values() if v == "yes")
        gt_yes_count += sum(1 for v in gt_neg.values() if v == "yes")

        # 只统计有预测值的样本
        for file_name, gt_ans in gt_pos.items():
            pred_ans = pred_pos.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                total_qa += 1
                if pred_ans == "yes":
                    model_yes_count += 1
                if pred_ans == "yes" and gt_ans == "no":
                    FP += 1
                    
        for file_name, gt_ans in gt_neg.items():
            pred_ans = pred_neg.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                total_qa += 1
                if pred_ans == "yes":
                    model_yes_count += 1
                if pred_ans == "yes" and gt_ans == "no":
                    FP += 1

        # 只对有效的预测进行配对
        pos_keys = [k for k in pred_pos.keys() if pred_pos[k] in ["yes", "no"]]
        for k1, k2 in combinations(pos_keys, 2):
            total_pairs += 1
            if (pred_pos[k1] == gt_pos.get(k1)) and (pred_pos[k2] == gt_pos.get(k2)):
                correct_pairs += 1

        neg_keys = [k for k in pred_neg.keys() if pred_neg[k] in ["yes", "no"]]
        for k1, k2 in combinations(neg_keys, 2):
            total_pairs += 1
            if (pred_neg[k1] == gt_neg.get(k1)) and (pred_neg[k2] == gt_neg.get(k2)):
                correct_pairs += 1

    qacc = correct_pairs / total_pairs if total_pairs > 0 else None
    pct_diff = (model_yes_count - gt_yes_count) / total_qa if total_qa > 0 else None

    return {
        "qACC": qacc,
        "FP": FP,
        "FP_rate": FP / total_qa if total_qa > 0 else None,
        "Pct_Diff": pct_diff,
        "model_yes_count": model_yes_count,
        "gt_yes_count": gt_yes_count,
        "total_qa": total_qa,
        "total_pairs": total_pairs,
        "skipped_samples": skipped_samples
    }


def evaluate_binary_qa_clip_level(json_data):
    # clip-level评测：自动适配binary_qa_0/binary_qa_1/...结构
    total_pairs = 0
    correct_pairs = 0
    FP = 0
    model_yes_count = 0
    gt_yes_count = 0
    total_qa = 0
    skipped_samples = 0

    for data in json_data:
        # 检查是否有predicted字段
        if "predicted" not in data:
            skipped_samples += 1
            continue
            
        # 找出所有binary_qa_*的key
        qa_keys = [k for k in data.keys() if k.startswith('binary_qa_')]
        if not qa_keys:
            skipped_samples += 1
            continue
            
        # 检查是否有成对的视频（clip-level逻辑）
        if not has_paired_videos_clip_level(qa_keys):
            skipped_samples += 1
            continue
            
        for qa_key in qa_keys:
            gt_list = data[qa_key]
            pred_list = data.get('predicted', {}).get(qa_key, [])

            # 构造gt字典
            gt_dict = {}
            for item in gt_list:
                if "file" in item and "answer" in item:
                    gt_dict[item['file']] = extract_yes_no(item['answer'])
            
            # 构造pred字典，使用鲁棒的预测值获取
            pred_dict = {}
            for item in pred_list:
                if "file" in item:
                    pred_val = safe_get_prediction(item)
                    if pred_val is not None:
                        pred_dict[item['file']] = extract_yes_no(pred_val)

            # 统计yes数量/FP
            for file_name, gt_ans in gt_dict.items():
                pred_ans = pred_dict.get(file_name)
                if pred_ans is not None and pred_ans in ["yes", "no"]:
                    total_qa += 1
                    if pred_ans == "yes":
                        model_yes_count += 1
                    if pred_ans == "yes" and gt_ans == "no":
                        FP += 1
                if gt_ans == "yes":
                    gt_yes_count += 1

            # 两两配对（只对有效的预测进行配对）
            keys = [k for k in pred_dict.keys() if pred_dict[k] in ["yes", "no"]]
            for k1, k2 in combinations(keys, 2):
                total_pairs += 1
                if (pred_dict[k1] == gt_dict.get(k1)) and (pred_dict[k2] == gt_dict.get(k2)):
                    correct_pairs += 1

    qacc = correct_pairs / total_pairs if total_pairs > 0 else None
    pct_diff = (model_yes_count - gt_yes_count) / total_qa if total_qa > 0 else None

    return {
        "qACC": qacc,
        "FP": FP,
        "FP_rate": FP / total_qa if total_qa > 0 else None,
        "Pct_Diff": pct_diff,
        "model_yes_count": model_yes_count,
        "gt_yes_count": gt_yes_count,
        "total_qa": total_qa,
        "total_pairs": total_pairs,
        "skipped_samples": skipped_samples
    }


def main():
    # 更新为新的目录结构
    base_video_path = '/hpc2hdd/home/hhuang118/VidHalluc/qa/VideoQA/results'
    base_clip_path = '/hpc2hdd/home/hhuang118/VidHalluc/qa/ClipQA/results'
    
    video_level_list = [
        'gpt-4o_binary_qa.json',
        'gemini-2.5-flash_binary_qa.json',
        'LLaMA-VID_binary_qa.json',
        'Qwen2.5-VL_binary_qa.json',
        'ShareGPT4Video_binary_qa.json',
        'VILA_binary_qa.json',
        'Video-LLaVA_binary_qa.json',
        'Video-ChatGPT_binary_qa.json',
        'VideoChat2_binary_qa.json',
        'VideoLLaMA2_binary_qa.json',
        'VideoLLaMA3_binary_qa.json',
        'PLLaVA_binary_qa.json',
    ]
    clip_level_list = [
        'gpt-4o_M_binary_qa.json',
        'gpt-4o_R_binary_qa.json',
        'gemini-2.5-flash_M_binary_qa.json',
        'gemini-2.5-flash_R_binary_qa.json',
        'LLaMA-VID_M_binary_qa.json',
        'LLaMA-VID_R_binary_qa.json',
        'Qwen2.5-VL_M_binary_qa.json',
        'Qwen2.5-VL_R_binary_qa.json',
        'ShareGPT4Video_M_binary_qa.json',
        'ShareGPT4Video_R_binary_qa.json',
        'VILA_M_binary_qa.json',
        'VILA_R_binary_qa.json',
        'Video-LLaVA_M_binary_qa.json',
        'Video-LLaVA_R_binary_qa.json',
        'Video-ChatGPT_M_binary_qa.json',
        'Video-ChatGPT_R_binary_qa.json',
        'VideoChat2_M_binary_qa.json',
        'VideoChat2_R_binary_qa.json',
        'VideoLLaMA2_M_binary_qa.json',
        'VideoLLaMA2_R_binary_qa.json',
        'VideoLLaMA3_M_binary_qa.json',
        'VideoLLaMA3_R_binary_qa.json',
        'PLLaVA_M_binary_qa.json',
        'PLLaVA_R_binary_qa.json',
    ]

    print("请选择评测模式：")
    print("1. video-level（二分类评测）")
    print("2. clip-level（二分类评测）")
    mode = input("请输入模式编号 (1 或 2): ").strip()

    if mode == "1":
        print("\n【video-level】评测结果：")
        for json_path in video_level_list:
            json_path_full = os.path.join(base_video_path, json_path)
            try:
                with open(json_path_full, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                result = evaluate_binary_qa(json_data)
                print(f'文件: {json_path_full}')
                print(result)
                print('-' * 40)
            except Exception as e:
                print(f'文件: {json_path_full} 评测出错: {e}')
                print('-' * 40)
    elif mode == "2":
        print("\n【clip-level】评测结果：")
        for json_path in clip_level_list:
            json_path_full = os.path.join(base_clip_path, json_path)
            try:
                with open(json_path_full, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                result = evaluate_binary_qa_clip_level(json_data)
                print(f'文件: {json_path_full}')
                print(result)
                print('-' * 40)
            except Exception as e:
                print(f'文件: {json_path_full} 评测出错: {e}')
                print('-' * 40)
    else:
        print("输入有误，请输入1或2。")

if __name__ == "__main__":
    main()
