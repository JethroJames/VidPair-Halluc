import json
import os
import csv
import logging
import argparse
from datetime import datetime
from itertools import combinations


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VideoQA binary prediction JSON files.")
    parser.add_argument(
        "--base-path",
        default=".",
        help="Directory containing *_binary_qa.json and *_T_binary_qa.json prediction files.",
    )
    return parser.parse_args()

def extract_yes_no(ans):
    """提取yes/no答案"""
    if not isinstance(ans, str):
        return "unknown"
    ans = ans.strip().lower()
    
    # 检查是否是无效的预测值
    invalid_values = ["", "unknown", "文件未找到", "file not found", "video not found", "error", "无法处理", "cannot process"]
    if ans in invalid_values:
        return "unknown"
    
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

def extract_category_from_id(id_str):
    """从ID中提取类别信息 - 取最右边下划线左边的字符串作为类别"""
    if not isinstance(id_str, str):
        return "unknown"

    # 去掉T_前缀（如果存在）
    if id_str.startswith('T_'):
        id_str = id_str[2:]

    # 找到最右边的下划线位置
    last_underscore_pos = id_str.rfind('_')
    if last_underscore_pos != -1:
        # 取最右边下划线左边的部分作为类别
        category = id_str[:last_underscore_pos]
        return category
    else:
        # 如果没有下划线，返回整个字符串
        return id_str if id_str else "unknown"

def extract_specific_category_from_id(id_str):
    """从ID中提取8个特定类别之一：action, color, location, number, object, person, dynamic_relation, static_relation, dynamic_attribute"""
    if not isinstance(id_str, str):
        return "unknown"

    # 去掉T_前缀（如果存在）
    if id_str.startswith('T_'):
        id_str = id_str[2:]

    # 定义8个特定类别
    specific_categories = [
        'action', 'color', 'location', 'number', 'object', 'person', 
        'dynamic_relation', 'static_relation', 'dynamic_attribute'
    ]
    
    # 找到最右边的下划线位置
    last_underscore_pos = id_str.rfind('_')
    if last_underscore_pos != -1:
        # 取最右边下划线左边的部分作为类别
        category = id_str[:last_underscore_pos]
        # 检查是否在特定类别列表中
        if category in specific_categories:
            return category
    
    return "other"

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

def has_paired_videos_regular(data, logger):
    """检查普通binary_qa文件是否有成对的视频文件"""
    # 检查predicted数据中是否有足够的视频进行配对
    if "predicted" not in data:
        logger.warning(f"ID {data.get('id', 'unknown')}: 缺少predicted字段")
        return False
    
    predicted = data["predicted"]
    if "binary_qa_pos" not in predicted or "binary_qa_neg" not in predicted:
        logger.warning(f"ID {data.get('id', 'unknown')}: 缺少binary_qa_pos或binary_qa_neg字段")
        return False
    
    # 收集GT数据中的所有视频文件名
    gt_files = []
    if "binary_qa_pos" in data and "binary_qa_neg" in data:
        for qa_list in [data["binary_qa_pos"], data["binary_qa_neg"]]:
            for qa in qa_list:
                if "file_name" in qa:
                    gt_files.append(qa["file_name"])
    
    # 收集predicted数据中的有效预测
    valid_predictions = 0
    predicted_files = []
    
    for qa_list in [predicted["binary_qa_pos"], predicted["binary_qa_neg"]]:
        for qa in qa_list:
            if "file_name" in qa:
                predicted_files.append(qa["file_name"])
                # 检查是否有有效的预测值
                pred_val = safe_get_prediction(qa)
                if pred_val is not None and pred_val.strip().lower() not in ["", "unknown", "文件未找到", "file not found", "video not found"]:
                    valid_predictions += 1
    
    # 检查GT数据中是否有至少2个不同的视频文件，且predicted数据中有足够的有效预测
    unique_gt_files = set(gt_files)
    unique_predicted_files = set(predicted_files)
    
    if len(unique_gt_files) >= 2 and valid_predictions >= 2:
        logger.info(f"ID {data.get('id', 'unknown')}: 成对视频 - GT文件: {sorted(unique_gt_files)}, 有效预测数: {valid_predictions}")
        return True
    else:
        if len(unique_gt_files) < 2:
            logger.warning(f"ID {data.get('id', 'unknown')}: 不成对视频 - GT文件数不足: {sorted(unique_gt_files)}")
        else:
            logger.warning(f"ID {data.get('id', 'unknown')}: 不成对视频 - 有效预测数不足: GT文件: {sorted(unique_gt_files)}, 有效预测数: {valid_predictions}")
        return False

def has_paired_videos_T(data, logger):
    """检查T_binary_qa文件是否有成对的视频文件（T文件总是成对的）"""
    # T文件总是成对的，因为每个id都有对应的T*_0.mp4
    if "predicted" in data and "binary_qa_pos" in data["predicted"] and "binary_qa_neg" in data["predicted"]:
        # 收集所有视频文件名并检查有效预测
        all_files = []
        valid_predictions = 0
        
        for qa_list in [data["predicted"]["binary_qa_pos"], data["predicted"]["binary_qa_neg"]]:
            for qa in qa_list:
                if "file_name" in qa:
                    all_files.append(qa["file_name"])
                    # 检查是否有有效的预测值
                    pred_val = safe_get_prediction(qa)
                    if pred_val is not None and pred_val.strip().lower() not in ["", "unknown", "文件未找到", "file not found", "video not found"]:
                        valid_predictions += 1
        
        unique_files = set(all_files)
        if valid_predictions >= 2:
            logger.info(f"ID {data.get('id', 'unknown')}: T文件成对视频 - 文件: {sorted(unique_files)}, 有效预测数: {valid_predictions}")
            return True
        else:
            logger.warning(f"ID {data.get('id', 'unknown')}: T文件有效预测数不足 - 文件: {sorted(unique_files)}, 有效预测数: {valid_predictions}")
            return False
    else:
        logger.warning(f"ID {data.get('id', 'unknown')}: T文件缺少必要字段")
        return False

def evaluate_binary_qa_regular(json_data, logger):
    """评测普通binary_qa文件"""
    vid_pair_total = 0
    vid_pair_correct = 0
    text_pair_total = 0
    text_pair_correct = 0
    FP = 0
    model_yes_count = 0
    gt_yes_count = 0
    total_qa = 0
    skipped_samples = 0
    
    # 按类别统计
    category_stats = {}
    # 按8个特定类别统计
    specific_category_stats = {}

    for data in json_data:
        # 提取类别信息
        category = extract_category_from_id(data.get('id', ''))
        if category not in category_stats:
            category_stats[category] = {
                'vid_pair_total': 0,
                'vid_pair_correct': 0,
                'text_pair_total': 0,
                'text_pair_correct': 0,
                'total_qa': 0,
                'model_yes_count': 0,
                'gt_yes_count': 0,
                'FP': 0
            }
        
        # 提取8个特定类别信息
        specific_category = extract_specific_category_from_id(data.get('id', ''))
        if specific_category not in specific_category_stats:
            specific_category_stats[specific_category] = {
                'vid_pair_total': 0,
                'vid_pair_correct': 0,
                'text_pair_total': 0,
                'text_pair_correct': 0,
                'total_qa': 0,
                'model_yes_count': 0,
                'gt_yes_count': 0,
                'FP': 0
            }
        
        # 检查是否有predicted字段
        if "predicted" not in data:
            logger.warning(f"ID {data.get('id', 'unknown')}: 缺少predicted字段，跳过")
            skipped_samples += 1
            continue
            
        # 检查是否有必要的键
        if "binary_qa_pos" not in data or "binary_qa_neg" not in data:
            logger.warning(f"ID {data.get('id', 'unknown')}: 缺少binary_qa_pos或binary_qa_neg字段，跳过")
            skipped_samples += 1
            continue
        
        # 检查是否有成对的视频（普通文件需要检查）
        if not has_paired_videos_regular(data, logger):
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
        
        # 统计GT中的yes数量
        gt_yes_count += sum(1 for v in gt_pos.values() if v == "yes")
        gt_yes_count += sum(1 for v in gt_neg.values() if v == "yes")

        # 只统计有预测值的样本
        for file_name, gt_ans in gt_pos.items():
            pred_ans = pred_pos.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                total_qa += 1
                category_stats[category]['total_qa'] += 1
                if pred_ans == "yes":
                    model_yes_count += 1
                    category_stats[category]['model_yes_count'] += 1
                if gt_ans == "yes":
                    category_stats[category]['gt_yes_count'] += 1
                if pred_ans == "yes" and gt_ans == "no":
                    FP += 1
                    category_stats[category]['FP'] += 1
                    
        for file_name, gt_ans in gt_neg.items():
            pred_ans = pred_neg.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                total_qa += 1
                category_stats[category]['total_qa'] += 1
                if pred_ans == "yes":
                    model_yes_count += 1
                    category_stats[category]['model_yes_count'] += 1
                if gt_ans == "yes":
                    category_stats[category]['gt_yes_count'] += 1
                if pred_ans == "yes" and gt_ans == "no":
                    FP += 1
                    category_stats[category]['FP'] += 1

        # 计算vid_pair qACC：组内正确性
        # 对于binary_qa_pos组
        pos_correct = True
        pos_has_valid_pred = False
        for file_name, gt_ans in gt_pos.items():
            pred_ans = pred_pos.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                pos_has_valid_pred = True
                if pred_ans != gt_ans:
                    pos_correct = False
                    break
        
        if pos_has_valid_pred:
            vid_pair_total += 1
            category_stats[category]['vid_pair_total'] += 1
            if pos_correct:
                vid_pair_correct += 1
                category_stats[category]['vid_pair_correct'] += 1
        
        # 对于binary_qa_neg组
        neg_correct = True
        neg_has_valid_pred = False
        for file_name, gt_ans in gt_neg.items():
            pred_ans = pred_neg.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                neg_has_valid_pred = True
                if pred_ans != gt_ans:
                    neg_correct = False
                    break
        
        if neg_has_valid_pred:
            vid_pair_total += 1
            category_stats[category]['vid_pair_total'] += 1
            if neg_correct:
                vid_pair_correct += 1
                category_stats[category]['vid_pair_correct'] += 1

        # 计算text_pair qACC：跨组正确性
        # 找到在pos和neg中都存在的视频文件
        common_files = set(gt_pos.keys()) & set(gt_neg.keys())
        for file_name in common_files:
            pos_pred = pred_pos.get(file_name)
            neg_pred = pred_neg.get(file_name)
            pos_gt = gt_pos.get(file_name)
            neg_gt = gt_neg.get(file_name)
            
            # 只有当两个预测都有效时才计算
            if (pos_pred is not None and pos_pred in ["yes", "no"] and 
                neg_pred is not None and neg_pred in ["yes", "no"]):
                text_pair_total += 1
                category_stats[category]['text_pair_total'] += 1
                if (pos_pred == pos_gt and neg_pred == neg_gt):
                    text_pair_correct += 1
                    category_stats[category]['text_pair_correct'] += 1

    vid_pair_qacc = vid_pair_correct / vid_pair_total if vid_pair_total > 0 else None
    text_pair_qacc = text_pair_correct / text_pair_total if text_pair_total > 0 else None
    pct_diff = (model_yes_count - gt_yes_count) / total_qa if total_qa > 0 else None

    # 计算各类别的wACC
    category_wacc = {}
    for category, stats in category_stats.items():
        if stats['vid_pair_total'] > 0 or stats['text_pair_total'] > 0:
            vid_qacc = stats['vid_pair_correct'] / stats['vid_pair_total'] if stats['vid_pair_total'] > 0 else 0
            text_qacc = stats['text_pair_correct'] / stats['text_pair_total'] if stats['text_pair_total'] > 0 else 0
            total_pairs = stats['vid_pair_total'] + stats['text_pair_total']
            if total_pairs > 0:
                wacc = (vid_qacc * stats['vid_pair_total'] + text_qacc * stats['text_pair_total']) / total_pairs
                category_wacc[category] = {
                    'wACC': wacc,
                    'vid_pair_qACC': vid_qacc,
                    'text_pair_qACC': text_qacc,
                    'vid_pair_total': stats['vid_pair_total'],
                    'text_pair_total': stats['text_pair_total'],
                    'total_qa': stats['total_qa'],
                    'FP': stats['FP'],
                    'FP_rate': stats['FP'] / stats['total_qa'] if stats['total_qa'] > 0 else 0
                }

    return {
        "vid_pair_qACC": vid_pair_qacc,
        "text_pair_qACC": text_pair_qacc,
        "FP": FP,
        "FP_rate": FP / total_qa if total_qa > 0 else None,
        "Pct_Diff": pct_diff,
        "model_yes_count": model_yes_count,
        "gt_yes_count": gt_yes_count,
        "total_qa": total_qa,
        "vid_pair_total": vid_pair_total,
        "text_pair_total": text_pair_total,
        "skipped_samples": skipped_samples,
        "category_wacc": category_wacc
    }

def evaluate_binary_qa_T(json_data, logger):
    """评测T_binary_qa文件"""
    vid_pair_total = 0
    vid_pair_correct = 0
    text_pair_total = 0
    text_pair_correct = 0
    FP = 0
    model_yes_count = 0
    gt_yes_count = 0
    total_qa = 0
    skipped_samples = 0
    
    # 按类别统计
    category_stats = {}

    for data in json_data:
        # 提取类别信息
        category = extract_category_from_id(data.get('id', ''))
        if category not in category_stats:
            category_stats[category] = {
                'vid_pair_total': 0,
                'vid_pair_correct': 0,
                'text_pair_total': 0,
                'text_pair_correct': 0,
                'total_qa': 0,
                'model_yes_count': 0,
                'gt_yes_count': 0,
                'FP': 0
            }
        
        # 检查是否有predicted字段
        if "predicted" not in data:
            logger.warning(f"ID {data.get('id', 'unknown')}: 缺少predicted字段，跳过")
            skipped_samples += 1
            continue
            
        # 检查是否有必要的键
        if "binary_qa_pos" not in data or "binary_qa_neg" not in data:
            logger.warning(f"ID {data.get('id', 'unknown')}: 缺少binary_qa_pos或binary_qa_neg字段，跳过")
            skipped_samples += 1
            continue
        
        # T文件总是成对的，不需要额外检查
        if not has_paired_videos_T(data, logger):
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
        
        # 统计GT中的yes数量
        gt_yes_count += sum(1 for v in gt_pos.values() if v == "yes")
        gt_yes_count += sum(1 for v in gt_neg.values() if v == "yes")

        # 只统计有预测值的样本
        for file_name, gt_ans in gt_pos.items():
            pred_ans = pred_pos.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                total_qa += 1
                category_stats[category]['total_qa'] += 1
                if pred_ans == "yes":
                    model_yes_count += 1
                    category_stats[category]['model_yes_count'] += 1
                if gt_ans == "yes":
                    category_stats[category]['gt_yes_count'] += 1
                if pred_ans == "yes" and gt_ans == "no":
                    FP += 1
                    category_stats[category]['FP'] += 1
                    
        for file_name, gt_ans in gt_neg.items():
            pred_ans = pred_neg.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                total_qa += 1
                category_stats[category]['total_qa'] += 1
                if pred_ans == "yes":
                    model_yes_count += 1
                    category_stats[category]['model_yes_count'] += 1
                if gt_ans == "yes":
                    category_stats[category]['gt_yes_count'] += 1
                if pred_ans == "yes" and gt_ans == "no":
                    FP += 1
                    category_stats[category]['FP'] += 1

        # 计算vid_pair qACC：组内正确性
        # 对于binary_qa_pos组
        pos_correct = True
        pos_has_valid_pred = False
        for file_name, gt_ans in gt_pos.items():
            pred_ans = pred_pos.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                pos_has_valid_pred = True
                if pred_ans != gt_ans:
                    pos_correct = False
                    break
        
        if pos_has_valid_pred:
            vid_pair_total += 1
            category_stats[category]['vid_pair_total'] += 1
            if pos_correct:
                vid_pair_correct += 1
                category_stats[category]['vid_pair_correct'] += 1
        
        # 对于binary_qa_neg组
        neg_correct = True
        neg_has_valid_pred = False
        for file_name, gt_ans in gt_neg.items():
            pred_ans = pred_neg.get(file_name)
            if pred_ans is not None and pred_ans in ["yes", "no"]:
                neg_has_valid_pred = True
                if pred_ans != gt_ans:
                    neg_correct = False
                    break
        
        if neg_has_valid_pred:
            vid_pair_total += 1
            category_stats[category]['vid_pair_total'] += 1
            if neg_correct:
                vid_pair_correct += 1
                category_stats[category]['vid_pair_correct'] += 1

        # 计算text_pair qACC：跨组正确性
        # 找到在pos和neg中都存在的视频文件
        common_files = set(gt_pos.keys()) & set(gt_neg.keys())
        for file_name in common_files:
            pos_pred = pred_pos.get(file_name)
            neg_pred = pred_neg.get(file_name)
            pos_gt = gt_pos.get(file_name)
            neg_gt = gt_neg.get(file_name)
            
            # 只有当两个预测都有效时才计算
            if (pos_pred is not None and pos_pred in ["yes", "no"] and 
                neg_pred is not None and neg_pred in ["yes", "no"]):
                text_pair_total += 1
                category_stats[category]['text_pair_total'] += 1
                if (pos_pred == pos_gt and neg_pred == neg_gt):
                    text_pair_correct += 1
                    category_stats[category]['text_pair_correct'] += 1

    vid_pair_qacc = vid_pair_correct / vid_pair_total if vid_pair_total > 0 else None
    text_pair_qacc = text_pair_correct / text_pair_total if text_pair_total > 0 else None
    pct_diff = (model_yes_count - gt_yes_count) / total_qa if total_qa > 0 else None

    # 计算各类别的wACC
    category_wacc = {}
    for category, stats in category_stats.items():
        if stats['vid_pair_total'] > 0 or stats['text_pair_total'] > 0:
            vid_qacc = stats['vid_pair_correct'] / stats['vid_pair_total'] if stats['vid_pair_total'] > 0 else 0
            text_qacc = stats['text_pair_correct'] / stats['text_pair_total'] if stats['text_pair_total'] > 0 else 0
            total_pairs = stats['vid_pair_total'] + stats['text_pair_total']
            if total_pairs > 0:
                wacc = (vid_qacc * stats['vid_pair_total'] + text_qacc * stats['text_pair_total']) / total_pairs
                category_wacc[category] = {
                    'wACC': wacc,
                    'vid_pair_qACC': vid_qacc,
                    'text_pair_qACC': text_qacc,
                    'vid_pair_total': stats['vid_pair_total'],
                    'text_pair_total': stats['text_pair_total'],
                    'total_qa': stats['total_qa'],
                    'FP': stats['FP'],
                    'FP_rate': stats['FP'] / stats['total_qa'] if stats['total_qa'] > 0 else 0
                }

    return {
        "vid_pair_qACC": vid_pair_qacc,
        "text_pair_qACC": text_pair_qacc,
        "FP": FP,
        "FP_rate": FP / total_qa if total_qa > 0 else None,
        "Pct_Diff": pct_diff,
        "model_yes_count": model_yes_count,
        "gt_yes_count": gt_yes_count,
        "total_qa": total_qa,
        "vid_pair_total": vid_pair_total,
        "text_pair_total": text_pair_total,
        "skipped_samples": skipped_samples,
        "category_wacc": category_wacc
    }

def get_model_order(model_name):
    """获取模型在排序中的优先级，数字越小优先级越高"""
    model_order = {
        'Video-ChatGPT': 1,
        'Video-LLaVA': 2,
        'VideoChat2': 3,
        'VideoLLaMA2': 4,
        'Qwen2.5-VL': 5,
        'R1-Onevision-7B': 6,
        'ThinkLite-VL': 7,
        'ShareGPT4Video': 8,
        'PLLaVA': 9,
        'LLaMA-VID': 10,
        'VILA': 11,
        'gpt-4o': 12,
        'gemini-2.5-flash': 13
    }
    return model_order.get(model_name, 999)  # 未知模型排在最后

def setup_logging(log_file):
    """设置日志"""
    logger = logging.getLogger('videoqa_evaluation')
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    args = parse_args()
    base_path = args.base_path
    
    # 创建输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    regular_output_dir = os.path.join(base_path, 'binary_qa')
    t_output_dir = os.path.join(base_path, 'T_binary_qa')
    
    # 创建输出目录
    os.makedirs(regular_output_dir, exist_ok=True)
    os.makedirs(t_output_dir, exist_ok=True)
    
    # 创建输出文件路径
    regular_log_file = os.path.join(regular_output_dir, f'videoqa_evaluation_{timestamp}.log')
    regular_csv_file = os.path.join(regular_output_dir, f'videoqa_evaluation_{timestamp}.csv')
    t_log_file = os.path.join(t_output_dir, f'videoqa_evaluation_{timestamp}.log')
    t_csv_file = os.path.join(t_output_dir, f'videoqa_evaluation_{timestamp}.csv')
    
    # 设置日志
    regular_logger = setup_logging(regular_log_file)
    t_logger = setup_logging(t_log_file)
    
    # 获取所有json文件
    all_files = [f for f in os.listdir(base_path) if f.endswith('.json')]
    
    # 分类文件
    regular_files = [f for f in all_files if f.endswith('_binary_qa.json') and not f.endswith('_T_binary_qa.json')]
    t_files = [f for f in all_files if f.endswith('_T_binary_qa.json')]
    
    print("=" * 60)
    print("VideoQA 二分类评测结果 (修正版)")
    print("=" * 60)
    
    results = {'regular': [], 't_files': []}
    
    # 评测普通binary_qa文件
    if regular_files:
        regular_logger.info("\n【普通 binary_qa 文件评测开始】")
        regular_logger.info("-" * 40)
        for json_file in sorted(regular_files):
            json_path = os.path.join(base_path, json_file)
            try:
                regular_logger.info(f"开始评测文件: {json_file}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                result = evaluate_binary_qa_regular(json_data, regular_logger)
                
                # 提取模型名称
                model_name = json_file.replace('_binary_qa.json', '')
                result['model'] = model_name
                results['regular'].append(result)
                
                regular_logger.info(f'文件: {json_file} 评测完成')
                regular_logger.info(f'vid_pair_qACC: {result["vid_pair_qACC"]:.4f}' if result["vid_pair_qACC"] is not None else 'vid_pair_qACC: None')
                regular_logger.info(f'text_pair_qACC: {result["text_pair_qACC"]:.4f}' if result["text_pair_qACC"] is not None else 'text_pair_qACC: None')
                regular_logger.info(f'FP: {result["FP"]}')
                regular_logger.info(f'FP_rate: {result["FP_rate"]:.4f}' if result["FP_rate"] is not None else 'FP_rate: None')
                regular_logger.info(f'Pct_Diff: {result["Pct_Diff"]:.4f}' if result["Pct_Diff"] is not None else 'Pct_Diff: None')
                regular_logger.info(f'总QA数: {result["total_qa"]}, vid_pair总数: {result["vid_pair_total"]}, text_pair总数: {result["text_pair_total"]}, 跳过样本: {result["skipped_samples"]}')
                
                # 输出各类别的wACC
                if 'category_wacc' in result and result['category_wacc']:
                    regular_logger.info('各类别wACC统计:')
                    for category, stats in result['category_wacc'].items():
                        regular_logger.info(f'  {category}: wACC={stats["wACC"]:.4f}, vid_pair_qACC={stats["vid_pair_qACC"]:.4f}, text_pair_qACC={stats["text_pair_qACC"]:.4f}, 样本数={stats["total_qa"]}')
                
                regular_logger.info('-' * 40)
                
            except Exception as e:
                regular_logger.error(f'评测文件 {json_file} 时出错: {str(e)}')
                continue

    # 评测T_binary_qa文件
    if t_files:
        t_logger.info("\n【T_binary_qa 文件评测开始】")
        t_logger.info("-" * 40)
        for json_file in sorted(t_files):
            json_path = os.path.join(base_path, json_file)
            try:
                t_logger.info(f"开始评测文件: {json_file}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                result = evaluate_binary_qa_T(json_data, t_logger)
                
                # 提取模型名称
                model_name = json_file.replace('_T_binary_qa.json', '')
                result['model'] = model_name
                results['t_files'].append(result)
                
                t_logger.info(f'文件: {json_file} 评测完成')
                t_logger.info(f'vid_pair_qACC: {result["vid_pair_qACC"]:.4f}' if result["vid_pair_qACC"] is not None else 'vid_pair_qACC: None')
                t_logger.info(f'text_pair_qACC: {result["text_pair_qACC"]:.4f}' if result["text_pair_qACC"] is not None else 'text_pair_qACC: None')
                t_logger.info(f'FP: {result["FP"]}')
                t_logger.info(f'FP_rate: {result["FP_rate"]:.4f}' if result["FP_rate"] is not None else 'FP_rate: None')
                t_logger.info(f'Pct_Diff: {result["Pct_Diff"]:.4f}' if result["Pct_Diff"] is not None else 'Pct_Diff: None')
                t_logger.info(f'总QA数: {result["total_qa"]}, vid_pair总数: {result["vid_pair_total"]}, text_pair总数: {result["text_pair_total"]}, 跳过样本: {result["skipped_samples"]}')
                
                # 输出各类别的wACC
                if 'category_wacc' in result and result['category_wacc']:
                    t_logger.info('各类别wACC统计:')
                    for category, stats in result['category_wacc'].items():
                        t_logger.info(f'  {category}: wACC={stats["wACC"]:.4f}, vid_pair_qACC={stats["vid_pair_qACC"]:.4f}, text_pair_qACC={stats["text_pair_qACC"]:.4f}, 样本数={stats["total_qa"]}')
                
                t_logger.info('-' * 40)
                
            except Exception as e:
                t_logger.error(f'评测文件 {json_file} 时出错: {str(e)}')
                continue

    # 保存结果到CSV
    # 保存普通binary_qa结果
    if results['regular']:
        with open(regular_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'vid_pair_qACC', 'text_pair_qACC', 'FP', 'FP_rate', 'Pct_Diff', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'Skipped_Samples'])
            
            # 按模型顺序排序
            sorted_results = sorted(results['regular'], key=lambda x: get_model_order(x['model']))
            for result in sorted_results:
                writer.writerow([
                    result['model'],
                    'Regular',
                    f"{result['vid_pair_qACC']:.4f}" if result['vid_pair_qACC'] is not None else "None",
                    f"{result['text_pair_qACC']:.4f}" if result['text_pair_qACC'] is not None else "None",
                    result['FP'],
                    f"{result['FP_rate']:.4f}" if result['FP_rate'] is not None else "None",
                    f"{result['Pct_Diff']:.4f}" if result['Pct_Diff'] is not None else "None",
                    result['total_qa'],
                    result['vid_pair_total'],
                    result['text_pair_total'],
                    result['skipped_samples']
                ])

    # 保存T_binary_qa结果
    if results['t_files']:
        with open(t_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'vid_pair_qACC', 'text_pair_qACC', 'FP', 'FP_rate', 'Pct_Diff', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'Skipped_Samples'])
            
            # 按模型顺序排序
            sorted_results = sorted(results['t_files'], key=lambda x: get_model_order(x['model']))
            for result in sorted_results:
                writer.writerow([
                    result['model'],
                    'T_Files',
                    f"{result['vid_pair_qACC']:.4f}" if result['vid_pair_qACC'] is not None else "None",
                    f"{result['text_pair_qACC']:.4f}" if result['text_pair_qACC'] is not None else "None",
                    result['FP'],
                    f"{result['FP_rate']:.4f}" if result['FP_rate'] is not None else "None",
                    f"{result['Pct_Diff']:.4f}" if result['Pct_Diff'] is not None else "None",
                    result['total_qa'],
                    result['vid_pair_total'],
                    result['text_pair_total'],
                    result['skipped_samples']
                ])

    # 保存类别统计结果
    # 保存普通binary_qa的类别统计
    if results['regular']:
        category_csv_file = os.path.join(regular_output_dir, f'videoqa_category_stats_{timestamp}.csv')
        with open(category_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'Category', 'wACC', 'vid_pair_qACC', 'text_pair_qACC', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'FP', 'FP_rate'])
            
            for result in results['regular']:
                if 'category_wacc' in result and result['category_wacc']:
                    for category, stats in result['category_wacc'].items():
                        writer.writerow([
                            result['model'],
                            'Regular',
                            category,
                            f"{stats['wACC']:.4f}",
                            f"{stats['vid_pair_qACC']:.4f}",
                            f"{stats['text_pair_qACC']:.4f}",
                            stats['total_qa'],
                            stats['vid_pair_total'],
                            stats['text_pair_total'],
                            stats['FP'],
                            f"{stats['FP_rate']:.4f}"
                        ])

    # 保存T_binary_qa的类别统计
    if results['t_files']:
        t_category_csv_file = os.path.join(t_output_dir, f'videoqa_category_stats_{timestamp}.csv')
        with open(t_category_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'Category', 'wACC', 'vid_pair_qACC', 'text_pair_qACC', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'FP', 'FP_rate'])
            
            for result in results['t_files']:
                if 'category_wacc' in result and result['category_wacc']:
                    for category, stats in result['category_wacc'].items():
                        writer.writerow([
                            result['model'],
                            'T_Files',
                            category,
                            f"{stats['wACC']:.4f}",
                            f"{stats['vid_pair_qACC']:.4f}",
                            f"{stats['text_pair_qACC']:.4f}",
                            stats['total_qa'],
                            stats['vid_pair_total'],
                            stats['text_pair_total'],
                            stats['FP'],
                            f"{stats['FP_rate']:.4f}"
                        ])

    # 打印最终结果摘要
    print("\n" + "=" * 80)
    print("VideoQA 二分类评测结果摘要 (修正版)")
    print("=" * 80)
    
    # 合并所有结果并按模型顺序排序
    all_results = results['regular'] + results['t_files']
    all_results.sort(key=lambda x: get_model_order(x['model']))
    
    print(f"{'排名':<4} {'模型':<20} {'类型':<10} {'vid_pair_qACC':<12} {'text_pair_qACC':<12} {'FP_rate':<8}")
    print("-" * 80)
    
    for i, result in enumerate(all_results, 1):
        model = result['model']
        file_type = 'Regular' if result in results['regular'] else 'T_Files'
        vid_qacc = f"{result['vid_pair_qACC']:.4f}" if result['vid_pair_qACC'] is not None else "None"
        text_qacc = f"{result['text_pair_qACC']:.4f}" if result['text_pair_qACC'] is not None else "None"
        fp_rate = f"{result['FP_rate']:.4f}" if result['FP_rate'] is not None else "None"
        
        print(f"{i:<4} {model:<20} {file_type:<10} {vid_qacc:<12} {text_qacc:<12} {fp_rate:<8}")

    print(f"\n普通binary_qa结果已保存到:")
    print(f"  日志文件: {regular_log_file}")
    print(f"  CSV文件: {regular_csv_file}")
    if results['regular']:
        print(f"  类别统计CSV文件: {category_csv_file}")
    
    print(f"\nT_binary_qa结果已保存到:")
    print(f"  日志文件: {t_log_file}")
    print(f"  CSV文件: {t_csv_file}")
    if results['t_files']:
        print(f"  类别统计CSV文件: {t_category_csv_file}")

if __name__ == "__main__":
    main()
