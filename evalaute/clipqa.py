#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClipQA Binary QA 评测脚本
处理M*_binary_qa.json和R*_binary_qa.json文件
"""

import json
import os
import csv
import logging
from datetime import datetime
from collections import defaultdict

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

def is_valid_prediction(pred_val):
    """检查预测值是否有效 - 只要包含yes或no就认为是有效的"""
    if pred_val is None:
        return False
    pred_val = str(pred_val).strip().lower()
    invalid_values = ["", "unknown", "文件未找到", "file not found", "video not found", "error", "无法处理", "cannot process", "推理失败", "inference failed"]

    # 检查是否是无效值
    if pred_val in invalid_values:
        return False

    # 只要包含yes或no就认为是有效预测
    return "yes" in pred_val or "no" in pred_val

def safe_get_prediction(qa_item):
    """安全获取预测值"""
    if isinstance(qa_item, dict):
        return qa_item.get("prediction")
    return None

def has_valid_predictions(data, logger):
    """检查ClipQA数据是否有有效预测"""
    if "predicted" not in data:
        logger.warning(f"ID {data.get('id', 'unknown')}: 缺少predicted字段")
        return False
    
    predicted = data["predicted"]
    if not isinstance(predicted, dict):
        logger.warning(f"ID {data.get('id', 'unknown')}: predicted不是字典格式")
        return False
    
    # 检查是否有至少一个binary_qa_X字典
    binary_qa_keys = [key for key in predicted.keys() if key.startswith("binary_qa_")]
    if not binary_qa_keys:
        logger.warning(f"ID {data.get('id', 'unknown')}: 没有找到binary_qa_X字段")
        return False
    
    # 检查每个binary_qa_X字典是否有有效预测
    valid_predictions_count = 0
    for key in binary_qa_keys:
        qa_list = predicted[key]
        if isinstance(qa_list, list):
            for qa in qa_list:
                pred_val = safe_get_prediction(qa)
                if is_valid_prediction(pred_val):
                    valid_predictions_count += 1
    
    if valid_predictions_count >= 2:
        logger.info(f"ID {data.get('id', 'unknown')}: 有效预测数: {valid_predictions_count}")
        return True
    else:
        logger.warning(f"ID {data.get('id', 'unknown')}: 有效预测数不足: {valid_predictions_count}")
        return False

def evaluate_clipqa_binary_qa(json_data, logger):
    """评测ClipQA binary_qa文件"""
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
        
        if not has_valid_predictions(data, logger):
            skipped_samples += 1
            continue

        predicted = data["predicted"]
        binary_qa_keys = [key for key in predicted.keys() if key.startswith("binary_qa_")]
        
        # 收集GT数据
        gt_data = {}
        for key in binary_qa_keys:
            if key in data and isinstance(data[key], list):
                gt_data[key] = {}
                for qa in data[key]:
                    if "file" in qa and "answer" in qa:
                        gt_data[key][qa["file"]] = qa["answer"]

        # 收集预测数据
        pred_data = {}
        for key in binary_qa_keys:
            if key in predicted and isinstance(predicted[key], list):
                pred_data[key] = {}
                for qa in predicted[key]:
                    if "file" in qa:
                        pred_val = safe_get_prediction(qa)
                        if is_valid_prediction(pred_val):
                            pred_data[key][qa["file"]] = extract_yes_no(pred_val)

        # 计算vid_pair qACC：组内正确性
        for key in binary_qa_keys:
            if key in gt_data and key in pred_data:
                gt_dict = gt_data[key]
                pred_dict = pred_data[key]
                
                # 检查是否有有效预测
                has_valid_pred = False
                for file_name, pred_ans in pred_dict.items():
                    if pred_ans in ["yes", "no"]:
                        has_valid_pred = True
                        break
                
                if has_valid_pred:
                    vid_pair_total += 1
                    category_stats[category]['vid_pair_total'] += 1
                    specific_category_stats[specific_category]['vid_pair_total'] += 1
                    
                    # 检查组内所有预测是否正确
                    group_correct = True
                    for file_name, gt_ans in gt_dict.items():
                        pred_ans = pred_dict.get(file_name)
                        if pred_ans is not None and pred_ans in ["yes", "no"]:
                            if pred_ans != gt_ans:
                                group_correct = False
                                break
                    
                    if group_correct:
                        vid_pair_correct += 1
                        category_stats[category]['vid_pair_correct'] += 1
                        specific_category_stats[specific_category]['vid_pair_correct'] += 1

        # 计算text_pair qACC：跨组正确性
        # 找到在多个binary_qa_X中都存在的视频文件
        common_files = set()
        for key in binary_qa_keys:
            if key in gt_data:
                common_files.update(gt_data[key].keys())
        
        # 检查每个文件在多个组中的预测
        for file_name in common_files:
            file_pairs = []
            for key in binary_qa_keys:
                if key in gt_data and key in pred_data:
                    if file_name in gt_data[key] and file_name in pred_data[key]:
                        gt_ans = gt_data[key][file_name]
                        pred_ans = pred_data[key][file_name]
                        if pred_ans in ["yes", "no"]:
                            file_pairs.append((gt_ans, pred_ans))
            
            if len(file_pairs) >= 2:
                text_pair_total += 1
                category_stats[category]['text_pair_total'] += 1
                specific_category_stats[specific_category]['text_pair_total'] += 1
                # 检查该文件在所有组中的预测是否都正确
                file_correct = True
                for gt_ans, pred_ans in file_pairs:
                    if pred_ans != gt_ans:
                        file_correct = False
                        break
                
                if file_correct:
                    text_pair_correct += 1
                    category_stats[category]['text_pair_correct'] += 1
                    specific_category_stats[specific_category]['text_pair_correct'] += 1

        # 计算其他指标
        for key in binary_qa_keys:
            if key in gt_data and key in pred_data:
                gt_dict = gt_data[key]
                pred_dict = pred_data[key]
                
                for file_name, gt_ans in gt_dict.items():
                    pred_ans = pred_dict.get(file_name)
                    if pred_ans is not None and pred_ans in ["yes", "no"]:
                        total_qa += 1
                        category_stats[category]['total_qa'] += 1
                        specific_category_stats[specific_category]['total_qa'] += 1
                        if pred_ans == "yes":
                            model_yes_count += 1
                            category_stats[category]['model_yes_count'] += 1
                            specific_category_stats[specific_category]['model_yes_count'] += 1
                        if gt_ans == "yes":
                            gt_yes_count += 1
                            category_stats[category]['gt_yes_count'] += 1
                            specific_category_stats[specific_category]['gt_yes_count'] += 1
                        if pred_ans == "yes" and gt_ans == "no":
                            FP += 1
                            category_stats[category]['FP'] += 1
                            specific_category_stats[specific_category]['FP'] += 1

    vid_pair_qacc = vid_pair_correct / vid_pair_total if vid_pair_total > 0 else None
    text_pair_qacc = text_pair_correct / text_pair_total if text_pair_total > 0 else None
    pct_diff = (model_yes_count - gt_yes_count) / total_qa if total_qa > 0 else None
    
    # 计算加权平均qACC
    weighted_qacc = None
    if vid_pair_qacc is not None and text_pair_qacc is not None:
        total_pairs = vid_pair_total + text_pair_total
        if total_pairs > 0:
            weighted_qacc = (vid_pair_qacc * vid_pair_total + text_pair_qacc * text_pair_total) / total_pairs

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

    # 计算8个特定类别的wACC
    specific_category_wacc = {}
    for category, stats in specific_category_stats.items():
        if stats['vid_pair_total'] > 0 or stats['text_pair_total'] > 0:
            vid_qacc = stats['vid_pair_correct'] / stats['vid_pair_total'] if stats['vid_pair_total'] > 0 else 0
            text_qacc = stats['text_pair_correct'] / stats['text_pair_total'] if stats['text_pair_total'] > 0 else 0
            total_pairs = stats['vid_pair_total'] + stats['text_pair_total']
            if total_pairs > 0:
                wacc = (vid_qacc * stats['vid_pair_total'] + text_qacc * stats['text_pair_total']) / total_pairs
                specific_category_wacc[category] = {
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
        "weighted_qACC": weighted_qacc,
        "FP": FP,
        "FP_rate": FP / total_qa if total_qa > 0 else None,
        "Pct_Diff": pct_diff,
        "model_yes_count": model_yes_count,
        "gt_yes_count": gt_yes_count,
        "total_qa": total_qa,
        "vid_pair_total": vid_pair_total,
        "text_pair_total": text_pair_total,
        "skipped_samples": skipped_samples,
        "category_wacc": category_wacc,
        "specific_category_wacc": specific_category_wacc
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

def setup_logging(log_file, logger_name):
    """设置日志"""
    logger = logging.getLogger(logger_name)
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
    base_path = 'Your Path Here'
    
    # 创建输出文件夹
    m_output_dir = os.path.join(base_path, 'M_binary_qa')
    r_output_dir = os.path.join(base_path, 'R_binary_qa')
    
    os.makedirs(m_output_dir, exist_ok=True)
    os.makedirs(r_output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志文件
    m_log_file = os.path.join(m_output_dir, f'clipqa_evaluation_final_{timestamp}.log')
    r_log_file = os.path.join(r_output_dir, f'clipqa_evaluation_final_{timestamp}.log')
    
    # 设置CSV文件
    m_csv_file = os.path.join(m_output_dir, f'clipqa_evaluation_final_{timestamp}.csv')
    r_csv_file = os.path.join(r_output_dir, f'clipqa_evaluation_final_{timestamp}.csv')
    
    # 设置日志
    m_logger = setup_logging(m_log_file, 'M_binary_qa_evaluation')
    r_logger = setup_logging(r_log_file, 'R_binary_qa_evaluation')
    
    # 查找所有*_M_binary_qa.json和*_R_binary_qa.json文件
    m_files = []
    r_files = []
    
    for file in os.listdir(base_path):
        if file.endswith('_M_binary_qa.json'):
            m_files.append(file)
        elif file.endswith('_R_binary_qa.json'):
            r_files.append(file)
    
    print(f"找到 {len(m_files)} 个M*_binary_qa.json文件")
    print(f"找到 {len(r_files)} 个R*_binary_qa.json文件")
    
    # 存储结果
    results = {'m_files': [], 'r_files': []}
    
    # 处理M*_binary_qa.json文件
    for json_file in sorted(m_files):
        file_path = os.path.join(base_path, json_file)
        m_logger.info(f'开始评测文件: {json_file}')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = evaluate_clipqa_binary_qa(data, m_logger)
            model_name = json_file.replace('_M_binary_qa.json', '')
            result['model'] = model_name
            results['m_files'].append(result)
            
            m_logger.info(f'文件: {json_file} 评测完成')
            m_logger.info(f'vid_pair_qACC: {result["vid_pair_qACC"]:.4f}' if result["vid_pair_qACC"] is not None else 'vid_pair_qACC: None')
            m_logger.info(f'text_pair_qACC: {result["text_pair_qACC"]:.4f}' if result["text_pair_qACC"] is not None else 'text_pair_qACC: None')
            m_logger.info(f'weighted_qACC: {result["weighted_qACC"]:.4f}' if result["weighted_qACC"] is not None else 'weighted_qACC: None')
            m_logger.info(f'FP: {result["FP"]}')
            m_logger.info(f'FP_rate: {result["FP_rate"]:.4f}' if result["FP_rate"] is not None else 'FP_rate: None')
            m_logger.info(f'Pct_Diff: {result["Pct_Diff"]:.4f}' if result["Pct_Diff"] is not None else 'Pct_Diff: None')
            m_logger.info(f'总QA数: {result["total_qa"]}, vid_pair总数: {result["vid_pair_total"]}, text_pair总数: {result["text_pair_total"]}, 跳过样本: {result["skipped_samples"]}')
            
            # 输出各类别的wACC
            if 'category_wacc' in result and result['category_wacc']:
                m_logger.info('各类别wACC统计:')
                for category, stats in result['category_wacc'].items():
                    m_logger.info(f'  {category}: wACC={stats["wACC"]:.4f}, vid_pair_qACC={stats["vid_pair_qACC"]:.4f}, text_pair_qACC={stats["text_pair_qACC"]:.4f}, 样本数={stats["total_qa"]}')
            
            m_logger.info('-' * 40)
            
        except Exception as e:
            m_logger.error(f'处理文件 {json_file} 时出错: {str(e)}')
    
    # 处理R*_binary_qa.json文件
    for json_file in sorted(r_files):
        file_path = os.path.join(base_path, json_file)
        r_logger.info(f'开始评测文件: {json_file}')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = evaluate_clipqa_binary_qa(data, r_logger)
            model_name = json_file.replace('_R_binary_qa.json', '')
            result['model'] = model_name
            results['r_files'].append(result)
            
            r_logger.info(f'文件: {json_file} 评测完成')
            r_logger.info(f'vid_pair_qACC: {result["vid_pair_qACC"]:.4f}' if result["vid_pair_qACC"] is not None else 'vid_pair_qACC: None')
            r_logger.info(f'text_pair_qACC: {result["text_pair_qACC"]:.4f}' if result["text_pair_qACC"] is not None else 'text_pair_qACC: None')
            r_logger.info(f'weighted_qACC: {result["weighted_qACC"]:.4f}' if result["weighted_qACC"] is not None else 'weighted_qACC: None')
            r_logger.info(f'FP: {result["FP"]}')
            r_logger.info(f'FP_rate: {result["FP_rate"]:.4f}' if result["FP_rate"] is not None else 'FP_rate: None')
            r_logger.info(f'Pct_Diff: {result["Pct_Diff"]:.4f}' if result["Pct_Diff"] is not None else 'Pct_Diff: None')
            r_logger.info(f'总QA数: {result["total_qa"]}, vid_pair总数: {result["vid_pair_total"]}, text_pair总数: {result["text_pair_total"]}, 跳过样本: {result["skipped_samples"]}')
            
            # 输出各类别的wACC
            if 'category_wacc' in result and result['category_wacc']:
                r_logger.info('各类别wACC统计:')
                for category, stats in result['category_wacc'].items():
                    r_logger.info(f'  {category}: wACC={stats["wACC"]:.4f}, vid_pair_qACC={stats["vid_pair_qACC"]:.4f}, text_pair_qACC={stats["text_pair_qACC"]:.4f}, 样本数={stats["total_qa"]}')
            
            r_logger.info('-' * 40)
            
        except Exception as e:
            r_logger.error(f'处理文件 {json_file} 时出错: {str(e)}')
    
    # 保存M_binary_qa结果
    if results['m_files']:
        with open(m_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'vid_pair_qACC', 'text_pair_qACC', 'weighted_qACC', 'FP', 'FP_rate', 'Pct_Diff', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'Skipped_Samples'])
            
            # 按自定义模型顺序排序
            sorted_results = sorted(results['m_files'], key=lambda x: get_model_order(x['model']))
            for result in sorted_results:
                writer.writerow([
                    result['model'],
                    'M_Files',
                    f"{result['vid_pair_qACC']:.4f}" if result['vid_pair_qACC'] is not None else "None",
                    f"{result['text_pair_qACC']:.4f}" if result['text_pair_qACC'] is not None else "None",
                    f"{result['weighted_qACC']:.4f}" if result['weighted_qACC'] is not None else "None",
                    result['FP'],
                    f"{result['FP_rate']:.4f}" if result['FP_rate'] is not None else "None",
                    f"{result['Pct_Diff']:.4f}" if result['Pct_Diff'] is not None else "None",
                    result['total_qa'],
                    result['vid_pair_total'],
                    result['text_pair_total'],
                    result['skipped_samples']
                ])

    # 保存R_binary_qa结果
    if results['r_files']:
        with open(r_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'vid_pair_qACC', 'text_pair_qACC', 'weighted_qACC', 'FP', 'FP_rate', 'Pct_Diff', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'Skipped_Samples'])
            
            # 按自定义模型顺序排序
            sorted_results = sorted(results['r_files'], key=lambda x: get_model_order(x['model']))
            for result in sorted_results:
                writer.writerow([
                    result['model'],
                    'R_Files',
                    f"{result['vid_pair_qACC']:.4f}" if result['vid_pair_qACC'] is not None else "None",
                    f"{result['text_pair_qACC']:.4f}" if result['text_pair_qACC'] is not None else "None",
                    f"{result['weighted_qACC']:.4f}" if result['weighted_qACC'] is not None else "None",
                    result['FP'],
                    f"{result['FP_rate']:.4f}" if result['FP_rate'] is not None else "None",
                    f"{result['Pct_Diff']:.4f}" if result['Pct_Diff'] is not None else "None",
                    result['total_qa'],
                    result['vid_pair_total'],
                    result['text_pair_total'],
                    result['skipped_samples']
                ])

    # 保存类别统计结果
    # 保存M_binary_qa的类别统计
    if results['m_files']:
        m_category_csv_file = os.path.join(m_output_dir, f'clipqa_category_stats_{timestamp}.csv')
        with open(m_category_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'Category', 'wACC', 'vid_pair_qACC', 'text_pair_qACC', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'FP', 'FP_rate'])
            
            for result in results['m_files']:
                if 'category_wacc' in result and result['category_wacc']:
                    for category, stats in result['category_wacc'].items():
                        writer.writerow([
                            result['model'],
                            'M_Files',
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

    # 保存R_binary_qa的类别统计
    if results['r_files']:
        r_category_csv_file = os.path.join(r_output_dir, f'clipqa_category_stats_{timestamp}.csv')
        with open(r_category_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'Category', 'wACC', 'vid_pair_qACC', 'text_pair_qACC', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'FP', 'FP_rate'])
            
            for result in results['r_files']:
                if 'category_wacc' in result and result['category_wacc']:
                    for category, stats in result['category_wacc'].items():
                        writer.writerow([
                            result['model'],
                            'R_Files',
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

    # 保存8个特定类别的统计结果
    # 保存M_binary_qa的特定类别统计
    if results['m_files']:
        m_specific_category_csv_file = os.path.join(m_output_dir, f'clipqa_specific_category_stats_{timestamp}.csv')
        with open(m_specific_category_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'Category', 'wACC', 'vid_pair_qACC', 'text_pair_qACC', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'FP', 'FP_rate'])
            
            for result in results['m_files']:
                if 'specific_category_wacc' in result and result['specific_category_wacc']:
                    for category, stats in result['specific_category_wacc'].items():
                        writer.writerow([
                            result['model'],
                            'M_Files',
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

    # 保存R_binary_qa的特定类别统计
    if results['r_files']:
        r_specific_category_csv_file = os.path.join(r_output_dir, f'clipqa_specific_category_stats_{timestamp}.csv')
        with open(r_specific_category_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Type', 'Category', 'wACC', 'vid_pair_qACC', 'text_pair_qACC', 'Total_QA', 'vid_pair_total', 'text_pair_total', 'FP', 'FP_rate'])
            
            for result in results['r_files']:
                if 'specific_category_wacc' in result and result['specific_category_wacc']:
                    for category, stats in result['specific_category_wacc'].items():
                        writer.writerow([
                            result['model'],
                            'R_Files',
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
    print("ClipQA 二分类评测结果摘要 (最终版)")
    print("=" * 80)
    
    # 合并所有结果并按自定义模型顺序排序
    all_results = results['m_files'] + results['r_files']
    all_results.sort(key=lambda x: get_model_order(x['model']))
    
    print(f"{'排名':<4} {'模型':<20} {'类型':<10} {'vid_pair_qACC':<12} {'text_pair_qACC':<12} {'weighted_qACC':<12} {'FP_rate':<8}")
    print("-" * 100)
    
    for i, result in enumerate(all_results, 1):
        model = result['model']
        file_type = 'M_Files' if result in results['m_files'] else 'R_Files'
        vid_qacc = f"{result['vid_pair_qACC']:.4f}" if result['vid_pair_qACC'] is not None else "None"
        text_qacc = f"{result['text_pair_qACC']:.4f}" if result['text_pair_qACC'] is not None else "None"
        weighted_qacc = f"{result['weighted_qACC']:.4f}" if result['weighted_qACC'] is not None else "None"
        fp_rate = f"{result['FP_rate']:.4f}" if result['FP_rate'] is not None else "None"
        
        print(f"{i:<4} {model:<20} {file_type:<10} {vid_qacc:<12} {text_qacc:<12} {weighted_qacc:<12} {fp_rate:<8}")

    print(f"\nM_binary_qa结果已保存到:")
    print(f"  日志文件: {m_log_file}")
    print(f"  CSV文件: {m_csv_file}")
    if results['m_files']:
        print(f"  类别统计CSV文件: {m_category_csv_file}")
        print(f"  8个特定类别统计CSV文件: {m_specific_category_csv_file}")
    
    print(f"\nR_binary_qa结果已保存到:")
    print(f"  日志文件: {r_log_file}")
    print(f"  CSV文件: {r_csv_file}")
    if results['r_files']:
        print(f"  类别统计CSV文件: {r_category_csv_file}")
        print(f"  8个特定类别统计CSV文件: {r_specific_category_csv_file}")

if __name__ == "__main__":
    main()
