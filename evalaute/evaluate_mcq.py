import json
import os

def extract_answer_letter(ans):
    """
    提取答案的首个大写字母（A/B/C/D等），若不存在则返回原答案
    """
    if not isinstance(ans, str):
        return ans
    for c in ans.strip():
        if c.isalpha():
            return c.upper()
    return ans.strip()

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

def extract_answer_sequence(ans):
    """
    处理clip-level答案，返回去除空格、转大写的字符串
    """
    if not isinstance(ans, str):
        return ans
    return ''.join(ans.strip().upper().split())

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
    binary_qa_keys = [k for k in qa_keys if k.startswith('binary_qa_') and k[10:].isdigit()]
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

def evaluate_mcq_f1_and_qacc_from_json(json_path):
    """
    video-level（单选）评测
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    TP = FP = 0
    total = 0
    total_pairs = 0
    correct_pairs = 0
    skipped_samples = 0
    
    # 按类别统计
    category_stats = {}

    for data in data_list:
        # 提取类别信息
        category = extract_category_from_id(data.get('id', ''))
        if category not in category_stats:
            category_stats[category] = {
                'TP': 0,
                'FP': 0,
                'total': 0,
                'total_pairs': 0,
                'correct_pairs': 0
            }
        
        if "predicted" not in data:
            skipped_samples += 1
            continue  # 跳过没有推理结果的样本

        # 检查是否有multiple_qa字段
        if "multiple_qa" not in data:
            skipped_samples += 1
            continue

        # 构建GT字典
        gt_dict = {}
        for qa in data["multiple_qa"]:
            if "file_name" in qa and "answer" in qa:
                gt_dict[qa["file_name"]] = extract_answer_letter(qa["answer"])
        
        # 构建预测字典，使用鲁棒的预测值获取
        pred_dict = {}
        for qa in data["predicted"]["multiple_qa"]:
            if "file_name" in qa:
                pred_val = safe_get_prediction(qa)
                if pred_val is not None:
                    pred_dict[qa["file_name"]] = extract_answer_letter(pred_val)

        # 检查是否有成对的视频
        all_files = list(gt_dict.keys())
        if not has_paired_videos_video_level(all_files):
            skipped_samples += 1
            continue

        # 只统计交集
        common_files = set(gt_dict.keys()) & set(pred_dict.keys())

        for file_name in common_files:
            total += 1
            category_stats[category]['total'] += 1
            specific_category_stats[specific_category]['total'] += 1
            if pred_dict[file_name] == gt_dict[file_name]:
                TP += 1
                category_stats[category]['TP'] += 1
                specific_category_stats[specific_category]['TP'] += 1
            else:
                FP += 1
                category_stats[category]['FP'] += 1
                specific_category_stats[specific_category]['FP'] += 1

        # 精确找出 _0.mp4, _1.mp4, _2.mp4
        file_0 = None
        file_1 = None
        file_2 = None
        for f in common_files:
            if f.endswith("_0.mp4"):
                file_0 = f
            elif f.endswith("_1.mp4"):
                file_1 = f
            elif f.endswith("_2.mp4"):
                file_2 = f

        pairs = []
        if file_0 and file_1:
            pairs.append((file_0, file_1))
        if file_0 and file_2:
            pairs.append((file_0, file_2))

        for file_a, file_b in pairs:
            total_pairs += 1
            category_stats[category]['total_pairs'] += 1
            specific_category_stats[specific_category]['total_pairs'] += 1
            if (pred_dict[file_a] == gt_dict[file_a]) and (pred_dict[file_b] == gt_dict[file_b]):
                correct_pairs += 1
                category_stats[category]['correct_pairs'] += 1
                specific_category_stats[specific_category]['correct_pairs'] += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FP) if (TP + FP) > 0 else 0  # recall同precision，因为没有FN
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / total if total > 0 else 0
    qacc = correct_pairs / total_pairs if total_pairs > 0 else None

    # 计算各类别的指标
    category_stats_result = {}
    for category, stats in category_stats.items():
        if stats['total'] > 0:
            cat_precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
            cat_recall = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
            cat_accuracy = stats['TP'] / stats['total'] if stats['total'] > 0 else 0
            cat_qacc = stats['correct_pairs'] / stats['total_pairs'] if stats['total_pairs'] > 0 else None
            
            category_stats_result[category] = {
                'F1': cat_f1,
                'Accuracy': cat_accuracy,
                'qACC': cat_qacc,
                'Total_files': stats['total'],
                'TP': stats['TP'],
                'FP': stats['FP'],
                'Total_pairs': stats['total_pairs'],
                'Correct_pairs': stats['correct_pairs']
            }

    return {
        "F1": f1,
        "Accuracy": accuracy,
        "qACC": qacc,
        "Total_files": total,
        "TP": TP,
        "FP": FP,
        "FN": 0,
        "Total_pairs": total_pairs,
        "Correct_pairs": correct_pairs,
        "skipped_samples": skipped_samples,
        "category_stats": category_stats_result
    }

def evaluate_mcq_f1_and_qacc_clip_level(json_path):
    """
    clip-level（多选序列）评测
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    TP = FP = 0
    total = 0
    total_pairs = 0
    correct_pairs = 0
    skipped_samples = 0
    
    # 按类别统计
    category_stats = {}
    # 按8个特定类别统计
    specific_category_stats = {}

    for data in data_list:
        # 提取类别信息
        category = extract_category_from_id(data.get('id', ''))
        if category not in category_stats:
            category_stats[category] = {
                'TP': 0,
                'FP': 0,
                'total': 0,
                'total_pairs': 0,
                'correct_pairs': 0
            }
        
        # 提取8个特定类别信息
        specific_category = extract_specific_category_from_id(data.get('id', ''))
        if specific_category not in specific_category_stats:
            specific_category_stats[specific_category] = {
                'TP': 0,
                'FP': 0,
                'total': 0,
                'total_pairs': 0,
                'correct_pairs': 0
            }
        
        if "predicted" not in data:
            skipped_samples += 1
            continue  # 跳过没有推理结果的样本

        # 检查是否有multiple_qa字段
        if "multiple_qa" not in data:
            skipped_samples += 1
            continue

        # 构建GT字典
        gt_dict = {}
        for qa in data["multiple_qa"]:
            if "file" in qa and "answer" in qa:
                gt_dict[qa["file"]] = extract_answer_sequence(qa["answer"])
        
        # 构建预测字典，使用鲁棒的预测值获取
        pred_dict = {}
        for qa in data["predicted"]["multiple_qa"]:
            if "file" in qa:
                pred_val = safe_get_prediction(qa)
                if pred_val is not None:
                    pred_dict[qa["file"]] = extract_answer_sequence(pred_val)

        # 检查是否有成对的视频（clip-level逻辑：只要有multiple_qa就认为成对）
        # 对于clip-level的MCQ，我们不需要检查配对，因为每个样本都是独立的

        # 只统计交集
        common_files = set(gt_dict.keys()) & set(pred_dict.keys())

        for file_name in common_files:
            total += 1
            category_stats[category]['total'] += 1
            specific_category_stats[specific_category]['total'] += 1
            if pred_dict[file_name] == gt_dict[file_name]:
                TP += 1
                category_stats[category]['TP'] += 1
                specific_category_stats[specific_category]['TP'] += 1
            else:
                FP += 1
                category_stats[category]['FP'] += 1
                specific_category_stats[specific_category]['FP'] += 1

        # 精确找出 _0.mp4, _1.mp4, _2.mp4
        file_0 = None
        file_1 = None
        file_2 = None
        for f in common_files:
            if f.endswith("_0.mp4"):
                file_0 = f
            elif f.endswith("_1.mp4"):
                file_1 = f
            elif f.endswith("_2.mp4"):
                file_2 = f

        pairs = []
        if file_0 and file_1:
            pairs.append((file_0, file_1))
        if file_0 and file_2:
            pairs.append((file_0, file_2))

        for file_a, file_b in pairs:
            total_pairs += 1
            category_stats[category]['total_pairs'] += 1
            specific_category_stats[specific_category]['total_pairs'] += 1
            if (pred_dict[file_a] == gt_dict[file_a]) and (pred_dict[file_b] == gt_dict[file_b]):
                correct_pairs += 1
                category_stats[category]['correct_pairs'] += 1
                specific_category_stats[specific_category]['correct_pairs'] += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FP) if (TP + FP) > 0 else 0  # recall同precision，因为没有FN
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / total if total > 0 else 0
    qacc = correct_pairs / total_pairs if total_pairs > 0 else None

    # 计算各类别的指标
    category_stats_result = {}
    for category, stats in category_stats.items():
        if stats['total'] > 0:
            cat_precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
            cat_recall = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
            cat_accuracy = stats['TP'] / stats['total'] if stats['total'] > 0 else 0
            cat_qacc = stats['correct_pairs'] / stats['total_pairs'] if stats['total_pairs'] > 0 else None
            
            category_stats_result[category] = {
                'F1': cat_f1,
                'Accuracy': cat_accuracy,
                'qACC': cat_qacc,
                'Total_files': stats['total'],
                'TP': stats['TP'],
                'FP': stats['FP'],
                'Total_pairs': stats['total_pairs'],
                'Correct_pairs': stats['correct_pairs']
            }

    return {
        "F1": f1,
        "Accuracy": accuracy,
        "qACC": qacc,
        "Total_files": total,
        "TP": TP,
        "FP": FP,
        "FN": 0,
        "Total_pairs": total_pairs,
        "Correct_pairs": correct_pairs,
        "skipped_samples": skipped_samples,
        "category_stats": category_stats_result
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

def main():
    # 更新为新的目录结构
    base_video_path = '/hpc2hdd/home/hhuang118/VidHalluc/qa/VideoQA/results'
    base_clip_path = '/hpc2hdd/home/hhuang118/VidHalluc/qa/ClipQA/results'
    
    # video-level的multiple_qa文件列表（目前看起来只有clip-level有multiple_qa文件）
    video_json_path_list = [
        # 目前VideoQA/results目录下没有multiple_qa文件，如果需要可以添加
    ]
    
    # clip-level的multiple_qa文件列表（按模型顺序排列）
    clip_json_path_list = [
        'Video-ChatGPT_R_M_multiple_qa.json',
        'Video-LLaVA_R_M_multiple_qa.json',
        'VideoChat2_R_M_multiple_qa.json',
        'VideoLLaMA2_R_M_multiple_qa.json',
        'Qwen2.5-VL_R_M_multiple_qa.json',
        'ShareGPT4Video_R_M_multiple_qa.json',
        'PLLaVA_R_M_multiple_qa.json',
        'LLaMA-VID_R_M_multiple_qa.json',
        'VILA_R_M_multiple_qa.json',
        'gpt-4o_R_M_multiple_qa.json',
        'gemini-2.5-flash_R_M_multiple_qa.json',
    ]

    print("请选择评测模式：")
    print("1. video-level（单选，答案只含1个字母）")
    print("2. clip-level（多选，答案可能为多个字母序列）")
    # 自动选择clip-level模式
    mode = "2"
    print(f"自动选择模式: {mode}")

    if mode == "1":
        print("\n【video-level】评测结果：")
        if not video_json_path_list:
            print("当前没有video-level的multiple_qa文件可供评测")
        else:
            for json_path in video_json_path_list:
                json_path_full = os.path.join(base_video_path, json_path)
                try:
                    result = evaluate_mcq_f1_and_qacc_from_json(json_path_full)
                    print(f'文件: {json_path_full}')
                    print(result)
                    
                    # 输出各类别的统计
                    if 'category_stats' in result and result['category_stats']:
                        print('各类别统计:')
                        for category, stats in result['category_stats'].items():
                            print(f'  {category}: F1={stats["F1"]:.4f}, Accuracy={stats["Accuracy"]:.4f}, qACC={stats["qACC"]:.4f if stats["qACC"] is not None else "None"}, 样本数={stats["Total_files"]}')
                    
                    print('-' * 40)
                except Exception as e:
                    print(f'文件: {json_path_full} 评测出错: {e}')
                    print('-' * 40)
    elif mode == "2":
        print("\n【clip-level】评测结果：")
        for json_path in clip_json_path_list:
            json_path_full = os.path.join(base_clip_path, json_path)
            try:
                result = evaluate_mcq_f1_and_qacc_clip_level(json_path_full)
                print(f'文件: {json_path_full}')
                print(result)
                
                # 输出各类别的统计
                if 'category_stats' in result and result['category_stats']:
                    print('各类别统计:')
                    for category, stats in result['category_stats'].items():
                        print(f'  {category}: F1={stats["F1"]:.4f}, Accuracy={stats["Accuracy"]:.4f}, qACC={stats["qACC"]:.4f if stats["qACC"] is not None else "None"}, 样本数={stats["Total_files"]}')
                
                print('-' * 40)
            except Exception as e:
                print(f'文件: {json_path_full} 评测出错: {e}')
                print('-' * 40)
    else:
        print("输入有误，请输入1或2。")

if __name__ == "__main__":
    main()
