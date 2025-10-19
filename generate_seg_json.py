#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path

def find_image_files(processed_dir, id_name):
    """
    查找指定ID目录下的所有图片文件
    返回: {
        'main': 主图片路径,
        'values_1': values_1图片路径,
        'values_2': values_2图片路径
    }
    """
    id_dir = os.path.join(processed_dir, id_name)
    if not os.path.exists(id_dir):
        return None
    
    image_files = {}
    
    # 查找主图片 (segment_1_*.png 但不包含 values_)
    main_pattern = os.path.join(id_dir, f"{id_name}_segment_1_*.png")
    main_files = glob.glob(main_pattern)
    main_files = [f for f in main_files if 'values_' not in f]
    
    if main_files:
        image_files['main'] = main_files[0]  # 取第一个匹配的
    
    # 查找values_1图片
    values1_pattern = os.path.join(id_dir, f"{id_name}_segment_1_values_1.png")
    if os.path.exists(values1_pattern):
        image_files['values_1'] = values1_pattern
    
    # 查找values_2图片
    values2_pattern = os.path.join(id_dir, f"{id_name}_segment_1_values_2.png")
    if os.path.exists(values2_pattern):
        image_files['values_2'] = values2_pattern
    
    return image_files

def fill_segment_text(segment, value, value_idx):
    """
    用候选值填充segment文本
    处理sequence类型的特殊格式 {values[0]}, {values[1]}, {values[2]}
    """
    if isinstance(value, list) and len(value) == 3:
        # sequence类型：value是长度为3的列表，需要替换 {values[0]}, {values[1]}, {values[2]}
        filled_text = segment
        for i in range(len(value)):
            filled_text = filled_text.replace(f'{{values[{i}]}}', value[i])
        return filled_text
    else:
        # 普通类型：直接替换 {values}
        return segment.replace('{values}', str(value))

def get_image_path_for_segment(processed_dir, id_name, value_key, value_string, seg_idx, value_idx, is_sequence=False):
    """
    根据segment索引生成对应的图像路径
    第一个segment使用segment_1的图片，后续segment使用前一个segment视频的尾帧
    """
    if seg_idx == 0:
        # 第一个segment使用segment_1的图片
        if is_sequence:
            # sequence类型：使用*_value_1.png图片
            value1_pattern = os.path.join(processed_dir, id_name, f"{id_name}_*_value_1.png")
            value1_files = glob.glob(value1_pattern)
            if value1_files:
                return value1_files[0]
        elif value_key == "main":
            # 查找主图片
            main_pattern = os.path.join(processed_dir, id_name, f"{id_name}_segment_1_*.png")
            main_files = glob.glob(main_pattern)
            main_files = [f for f in main_files if 'values_' not in f]
            if main_files:
                return main_files[0]
        else:
            # 查找values图片
            values_pattern = os.path.join(processed_dir, id_name, f"{id_name}_segment_1_{value_key}.png")
            if os.path.exists(values_pattern):
                return values_pattern
    else:
        # 后续segment使用前一个segment视频的尾帧
        if is_sequence:
            # sequence类型：使用 {id}_segment_{segment_index+1}_value_1.png 格式
            end_frame_path = os.path.join(
                processed_dir, id_name, 
                f"{id_name}_segment_{seg_idx + 1}_value_1.png"
            )
            return end_frame_path
        else:
            # 普通类型：使用 {id}_segment_{segment_index+1}_values_{value_index}.png 格式
            end_frame_path = os.path.join(
                processed_dir, id_name, 
                f"{id_name}_segment_{seg_idx + 1}_values_{value_idx}.png"
            )
            return end_frame_path
    
    return None

def generate_segment_data(story_data, processed_dir):
    """
    为每个ID生成视频片段数据
    """
    seg_data = []
    
    for item in story_data:
        id_name = item['id']
        segments = item['segments']
        values = item['values']
        
        print(f"处理 {id_name}...")
        
        # 检查是否是sequence类型
        is_sequence = isinstance(values[0], list) if values else False
        
        if is_sequence:
            # sequence类型：只处理第一组values
            first_values = values[0]  # 只取第一组列表
            print(f"  sequence类型，使用第一组values: {first_values}")
            
            # 为每个segment生成数据（使用第一组的三个值）
            for seg_idx, segment in enumerate(segments):
                if seg_idx < len(first_values):
                    value_string = first_values[seg_idx]  # 使用对应位置的单个值
                    filled_text = fill_segment_text(segment, first_values, 0)  # 传入完整列表用于填充
                    
                    # 获取图像路径
                    image_path = get_image_path_for_segment(processed_dir, id_name, "main", first_values, seg_idx, 0, is_sequence=True)
                    
                    if image_path:
                        segment_data = {
                            "id": id_name,
                            "value_index": 0,
                            "value": value_string,  # 存储单个字符串值
                            "segment_index": seg_idx,
                            "text": filled_text,
                            "image_path": image_path,
                            "video_output_path": f"videos/{id_name}_segment_{seg_idx + 1}_values_0.mp4"
                        }
                        seg_data.append(segment_data)
                        print(f"    生成segment {seg_idx}: {value_string}")
                        print(f"      图像路径: {os.path.basename(image_path)}")
                    else:
                        print(f"    警告: 未找到segment {seg_idx}的图像")
        else:
            # 普通类型：处理所有候选值，确保所有value_index和segment_index的组合都存在
            for value_idx, value in enumerate(values):
                value_key = f"values_{value_idx}" if value_idx > 0 else "main"
                
                # 生成每个segment的数据
                for seg_idx, segment in enumerate(segments):
                    filled_text = fill_segment_text(segment, value, value_idx)
                    
                    # 获取图像路径
                    image_path = get_image_path_for_segment(processed_dir, id_name, value_key, value, seg_idx, value_idx, is_sequence=False)
                    
                    if image_path:
                        segment_data = {
                            "id": id_name,
                            "value_index": value_idx,
                            "value": value,
                            "segment_index": seg_idx,
                            "text": filled_text,
                            "image_path": image_path,
                            "video_output_path": f"videos/{id_name}_segment_{seg_idx + 1}_values_{value_idx}.mp4"
                        }
                        seg_data.append(segment_data)
                        print(f"    生成 {value_key} segment {seg_idx}: {str(value)[:30]}...")
                        print(f"      图像路径: {os.path.basename(image_path)}")
                    else:
                        print(f"    警告: 未找到 {value_key} segment {seg_idx}的图像")
        
        print(f"  完成 {id_name}")
    
    return seg_data

def main():
    # 设置工作目录
    work_dir = "/hpc2hdd/home/hhuang118/VidHalluc"
    os.chdir(work_dir)
    
    print("开始生成seg.json...")
    
    # 读取story.json
    with open("story.json", "r", encoding="utf-8") as f:
        story_data = json.load(f)
    
    print(f"读取了 {len(story_data)} 个story条目")
    
    # 生成segment数据
    processed_dir = "processed_data"
    seg_data = generate_segment_data(story_data, processed_dir)
    
    # 保存seg.json
    with open("seg.json", "w", encoding="utf-8") as f:
        json.dump(seg_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完成！")
    print(f"总共生成了 {len(seg_data)} 个视频片段数据")
    print(f"数据已保存到 seg.json")
    
    # 显示统计信息
    print("\n统计信息:")
    id_count = {}
    value_count = {}
    segment_count = {}
    
    for item in seg_data:
        id_name = item['id']
        value_idx = item['value_index']
        seg_idx = item['segment_index']
        
        id_count[id_name] = id_count.get(id_name, 0) + 1
        value_count[value_idx] = value_count.get(value_idx, 0) + 1
        segment_count[seg_idx] = segment_count.get(seg_idx, 0) + 1
    
    print(f"  涉及ID数量: {len(id_count)}")
    print(f"  每个候选值的片段数: {dict(sorted(value_count.items()))}")
    print(f"  每个segment的片段数: {dict(sorted(segment_count.items()))}")
    
    # 显示前几个示例
    print("\n前5个片段示例:")
    for i, item in enumerate(seg_data[:5]):
        print(f"  {i+1}. ID: {item['id']}, 候选值: {item['value_index']}, Segment: {item['segment_index']}")
        print(f"     文本: {item['text'][:50]}...")
        print(f"     图片: {os.path.basename(item['image_path'])}")
        print(f"     视频: {os.path.basename(item['video_output_path'])}")
        print()

if __name__ == "__main__":
    main()
