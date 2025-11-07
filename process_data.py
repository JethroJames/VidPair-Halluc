#!/usr/bin/env python3
import os
import json
import shutil
import glob
from pathlib import Path

def extract_sort_key(id_name):
    """提取排序键，用于按类别和编号排序"""
    parts = id_name.split('_')
    if len(parts) >= 3:
        category = '_'.join(parts[:-1])  # 除了最后一部分（数字）之外的所有部分
        number = int(parts[-1])  # 最后一部分是数字
    else:
        category = parts[0]
        number = int(parts[1]) if len(parts) > 1 else 0
    
    return (category, number)

def find_json_file(category):
    """查找对应类别的JSON文件"""
    possible_names = [
        f"{category}.json",
        f"{category}_edit.json",
        f"{category}_data.json"
    ]
    
    # 特殊处理action类别，因为文件名是actioin.json
    if category == "action":
        possible_names = ["actioin.json"] + possible_names
    
    for name in possible_names:
        json_path = f"raw_data/{category}/{name}"
        if os.path.exists(json_path):
            return json_path
    
    return None

def main():
    # 设置工作目录
    work_dir = "Your Path Here"
    os.chdir(work_dir)
    
    print("开始数据处理任务...")
    
    # 1. 提取VidPair-Halluc目录中的ID并排序
    print("步骤1: 提取VidPair-Halluc目录中的ID并排序...")
    vidpair_dir = "VidPair-Halluc"
    id_list = []
    
    for item in os.listdir(vidpair_dir):
        if os.path.isdir(os.path.join(vidpair_dir, item)) and '_' in item:
            id_list.append(item)
    
    # 按照类别和编号排序
    id_list.sort(key=extract_sort_key)
    
    # 保存ID到文件
    with open("id.txt", "w") as f:
        for id_name in id_list:
            f.write(f"{id_name}\n")
    
    print(f"已提取的ID数量: {len(id_list)}")
    print("ID列表已保存到 id.txt")
    print("前10个ID:")
    for i, id_name in enumerate(id_list[:10]):
        print(f"  {i+1}. {id_name}")
    
    # 2. 创建processed_data目录
    processed_dir = "processed_data"
    os.makedirs(processed_dir, exist_ok=True)
    
    # 3. 存储JSON数据 - 使用列表来保持顺序
    story_data = []  # 使用列表，按照处理顺序存储
    
    # 4. 处理每个ID
    print("\n步骤2: 开始处理每个ID...")
    
    for i, id_name in enumerate(id_list, 1):
        print(f"处理 {i}/{len(id_list)}: {id_name}")
        
        # 提取类别：找到最后一个下划线前的部分
        parts = id_name.split('_')
        if len(parts) >= 3:
            category = '_'.join(parts[:-1])  # 除了最后一部分（数字）之外的所有部分
        else:
            category = parts[0]  # 如果只有两部分，取第一部分
        
        print(f"  类别: {category}")
        
        # 创建目标目录
        target_dir = os.path.join(processed_dir, id_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 查找并复制图片文件
        images_found = 0
        
        if category == "sequence":
            # sequence类型：只复制*_value1.png图片
            value1_pattern = f"raw_data/{category}/{id_name}_*_value_1.png"
            value1_images = glob.glob(value1_pattern)
            
            if value1_images:
                value1_image = value1_images[0]
                shutil.copy2(value1_image, target_dir)
                print(f"  复制sequence图片: {os.path.basename(value1_image)}")
                images_found += 1
            else:
                print(f"  警告: 未找到sequence图片 for {id_name}")
        else:
            # 其他类型：按原逻辑处理
            # 1. 在raw_data/{category}目录中查找主图片
            main_pattern = f"raw_data/{category}/{id_name}_segment_1_*.png"
            main_images = glob.glob(main_pattern)
            
            if main_images:
                main_image = main_images[0]  # 取第一个匹配的
                shutil.copy2(main_image, target_dir)
                print(f"  复制主图片: {os.path.basename(main_image)}")
                images_found += 1
            else:
                print(f"  警告: 未找到主图片 for {id_name}")
            
            # 2. 在raw_data/{category}/results目录中查找values图片
            results_dir = f"raw_data/{category}/results"
            if os.path.exists(results_dir):
                # 查找values_1图片
                values1_pattern = f"{results_dir}/{id_name}_*_values_1.png"
                values1_images = glob.glob(values1_pattern)
                
                if values1_images:
                    values1_image = values1_images[0]
                    shutil.copy2(values1_image, target_dir)
                    print(f"  复制values1图片: {os.path.basename(values1_image)}")
                    images_found += 1
                
                # 查找values_2图片
                values2_pattern = f"{results_dir}/{id_name}_*_values_2.png"
                values2_images = glob.glob(values2_pattern)
                
                if values2_images:
                    values2_image = values2_images[0]
                    shutil.copy2(values2_image, target_dir)
                    print(f"  复制values2图片: {os.path.basename(values2_image)}")
                    images_found += 1
            else:
                print(f"  提示: results目录不存在: {results_dir}")
        
        print(f"  找到图片数量: {images_found}")
        
        # 3. 从JSON文件中提取对应的数据
        json_file = find_json_file(category)
        found_data = None
        
        if json_file:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 查找对应的ID数据
                if isinstance(json_data, dict):
                    # 如果JSON是字典格式，查找对应的键
                    for key, value in json_data.items():
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and item.get('id') == id_name:
                                    found_data = item
                                    break
                        if found_data:
                            break
                elif isinstance(json_data, list):
                    # 如果JSON是列表格式
                    for item in json_data:
                        if isinstance(item, dict) and item.get('id') == id_name:
                            found_data = item
                            break
                
                if found_data:
                    print(f"  提取JSON数据成功")
                else:
                    print(f"  警告: 在 {json_file} 中未找到ID {id_name} 的数据")
            except Exception as e:
                print(f"  错误: 读取JSON文件失败 {json_file}: {e}")
        else:
            print(f"  警告: 未找到JSON文件 for {category}")
        
        # 4. 将数据添加到story_data中（无论是否找到JSON数据都添加，保持顺序）
        if found_data:
            story_data.append(found_data)
        else:
            # 如果没有找到JSON数据，创建一个占位符条目
            placeholder = {
                "id": id_name,
                "segments": ["数据未找到"],
                "values": ["数据未找到"]
            }
            story_data.append(placeholder)
            print(f"  添加占位符条目 for {id_name}")
        
        print(f"  完成处理 {id_name}")
        print("")
    
    # 5. 保存story.json（已经按照id_list的顺序排列）
    print("步骤3: 生成story.json...")
    with open("story.json", "w", encoding="utf-8") as f:
        json.dump(story_data, f, ensure_ascii=False, indent=2)
    
    print("")
    print("数据处理完成！")
    print(f"处理了 {len(id_list)} 个ID")
    print("图片已复制到 processed_data/ 目录")
    print(f"JSON数据已保存到 story.json (包含 {len(story_data)} 个条目)")
    print("")
    print("story.json 结构预览:")
    if story_data:
        print(json.dumps(story_data[0], ensure_ascii=False, indent=2))
    else:
        print("[]")
    
    # 6. 显示类别统计信息
    print("\n类别统计:")
    category_count = {}
    for item in story_data:
        if item.get('id') and item['id'] != "数据未找到":
            category = item['id'].split('_')[0] if '_' in item['id'] else item['id']
            if len(item['id'].split('_')) >= 3:
                category = '_'.join(item['id'].split('_')[:-1])
            category_count[category] = category_count.get(category, 0) + 1
    
    for category, count in sorted(category_count.items()):
        print(f"  {category}: {count} 个条目")
    
    # 7. 验证顺序
    print("\n验证ID顺序（前20个）:")
    for i, item in enumerate(story_data[:20]):
        print(f"  {i+1:2d}. {item.get('id', 'Unknown')}")

if __name__ == "__main__":
    main()
