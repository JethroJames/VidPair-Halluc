import os
import json

# 定义路径
video_path = "Your Path Here"
json_path = "Your Path Here"
output_json_path = "Your Path Here"


# 确保输出目录存在
os.makedirs(output_json_path, exist_ok=True)

all_categories_data = {}

# 遍历视频目录中的所有文件
for file_name in os.listdir(video_path):
    if not file_name.endswith('.mp4'):
        continue

    # 提取 action 和 index 信息
    action, action_id, index = file_name.split('.')[0].rsplit('_', 2)
    # import pdb; pdb.set_trace()
    
    # 构造 JSON 文件路径
    json_file_name = f"{action}.json"
    json_file_path = os.path.join(json_path, json_file_name)
    
    # 检查 JSON 文件是否存在
    if not os.path.exists(json_file_path):
        print(f"JSON file for {action} not found, skipping...")
        continue
    
    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 找到匹配的 action
    action_data = next((item for item in data[action] if item['id'] == f"{action}_{action_id}"), None)
    if not action_data:
        print(f"No matching action found for {action}_{action_id} in {json_file_name}, skipping...")
        continue
    
    # 解析替换 index
    replacement_indices = [int(i) for i in index]
    
    # 根据 index 替换 segments
    modified_segments = []
    for i, segment in enumerate(action_data['segments']):
        value_index = replacement_indices[i]
        value_to_insert = action_data['values'][value_index]
        modified_segment = segment.replace("{values}", value_to_insert)
        modified_segments.append(modified_segment)
    
    # 准备新的 JSON 数据
    modified_action_data = {
        "id": f"{action}_{action_id}_{index}",
        "segments": modified_segments
    }

    if action not in all_categories_data:
        all_categories_data[action] = []
    all_categories_data[action].append(modified_action_data)
    
    # 写入新的 JSON 文件
    for action, actions in all_categories_data.items():
        output_file_path = os.path.join(output_json_path, f"{action}.json")
        with open(output_file_path, 'w') as f:
            json.dump({"action": actions}, f, indent=2)
    
    print(f"Processed and saved: {output_file_path}")

print("All files processed.")
