import os
import time
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
import json
from glob import glob
from openai import OpenAI
import base64
import io
import concurrent.futures
from threading import Lock

# ========== 配置 ==========

INPUT_DIR = "/hpc2hdd/home/hhuang118/VidHalluc/qa/ClipQA"
VIDEO_FOLDER = "/hpc2hdd/home/hhuang118/VidHalluc/VidPair-Halluc"
OUTPUT_DIR = "/hpc2hdd/home/hhuang118/VidHalluc/qa/ClipQA/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

num_frames = 16

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-02T1Augp34uXctQdOSEobu1tYul1nXsL3Oh34g9Hxn5h6Rbx",
    base_url="https://api.gptplus5.com/v1"
)

# ========== System Prompts ==========

BINARY_SYSTEM_PROMPT = (
    "You are an expert video question answering assistant. "
    "Answer the following question about the video. "
    "You must answer with exactly one word: either 'yes' or 'no'. "
    "Do not include any other text, punctuation, or explanation."
)
MULTIPLE_SYSTEM_PROMPT = (
    "You are an expert video question answering assistant. "
    "You must output only the selected option letter(s) without any spaces, commas, or other characters. "
    "For example: 'A', 'B', 'ABC', 'BAC', etc. "
    "Do not include any other text, punctuation, or explanation."
)
OPEN_SYSTEM_PROMPT = (
    "You are an expert video question answering assistant. "
    "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. "
    "Answer the following question concisely, highlighting any significant events, characters, or objects that appear throughout the frames."
)

# ========== 并发控制 ==========
MAX_WORKERS = 8  # 最大并发数
api_lock = Lock()  # API调用锁
save_lock = Lock()  # 保存锁

# ========== 视频帧拼接 ==========

def create_frame_grid(img_array, interval_width=50):
    n, h, w, c = img_array.shape
    grid_size = int(np.ceil(np.sqrt(n)))

    horizontal_band = np.ones((h, interval_width, c),
                              dtype=img_array.dtype) * 255
    vertical_band = np.ones((interval_width, w + (grid_size - 1)
                            * (w + interval_width), c), dtype=img_array.dtype) * 255

    rows = []
    for i in range(grid_size):
        row_frames = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n:
                frame = img_array[idx]
            else:
                frame = np.ones_like(img_array[0]) * 255
            if j > 0:
                row_frames.append(horizontal_band)
            row_frames.append(frame)
        combined_row = np.concatenate(row_frames, axis=1)
        if i > 0:
            rows.append(vertical_band)
        rows.append(combined_row)

    final_grid = np.concatenate(rows, axis=0)
    return final_grid

def resize_image_grid(image, max_length=1920):
    width, height = image.size
    if max(width, height) > max_length:
        if width > height:
            scale = max_length / width
        else:
            scale = max_length / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = image.resize((new_width, new_height), Image.BILINEAR)
    else:
        img_resized = image
    return img_resized

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, return_msg=False, num_frames=4):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments)
        img_array = vr.get_batch(frame_indices).asnumpy()
        img_grid = create_frame_grid(img_array, 50)
        img_grid = Image.fromarray(img_grid).convert("RGB")
        img_grid = resize_image_grid(img_grid)
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return img_grid, msg
        else:
            return img_grid
    except Exception as e:
        print(f"加载视频失败 {video_path}: {e}")
        return None

# ========== gemini-2.5-pro 推理 ==========

def clean_model_output(response, qa_type):
    """清理模型输出，确保格式正确"""
    if not response or response == "API_ERROR":
        return response
    
    response = response.strip()
    
    if qa_type == "binary":
        # 对于binary QA，只保留yes或no
        response_lower = response.lower()
        if "yes" in response_lower and "no" not in response_lower:
            return "yes"
        elif "no" in response_lower and "yes" not in response_lower:
            return "no"
        else:
            # 如果包含both，返回第一个出现的
            if "yes" in response_lower:
                return "yes"
            elif "no" in response_lower:
                return "no"
            else:
                return response  # 保持原样，让后续处理
    
    elif qa_type == "multiple":
        # 对于multiple QA，只保留字母，去除其他字符
        import re
        letters = re.findall(r'[A-Z]', response.upper())
        if letters:
            return ''.join(letters)
        else:
            return response  # 保持原样
    
    return response

def gpt4o_answer(prompt, img_grid, max_tokens=512, system_prompt=None, max_retries=3, qa_type=None):
    img_byte_arr = io.BytesIO()
    img_grid.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{img_base64}"

    system_content = system_prompt if system_prompt is not None else \
        "You are a helpful assistant for video content understanding."
    
    for attempt in range(max_retries):
        try:
            with api_lock:  # 使用锁确保API调用不冲突
                response = client.chat.completions.create(
                    model="gemini-2.5-pro",
                    messages=[
                        {"role": "system", "content": system_content},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_data_url}}
                            ]
                        }
                    ],
                    max_tokens=max_tokens
                )
            raw_response = response.choices[0].message.content.strip()
            # 清理输出格式
            if qa_type:
                return clean_model_output(raw_response, qa_type)
            return raw_response
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"所有重试失败，返回默认响应")
                return "API_ERROR"

def single_test(vid_path, qs, pre_query_prompt=None, num_frames=16, system_prompt=None, qa_type=None):
    if num_frames != 0:
        vid, msg = load_video(
            vid_path, num_segments=num_frames, return_msg=True)
    else:
        vid, msg = None, 'num_frames is 0, not inputing image'
    
    if vid is None:
        return "VIDEO_LOAD_ERROR"
    
    img_grid = vid
    # 构建完整的问题prompt
    if pre_query_prompt is not None:
        full_prompt = pre_query_prompt + "\n" + qs
    else:
        full_prompt = qs
    
    llm_response = gpt4o_answer(full_prompt, img_grid, max_tokens=512, system_prompt=system_prompt, qa_type=qa_type)
    print(f'### 处理: {os.path.basename(vid_path)}')
    print(f'### 问题: {qs[:100]}...')
    print(f'### 回答: {llm_response}')
    return llm_response

# ========== 并发处理函数 ==========

def process_single_qa(qa_item, video_id_folder, system_prompt, qa_type):
    """处理单个QA项目"""
    file_name = qa_item["file"]  # 使用 "file" 而不是 "file_name"
    video_path = f"{video_id_folder}/{file_name}"
    question = qa_item["question"]
    
    if not os.path.exists(video_path):
        return None
    
    prediction = single_test(
        video_path,
        qs=question,
        pre_query_prompt=None,
        num_frames=num_frames,
        system_prompt=system_prompt,
        qa_type=qa_type
    )
    
    return {
        "file": file_name,  # 使用 "file" 而不是 "file_name"
        "prediction": prediction,
        "qa_type": qa_type
    }


# ========== 数据读取和处理 ==========

def check_video_files_exist(video_id, video_folder, qa_items):
    """检查QA项目对应的视频文件是否存在"""
    existing_files = []
    missing_files = []
    
    # 从video_id提取基础文件夹名
    # 例如: static_relation_0040_001 -> static_relation_0040
    base_folder = video_id.split('_')[:-1]  # 去掉最后一部分
    base_folder = '_'.join(base_folder)
    
    for qa in qa_items:
        file_name = qa["file"]  # 使用 "file" 而不是 "file_name"
        video_path = f"{video_folder}/{base_folder}/{file_name}"
        if os.path.exists(video_path):
            existing_files.append(qa)
        else:
            missing_files.append(file_name)
    
    if missing_files:
        print(f"[警告] {video_id}: 以下文件不存在: {missing_files}")
        print(f"[调试] 查找路径: {video_folder}/{base_folder}/")
        # 列出实际存在的文件
        actual_folder = f"{video_folder}/{base_folder}"
        if os.path.exists(actual_folder):
            actual_files = os.listdir(actual_folder)
            print(f"[调试] 实际存在的文件: {actual_files[:10]}...")  # 只显示前10个
    
    return existing_files

def check_single_qa_type_pairing(video_id, video_folder, qa_items, qa_type):
    """检查单个QA类型的配对情况"""
    base_folder = video_id.split('_')[:-1]  # 去掉最后一部分
    base_folder = '_'.join(base_folder)
    video_id_folder = f"{video_folder}/{base_folder}"
    
    if not os.path.exists(video_id_folder):
        print(f"[跳过] {video_id}: {qa_type} - 视频文件夹不存在")
        return False, []
    
    if not qa_items:
        return False, []
    
    existing_items = []
    missing_count = 0
    
    for qa in qa_items:
        file_name = qa["file"]
        video_path = f"{video_id_folder}/{file_name}"
        if os.path.exists(video_path):
            existing_items.append(qa)
        else:
            missing_count += 1
    
    if missing_count > 0:
        print(f"[跳过] {video_id}: {qa_type} 中有 {missing_count}/{len(qa_items)} 个文件不存在，不配对")
        return False, []
    
    print(f"[通过] {video_id}: {qa_type} 配对正常，{len(existing_items)} 个样本")
    return True, existing_items

def check_qa_pairing(video_id, video_folder, entry):
    """检查所有QA类型的配对情况，返回可处理的QA类型"""
    base_folder = video_id.split('_')[:-1]  # 去掉最后一部分
    base_folder = '_'.join(base_folder)
    video_id_folder = f"{video_folder}/{base_folder}"
    
    if not os.path.exists(video_id_folder):
        print(f"[跳过] {video_id}: 视频文件夹不存在")
        return {}
    
    processable_qa = {}
    
    # 检查所有可能的binary_qa类型
    for i in range(3):  # binary_qa_0, binary_qa_1, binary_qa_2
        qa_type = f"binary_qa_{i}"
        qa_items = entry.get(qa_type, [])
        if qa_items:
            is_paired, existing_items = check_single_qa_type_pairing(video_id, video_folder, qa_items, qa_type)
            if is_paired:
                processable_qa[qa_type] = existing_items
    
    # 检查multiple_qa
    multiple_qa_items = entry.get("multiple_qa", [])
    if multiple_qa_items:
        is_paired, existing_items = check_single_qa_type_pairing(video_id, video_folder, multiple_qa_items, "multiple_qa")
        if is_paired:
            processable_qa["multiple_qa"] = existing_items
    
    # 检查open_qa
    open_qa_items = entry.get("open_qa", [])
    if open_qa_items:
        is_paired, existing_items = check_single_qa_type_pairing(video_id, video_folder, open_qa_items, "open_qa")
        if is_paired:
            processable_qa["open_qa"] = existing_items
    
    return processable_qa

def process_clipqa_data(data, video_folder, output_path, model_name="gemini-2.5-pro", max_items=None):
    """处理ClipQA数据"""
    print(f"=== 开始处理ClipQA数据 ===")
    print(f"数据条目数: {len(data)}")
    print(f"视频文件夹: {video_folder}")
    print(f"输出路径: {output_path}")
    print(f"并发数: {MAX_WORKERS}")
    
    processed_count = 0
    valid_entries = []  # 只保存有效的条目
    
    # 如果指定了最大处理数量，只处理前N个
    if max_items:
        data = data[:max_items]
        print(f"限制处理数量: {max_items}")
    
    for i, entry in enumerate(data):
        print(f"\n处理第 {i+1}/{len(data)} 个条目...")
        video_id = entry["id"]
        
        # 检查所有QA类型的配对情况
        processable_qa = check_qa_pairing(video_id, video_folder, entry)
        
        if not processable_qa:
            print(f"[跳过] {video_id}: 没有可处理的QA类型")
            continue
        
        # 获取视频文件夹路径
        base_folder = video_id.split('_')[:-1]  # 去掉最后一部分
        base_folder = '_'.join(base_folder)
        video_id_folder = f"{video_folder}/{base_folder}"
        
        print(f"处理 {video_id}: 可处理的QA类型: {list(processable_qa.keys())}")
        
        # 处理每个可处理的QA类型
        predicted_dict = {}
        
        for qa_type, qa_items in processable_qa.items():
            print(f"  处理 {qa_type}: {len(qa_items)} 个样本")
            
            # 为每个QA类型创建对应的system prompt
            if qa_type.startswith("binary_qa"):
                system_prompt = BINARY_SYSTEM_PROMPT
                qa_type_for_processing = "binary"
            elif qa_type == "multiple_qa":
                system_prompt = MULTIPLE_SYSTEM_PROMPT
                qa_type_for_processing = "multiple"
            elif qa_type == "open_qa":
                system_prompt = OPEN_SYSTEM_PROMPT
                qa_type_for_processing = "open"
            else:
                continue
            
            # 并发处理当前QA类型的所有项目
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(qa_items))) as executor:
                future_to_qa = {
                    executor.submit(
                        process_single_qa, 
                        qa_item, 
                        video_id_folder, 
                        system_prompt,
                        qa_type_for_processing
                    ): qa_item for qa_item in qa_items
                }
                
                for future in concurrent.futures.as_completed(future_to_qa):
                    qa_item = future_to_qa[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"处理QA项目失败 {qa_item['file']}: {e}")
            
            # 移除qa_type和system_prompt字段
            for result in results:
                if "qa_type" in result:
                    del result["qa_type"]
                if "system_prompt" in result:
                    del result["system_prompt"]
            
            if results:
                predicted_dict[qa_type] = results
                print(f"  {qa_type} 处理完成: {len(results)} 个结果")
        
        # 只有当有预测结果时才添加到有效条目中
        if predicted_dict:
            entry["predicted"] = predicted_dict
            valid_entries.append(entry)
            processed_count += 1
            print(f"[处理完成] {video_id} - 已处理 {processed_count} 个条目")
        else:
            print(f"[跳过] {video_id}: 没有生成任何预测结果")
        
        # 每处理5次就保存一次（优化后更频繁保存）
        if processed_count % 5 == 0:
            print(f"[保存中间结果] 已处理 {processed_count} 个条目，正在保存...")
            with save_lock:
                with open(output_path, "w") as f:
                    json.dump(valid_entries, f, indent=4, ensure_ascii=False)
            print(f"[✓] 中间结果已保存到：{output_path}")

    # 最终保存
    with save_lock:
        with open(output_path, "w") as f:
            json.dump(valid_entries, f, indent=4, ensure_ascii=False)
    print(f"[✓] 最终结果已保存到：{output_path}")
    print(f"[统计] 原始数据: {len(data)} 个条目，有效处理: {len(valid_entries)} 个条目")

def test_api_connection():
    """测试API连接"""
    print("测试API连接...")
    try:
        # 创建一个简单的测试图像
        test_img = Image.new('RGB', (100, 100), color='red')
        test_response = gpt4o_answer("What color is this image?", test_img, max_tokens=10)
        print(f"API测试成功，响应: {test_response}")
        return True
    except Exception as e:
        print(f"API测试失败: {str(e)}")
        return False

def main():
    print("=== 开始优化版ClipQA处理（真正并发） ===")
    
    # 首先测试API连接
    print("测试API连接...")
    if not test_api_connection():
        print("API连接失败，请检查网络和API配置")
        return
    
    print("API连接测试通过")
    
    # ========== 第一阶段：处理 Binary QA 文件 ==========
    print("\n" + "="*50)
    print("第一阶段：处理 Binary QA 文件")
    print("="*50)
    
    # 处理 R_binary_qa.json
    r_binary_qa_path = os.path.join(INPUT_DIR, "R_binary_qa.json")
    print(f"\n检查文件: {r_binary_qa_path}")
    if os.path.exists(r_binary_qa_path):
        print("开始处理 R_binary_qa.json...")
        with open(r_binary_qa_path, 'r') as f:
            r_binary_data = json.load(f)
        print(f"加载了 {len(r_binary_data)} 个条目")
        
        output_path = os.path.join(OUTPUT_DIR, "gemini-2.5-pro_R_binary_qa.json")
        print(f"输出路径: {output_path}")
        process_clipqa_data(r_binary_data, VIDEO_FOLDER, output_path, "gemini-2.5-pro")
        print("R_binary_qa.json 处理完成！")
    else:
        print(f"未找到文件：{r_binary_qa_path}")

    # 处理 M_binary_qa.json
    m_binary_qa_path = os.path.join(INPUT_DIR, "M_binary_qa.json")
    print(f"\n检查文件: {m_binary_qa_path}")
    if os.path.exists(m_binary_qa_path):
        print("开始处理 M_binary_qa.json...")
        with open(m_binary_qa_path, 'r') as f:
            m_binary_data = json.load(f)
        print(f"加载了 {len(m_binary_data)} 个条目")
        
        output_path = os.path.join(OUTPUT_DIR, "gemini-2.5-pro_M_binary_qa.json")
        print(f"输出路径: {output_path}")
        process_clipqa_data(m_binary_data, VIDEO_FOLDER, output_path, "gemini-2.5-pro")
        print("M_binary_qa.json 处理完成！")
    else:
        print(f"未找到文件：{m_binary_qa_path}")
    
    # ========== 第二阶段：处理 Multiple QA 文件 ==========
    print("\n" + "="*50)
    print("第二阶段：处理 Multiple QA 文件")
    print("="*50)
    
    # 处理 R_M_multiple_qa.json
    r_m_multiple_qa_path = os.path.join(INPUT_DIR, "R_M_multiple_qa.json")
    print(f"\n检查文件: {r_m_multiple_qa_path}")
    if os.path.exists(r_m_multiple_qa_path):
        print("开始处理 R_M_multiple_qa.json...")
        with open(r_m_multiple_qa_path, 'r') as f:
            r_m_multiple_data = json.load(f)
        print(f"加载了 {len(r_m_multiple_data)} 个条目")
        
        output_path = os.path.join(OUTPUT_DIR, "gemini-2.5-pro_R_M_multiple_qa.json")
        print(f"输出路径: {output_path}")
        process_clipqa_data(r_m_multiple_data, VIDEO_FOLDER, output_path, "gemini-2.5-pro")
        print("R_M_multiple_qa.json 处理完成！")
    else:
        print(f"未找到文件：{r_m_multiple_qa_path}")
    
    print("\n" + "="*50)
    print("=== 所有处理完成 ===")
    print("="*50)

if __name__ == "__main__":
    main()
