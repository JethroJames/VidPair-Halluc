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

# ========== 配置 ==========

INPUT_DIR = os.environ.get("VIDPAIR_INPUT_DIR", ".")
VIDEO_FOLDER = os.environ.get("VIDPAIR_VIDEO_FOLDER", "VidPair-Halluc")
OUTPUT_DIR = os.environ.get("VIDPAIR_OUTPUT_DIR", "outputs/clipqa_gpt4o")
os.makedirs(OUTPUT_DIR, exist_ok=True)

num_frames = 16

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "missing-api-key"),
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
)

# ========== System Prompts ==========

BINARY_SYSTEM_PROMPT = (
    "You are an expert video question answering assistant. "
    "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. "
    "Answer the following question about the video. Only answer 'Yes' or 'No'."
)
MULTIPLE_SYSTEM_PROMPT = (
    "You are an expert video question answering assistant. "
    "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. "
    "Answer the following multiple-choice question about the video. "
    "Only output the selected option letter(s) (e.g., 'A', 'B', 'C'), and do not output extra text or explanation."
)
OPEN_SYSTEM_PROMPT = (
    "You are an expert video question answering assistant. "
    "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. "
    "Answer the following question concisely, highlighting any significant events, characters, or objects that appear throughout the frames."
)

# ========== 工具函数 ==========

def create_frame_grid(img_array, interval_width=50):
    n, h, w, c = img_array.shape
    grid_size = int(np.ceil(np.sqrt(n)))
    horizontal_band = np.ones((h, interval_width, c), dtype=img_array.dtype) * 255
    vertical_band = np.ones((interval_width, w + (grid_size - 1) * (w + interval_width), c), dtype=img_array.dtype) * 255

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
        scale = max_length / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = image.resize((new_width, new_height), Image.BILINEAR)
    else:
        img_resized = image
    return img_resized

def video_answer_gpt4o(prompt, img_grid, max_tokens=512, print_res=False, max_retries=5, retry_interval=10):
    buffered = io.BytesIO()
    img_grid.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }
    ]

    last_exception = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
            if print_res:
                print('### PROMPTING GPT-4o WITH: ', prompt)
                print('### GPT-4o OUTPUT TEXT:  ', answer)
            return answer
        except Exception as e:
            last_exception = e
            print(f"[警告] GPT-4o API请求失败（第{attempt+1}/{max_retries}次），错误信息：{e}")
            if attempt < max_retries - 1:
                print(f"将在{retry_interval}秒后重试...")
                time.sleep(retry_interval)
            else:
                print("[错误] 已达到最大重试次数，仍然失败。")
    # 如果一直失败，返回默认响应
    return "API_ERROR"

def single_test(vid_path, qs, system_prompt, num_frames=16):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(video_path, num_segments=8):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames_total = len(vr)
        frame_indices = get_index(num_frames_total, num_segments)
        img_array = vr.get_batch(frame_indices).asnumpy()
        img_grid = create_frame_grid(img_array, 50)
        img_grid = Image.fromarray(img_grid).convert("RGB")
        img_grid = resize_image_grid(img_grid)
        return img_grid

    img_grid = load_video(vid_path, num_segments=num_frames)
    qs_full = f"{system_prompt}\nQuestion: {qs}"
    llm_response = video_answer_gpt4o(qs_full, img_grid, max_tokens=512, print_res=True)
    return llm_response

def readjson_dir(input_dir):
    json_files = glob(os.path.join(input_dir, "*.json"))
    all_data = []
    for path in json_files:
        with open(path, 'r') as f:
            data = json.load(f)
        filename = os.path.basename(path)
        all_data.append((filename, data))
    return all_data

def count_existing_videos(id_prefix):
    return len(glob(os.path.join(VIDEO_FOLDER, f"{id_prefix}/{id_prefix}_*.mp4")))

def process_binaryqa(json_name, data):
    processed_count = 0
    
    for entry in data:
        if json_name == "T_binary_qa.json":
            video_id = entry["id"]
        else:
            video_id = entry["id"].rsplit('_', 1)[0]
        video_folder = f"{VIDEO_FOLDER}/{video_id}"
        if not os.path.exists(video_folder):
            print(f"[跳过] 视频文件夹不存在：{video_id}")
            continue

        preds_0, preds_1, preds_2, preds_pos, preds_neg = [], [], [], [], []

        if "binary_qa_0" in entry:
            for qa in entry["binary_qa_0"]:
                file_name = qa["file"]
                video_path = f"{video_folder}/{file_name}"
                if not os.path.exists(video_path): continue
                q = qa["question"]
                pred = single_test(video_path, q, BINARY_SYSTEM_PROMPT, num_frames=num_frames)
                preds_0.append({"file_name": qa["file"], "prediction": pred})

        if "binary_qa_1" in entry:
            for qa in entry["binary_qa_1"]:
                file_name = qa["file"]
                video_path = f"{video_folder}/{file_name}"
                if not os.path.exists(video_path): continue
                q = qa["question"]
                pred = single_test(video_path, q, BINARY_SYSTEM_PROMPT, num_frames=num_frames)
                preds_1.append({"file_name": qa["file"], "prediction": pred})

        if "binary_qa_2" in entry:
            for qa in entry["binary_qa_2"]:
                file_name = qa["file"]
                video_path = f"{video_folder}/{file_name}"
                if not os.path.exists(video_path): continue
                q = qa["question"]
                pred = single_test(video_path, q, BINARY_SYSTEM_PROMPT, num_frames=num_frames)
                preds_2.append({"file_name": qa["file"], "prediction": pred})

        if "binary_qa_pos" in entry:
            for qa in entry["binary_qa_pos"]:
                file_name = qa["file_name"]
                video_path = f"{video_folder}/{file_name}"
                if not os.path.exists(video_path): continue
                q = qa["question"]
                pred = single_test(video_path, q, BINARY_SYSTEM_PROMPT, num_frames=num_frames)
                preds_pos.append({"file_name": qa["file_name"], "prediction": pred})

        if "binary_qa_neg" in entry:
            for qa in entry["binary_qa_neg"]:
                file_name = qa["file_name"]
                video_path = f"{video_folder}/{file_name}"
                if not os.path.exists(video_path): continue
                q = qa["question"]
                pred = single_test(video_path, q, BINARY_SYSTEM_PROMPT, num_frames=num_frames)
                preds_neg.append({"file_name": qa["file_name"], "prediction": pred})

        predicted = {}
        if preds_0: predicted["binary_qa_0"] = preds_0
        if preds_1: predicted["binary_qa_1"] = preds_1
        if preds_2: predicted["binary_qa_2"] = preds_2
        if preds_pos: predicted["binary_qa_pos"] = preds_pos
        if preds_neg: predicted["binary_qa_neg"] = preds_neg

        if predicted:
            entry["predicted"] = predicted
        
        processed_count += 1
        print(f"[处理完成] {video_id} - 已处理 {processed_count} 个条目")
        
        # 每处理100次就保存一次
        if processed_count % 100 == 0:
            print(f"[保存中间结果] 已处理 {processed_count} 个条目，正在保存...")
            output_path = os.path.join(OUTPUT_DIR, json_name)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[✓] 中间结果已保存到：{output_path}")

    output_path = os.path.join(OUTPUT_DIR, json_name)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[✓] 已处理 BinaryQA 文件：{output_path}")

def process_multipleqa(json_name, data):
    processed_count = 0
    
    for entry in data:
        video_id = entry["id"].rsplit('_', 1)[0]
        video_folder = f"{VIDEO_FOLDER}/{video_id}"
        if not os.path.exists(video_folder): 
            print(f"[跳过] 视频文件夹不存在：{video_id}")
            continue
        if count_existing_videos(video_id) < 2:
            print(f"[跳过 MultipleQA] 视频配对不足：{video_id}")
            continue

        preds = []
        for item in entry["multiple_qa"]:
            file_name = item["file"]
            video_path = f"{video_folder}/{file_name}"
            if not os.path.exists(video_path): continue
            q = item["question"]
            pred = single_test(video_path, q, MULTIPLE_SYSTEM_PROMPT, num_frames=num_frames)
            preds.append({"file_name": item["file"], "prediction": pred})

        entry["predicted"] = {"multiple_qa": preds}
        
        processed_count += 1
        print(f"[处理完成] {video_id} - 已处理 {processed_count} 个条目")
        
        # 每处理100次就保存一次
        if processed_count % 100 == 0:
            print(f"[保存中间结果] 已处理 {processed_count} 个条目，正在保存...")
            output_path = os.path.join(OUTPUT_DIR, json_name)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[✓] 中间结果已保存到：{output_path}")

    output_path = os.path.join(OUTPUT_DIR, json_name)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[✓] 已处理 MultipleQA 文件：{output_path}")

def process_openqa(json_name, data):
    processed_count = 0
    
    for entry in data:
        video_id = entry["id"]
        video_folder = f"{VIDEO_FOLDER}/{video_id}"
        if not os.path.exists(video_folder): 
            print(f"[跳过] 视频文件夹不存在：{video_id}")
            continue
        if count_existing_videos(video_id) < 2:
            print(f"[跳过 OpenQA] 视频配对不足：{video_id}")
            continue

        preds = []
        for item in entry["open_qa"]:
            video_path = os.path.join(video_folder, item["file_name"])
            if not os.path.exists(video_path): continue
            q = item["question"]
            pred = single_test(video_path, q, OPEN_SYSTEM_PROMPT, num_frames=num_frames)
            preds.append({"file_name": item["file_name"], "prediction": pred})

        entry["predicted"] = {"open_qa": preds}
        
        processed_count += 1
        print(f"[处理完成] {video_id} - 已处理 {processed_count} 个条目")
        
        # 每处理100次就保存一次
        if processed_count % 100 == 0:
            print(f"[保存中间结果] 已处理 {processed_count} 个条目，正在保存...")
            output_path = os.path.join(OUTPUT_DIR, json_name)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[✓] 中间结果已保存到：{output_path}")

    output_path = os.path.join(OUTPUT_DIR, json_name)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[✓] 已处理 OpenQA 文件：{output_path}")

def test_api_connection():
    """测试API连接"""
    print("测试API连接...")
    try:
        # 创建一个简单的测试图像
        test_img = Image.new('RGB', (100, 100), color='red')
        test_response = video_answer_gpt4o("What color is this image?", test_img, max_tokens=10, print_res=False)
        print(f"API测试成功，响应: {test_response}")
        return True
    except Exception as e:
        print(f"API测试失败: {str(e)}")
        return False

def main():
    # 首先测试API连接
    if not test_api_connection():
        print("API连接失败，请检查网络和API配置")
        return
    
    all_jsons = readjson_dir(INPUT_DIR)
    for json_name, data in all_jsons:
        if "binary" in json_name.lower():
            process_binaryqa(json_name, data)
        elif "multiple" in json_name.lower():
            process_multipleqa(json_name, data)
        elif "open" in json_name.lower():
            process_openqa(json_name, data)
        else:
            print(f"[跳过] 无法识别类型：{json_name}")

if __name__ == "__main__":
    main()
