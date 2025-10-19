import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 定义路径
base_path = "/data/harold/negative_tool"
value_paths = ["value_0_test", "value_1_test", "value_2_test"]

output_path = "/data/harold/negative_tool/replaced_video"

# 获取所有文件名
files_0 = os.listdir(os.path.join(base_path, value_paths[0]))

# 遍历 value_0_test 目录中的所有文件
for file_name in files_0:
    if not file_name.endswith('.mp4'):
        continue
    
    # 检查其他两个目录中是否存在同名文件
    file_exists_in_1 = os.path.exists(os.path.join(base_path, value_paths[1], file_name))
    file_exists_in_2 = os.path.exists(os.path.join(base_path, value_paths[2], file_name))
    
    # 只处理至少一个目录存在该文件的情况
    if not (file_exists_in_1 or file_exists_in_2):
        continue
    
    # 如果至少一个目录存在该文件，进行处理
    file_paths = [os.path.join(base_path, vp, file_name) for vp in value_paths]
    
    # 读取视频并切割成三个片段
    segments = []
    for path in file_paths:
        if os.path.exists(path):
            try:
                clip = VideoFileClip(path)
                duration = clip.duration
                segment_duration = duration / 3
                
                for i in range(3):
                    start_time = i * segment_duration
                    end_time = (i + 1) * segment_duration
                    segment = clip.subclip(start_time, end_time)
                    segments.append(segment)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    
    # 检查是否成功读取到所有片段
    if len(segments) < 3:
        print(f"Not enough segments from value_0 for {file_name}, skipping...")
        continue
    
    # 生成组合，确保至少一个片段被替换
    combinations = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # 确保至少有一个片段被替换
                if not (i == 0 and j == 0 and k == 0):
                    combo = (i, j, k)
                    replacement_info = [str(combo[idx]) if combo[idx] != 0 else '0' for idx in range(3)]
                    combinations.append((combo, replacement_info))
    
    # 处理每个组合并生成新视频
    for idx, (combo, replacement_info) in enumerate(combinations):
        # import pdb; pdb.set_trace()
        if 0 in combo:
            try:
                new_clips = [segments[3 * seg_idx + part_idx] for part_idx, seg_idx in enumerate(combo)]
                final_clip = concatenate_videoclips(new_clips, method="compose")
                out_filename = f"{os.path.splitext(file_name)[0]}_{''.join(replacement_info)}.mp4"
                out_path = os.path.join(output_path, out_filename)
                final_clip.write_videofile(out_path, codec="libx264", fps=24)
                final_clip.close()
            except Exception as e:
                print(f"Error creating video for {file_name} with combo {combo}: {e}")
        else:
            continue

print("Processing complete.")