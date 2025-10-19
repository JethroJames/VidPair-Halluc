import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import colorx

# 定义路径
input_path = "/data/harold/negative_tool/replaced_video"
output_path = "/data/harold/negative_tool/masked_video"

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 遍历目录中的所有视频文件
for file_name in os.listdir(input_path):
    if not file_name.endswith('.mp4'):
        continue
    
    # 解析文件名来确定哪些片段属于value_0
    parts = file_name.split('_')[-1].split('.')[0]  # 获取最后的组合信息部分
    value_0_indices = [i for i, part in enumerate(parts) if part == '0']
    
    # 读取视频
    video_path = os.path.join(input_path, file_name)
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        segment_duration = duration / 3
        
        # 遍历3个片段
        masked_clips = []
        for i in range(3):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            segment = clip.subclip(start_time, end_time)
            
            # 对属于value_0的片段进行遮罩
            if i in value_0_indices:
                # 使用colorx将片段变为黑色
                masked_segment = colorx(segment, 0)  # 将亮度调整为0，变成黑色
                masked_clips.append(masked_segment)
            else:
                masked_clips.append(segment)
        
        # 拼接新的视频
        final_clip = concatenate_videoclips(masked_clips, method="compose")
        out_filename = f"masked_{file_name}"
        out_path = os.path.join(output_path, out_filename)
        final_clip.write_videofile(out_path, codec="libx264", fps=24)
        final_clip.close()
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

print("Masking complete.")
