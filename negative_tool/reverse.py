import cv2
import os

folder_path = "/data/harold/negative_tool/test_video"
output_path = "/data/harold/negative_tool/reversed_video"

for filename in os.listdir(folder_path):
    if filename.endswith(".mp4"):
        file_path = os.path.join(folder_path, filename)
        
        # 打开视频
        cap = cv2.VideoCapture(file_path)
        frames = []

        # 读取所有帧
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # 倒放帧
        frames.reverse()

        # 获取视频信息
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_path, f"reversed_{filename}"), fourcc, 30, (width, height))

        # 写入倒放后的帧
        for frame in frames:
            out.write(frame)

        out.release()

        print(f"Processed and saved: reversed_{filename}")

print("All videos have been processed!")
