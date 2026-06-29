import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Reverse MP4 videos frame-by-frame.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input MP4 files.")
    parser.add_argument("--output-dir", required=True, help="Directory for reversed MP4 outputs.")
    parser.add_argument("--fps", type=float, default=30, help="Output FPS.")
    return parser.parse_args()


def main():
    args = parse_args()
    import cv2

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".mp4"):
            continue

        file_path = os.path.join(args.input_dir, filename)
        cap = cv2.VideoCapture(file_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            print(f"Skipping empty/unreadable video: {file_path}")
            continue

        frames.reverse()
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(args.output_dir, f"reversed_{filename}")
        out = cv2.VideoWriter(out_path, fourcc, args.fps, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()

        print(f"Processed and saved: {out_path}")

    print("All videos have been processed!")


if __name__ == "__main__":
    main()
