import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Mask value_0 segments in composed videos.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input MP4 files.")
    parser.add_argument("--output-dir", required=True, help="Directory for masked MP4 outputs.")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
    return parser.parse_args()


def main():
    args = parse_args()
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    from moviepy.video.fx.all import colorx

    os.makedirs(args.output_dir, exist_ok=True)

    for file_name in os.listdir(args.input_dir):
        if not file_name.endswith(".mp4"):
            continue

        parts = file_name.split("_")[-1].split(".")[0]
        value_0_indices = [i for i, part in enumerate(parts) if part == "0"]
        video_path = os.path.join(args.input_dir, file_name)

        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            segment_duration = duration / 3

            masked_clips = []
            for i in range(3):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                segment = clip.subclip(start_time, end_time)
                masked_clips.append(colorx(segment, 0) if i in value_0_indices else segment)

            final_clip = concatenate_videoclips(masked_clips, method="compose")
            out_filename = f"masked_{file_name}"
            out_path = os.path.join(args.output_dir, out_filename)
            final_clip.write_videofile(out_path, codec="libx264", fps=args.fps)
            final_clip.close()
            clip.close()
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print("Masking complete.")


if __name__ == "__main__":
    main()
