import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compose replacement videos from value-specific segment videos.")
    parser.add_argument("--base-dir", required=True, help="Directory containing value subdirectories.")
    parser.add_argument(
        "--value-dirs",
        nargs=3,
        default=["value_0_test", "value_1_test", "value_2_test"],
        help="Three subdirectories corresponding to values 0, 1, and 2.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for composed MP4 outputs.")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
    return parser.parse_args()


def main():
    args = parse_args()
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    os.makedirs(args.output_dir, exist_ok=True)

    files_0 = os.listdir(os.path.join(args.base_dir, args.value_dirs[0]))
    for file_name in files_0:
        if not file_name.endswith(".mp4"):
            continue

        file_exists_in_1 = os.path.exists(os.path.join(args.base_dir, args.value_dirs[1], file_name))
        file_exists_in_2 = os.path.exists(os.path.join(args.base_dir, args.value_dirs[2], file_name))
        if not (file_exists_in_1 or file_exists_in_2):
            continue

        file_paths = [os.path.join(args.base_dir, vp, file_name) for vp in args.value_dirs]
        segments = []
        source_clips = []
        for path in file_paths:
            if os.path.exists(path):
                try:
                    clip = VideoFileClip(path)
                    source_clips.append(clip)
                    duration = clip.duration
                    segment_duration = duration / 3
                    for i in range(3):
                        start_time = i * segment_duration
                        end_time = (i + 1) * segment_duration
                        segments.append(clip.subclip(start_time, end_time))
                except Exception as e:
                    print(f"Error processing {path}: {e}")

        if len(segments) < 3:
            print(f"Not enough segments from value_0 for {file_name}, skipping...")
            for clip in source_clips:
                clip.close()
            continue

        combinations = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if not (i == 0 and j == 0 and k == 0):
                        combo = (i, j, k)
                        replacement_info = [str(combo[idx]) if combo[idx] != 0 else "0" for idx in range(3)]
                        combinations.append((combo, replacement_info))

        for combo, replacement_info in combinations:
            if 0 not in combo:
                continue
            try:
                new_clips = [segments[3 * seg_idx + part_idx] for part_idx, seg_idx in enumerate(combo)]
                final_clip = concatenate_videoclips(new_clips, method="compose")
                out_filename = f"{os.path.splitext(file_name)[0]}_{''.join(replacement_info)}.mp4"
                out_path = os.path.join(args.output_dir, out_filename)
                final_clip.write_videofile(out_path, codec="libx264", fps=args.fps)
                final_clip.close()
            except Exception as e:
                print(f"Error creating video for {file_name} with combo {combo}: {e}")

        for clip in source_clips:
            clip.close()

    print("Processing complete.")


if __name__ == "__main__":
    main()
