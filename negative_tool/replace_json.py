import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Generate JSON metadata for replacement-composed videos.")
    parser.add_argument("--video-dir", required=True, help="Directory containing replacement MP4 files.")
    parser.add_argument("--json-dir", required=True, help="Directory containing source category JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated category JSON files.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_categories_data = {}
    for file_name in os.listdir(args.video_dir):
        if not file_name.endswith(".mp4"):
            continue

        action, action_id, index = file_name.split(".")[0].rsplit("_", 2)
        json_file_name = f"{action}.json"
        json_file_path = os.path.join(args.json_dir, json_file_name)

        if not os.path.exists(json_file_path):
            print(f"JSON file for {action} not found, skipping...")
            continue

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        action_data = next((item for item in data[action] if item["id"] == f"{action}_{action_id}"), None)
        if not action_data:
            print(f"No matching action found for {action}_{action_id} in {json_file_name}, skipping...")
            continue

        replacement_indices = [int(i) for i in index]
        modified_segments = []
        for i, segment in enumerate(action_data["segments"]):
            value_index = replacement_indices[i]
            value_to_insert = action_data["values"][value_index]
            modified_segments.append(segment.replace("{values}", value_to_insert))

        modified_action_data = {
            "id": f"{action}_{action_id}_{index}",
            "segments": modified_segments,
        }

        all_categories_data.setdefault(action, []).append(modified_action_data)

    for action, actions in all_categories_data.items():
        output_file_path = os.path.join(args.output_dir, f"{action}.json")
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump({action: actions}, f, ensure_ascii=False, indent=2)
        print(f"Processed and saved: {output_file_path}")

    print("All files processed.")


if __name__ == "__main__":
    main()
