# VidPair-Halluc

**No Place to Hide: Benchmarking Video Hallucination with Background-Controlled Pairs**

[Project page](https://jethrojames.github.io/VidPair-Halluc/) |
[Dataset](https://huggingface.co/datasets/Jethro37/VidPair-Halluc) |
[Code](https://github.com/JethroJames/VidPair-Halluc)

<p align="center">
  <img src="assets/flowchart.png" alt="VidPair-Halluc teaser figure" width="800">
</p>

## Overview

VidPair-Halluc is a paired-video benchmark for evaluating whether video-language models answer from visible evidence or from scene priors. Each pair is designed to keep the surrounding scene context comparable while changing the answer-relevant foreground evidence or temporal order.

The v1.0 public dataset contains:

- 2,000 video clips organized as 1,000 background-controlled paired contrasts.
- 11,523 QA rows: 4,000 binary, 4,000 multiple-choice, and 3,523 open-ended prompts.
- Ten semantic aspects: object, action, color, number, person, location, static relation, dynamic relation, dynamic attribute, and temporal sequence.

Videos and QA files are hosted on Hugging Face. This repository contains lightweight release readers, validation code, construction utilities, and model-evaluation examples.

## Repository Structure

```text
.
+-- README.md
+-- requirements.txt
+-- assets/
|   +-- flowchart.png
+-- vidpair_halluc/
|   +-- dataset.py
+-- scripts/
|   +-- validate_release.py
+-- process_data.py
+-- generate_seg_json.py
+-- story.json
+-- seg.json
+-- negative_tool/
|   +-- mask.py
|   +-- replace.py
|   +-- replace_json.py
|   +-- reverse.py
+-- evaluate/
    +-- clipqa.py
    +-- clipqa_gemini_2_5_pro.py
    +-- clipqa_gpt4o.py
    +-- videoqa.py
```

## Dataset Layout

Download the public dataset from:

```text
https://huggingface.co/datasets/Jethro37/VidPair-Halluc
```

The Hugging Face dataset root is expected to contain:

```text
VidPair-Halluc/<id>/<video>.mp4
metadata.csv
pairs/pairs.json
qa/hf/binary.jsonl
qa/hf/multiple_choice.jsonl
qa/hf/open_ended.jsonl
qa/hf/all_qa.jsonl
```

Canonical QA rows use `repo_video_path` paths such as `VidPair-Halluc/action_0002/M_action_0002_001.mp4`, so every QA item can be mapped back to a concrete video file.

## Quick Start

Install optional dependencies for video/model evaluation:

```bash
pip install -r requirements.txt
```

Validate a local dataset checkout or Hugging Face snapshot:

```bash
python scripts/validate_release.py --dataset-root /path/to/VidPair-Halluc-dataset
```

Use the lightweight reader:

```python
from vidpair_halluc import ReleaseDataset

dataset = ReleaseDataset("/path/to/VidPair-Halluc-dataset")
print(dataset.count_videos())
first_row = next(dataset.qa_rows("binary", parse_json_fields=True))
print(first_row["question"])
print(dataset.video_path(first_row))
```

## Construction Utilities

The construction scripts are for rebuilding intermediate `story.json` and `seg.json` files from raw assets. They are not required for loading the released Hugging Face dataset.

```bash
python process_data.py \
  --work-dir /path/to/workdir \
  --vidpair-dir VidPair-Halluc \
  --raw-data-dir raw_data \
  --processed-dir processed_data \
  --story-output story.json

python generate_seg_json.py \
  --work-dir /path/to/workdir \
  --story-json story.json \
  --processed-dir processed_data \
  --output seg.json
```

The `negative_tool/` scripts are command-line utilities for generating masked, reordered, and reversed variants during construction. Run each script with `--help` for its arguments.

## Evaluation

The `evaluate/` directory contains model-calling examples for clip-level and video-level QA. These scripts require local video paths plus model/API configuration. The release validator above should be run first to ensure the QA rows and video files are path-consistent.

## Citation

```bibtex
@article{huang2026no,
  title={No Place to Hide: Benchmarking Video Hallucination with Background-Controlled Pairs},
  author={Huang, Haojian and Chen, Harold Haodong and Luo, Meng and Du, Junjia and Xu, Shanqing and Chen, Ziheng and Huang, Yanxiang and Li, Yinchuan and Chen, Ying-Cong},
  journal={arXiv preprint arXiv:2606.31933},
  year={2026}
}
```
