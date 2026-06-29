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

The released dataset currently contains:

- 2,000 video clips organized as 1,000 background-controlled paired contrasts.
- 11,523 QA rows: 4,000 binary, 4,000 multiple-choice, and 3,523 open-ended prompts.
- Ten semantic aspects: object, action, color, number, person, location, static relation, dynamic relation, dynamic attribute, and temporal sequence.

Video and QA files are hosted on Hugging Face. This repository contains the construction and evaluation utilities used with the dataset.

## Repository Structure

```text
.
+-- README.md
+-- assets/
|   +-- flowchart.png
+-- process_data.py
+-- generate_seg_json.py
+-- story.json
+-- seg.json
+-- negative_tool/
|   +-- mask.py
|   +-- replace.py
|   +-- replace_json.py
|   +-- reverse.py
+-- evalaute/
    +-- clipqa.py
    +-- clipqa_gemini_2_5_pro.py
    +-- clipqa_gpt4o.py
    +-- videoqa.py
```

## Data

Download the public dataset from:

```text
https://huggingface.co/datasets/Jethro37/VidPair-Halluc
```

The dataset package is organized around paired video clips and QA JSON files so that binary, multiple-choice, and open-ended evaluation rows can be mapped back to concrete video samples.

## Quick Start

Install the dependencies needed by the specific script you want to run, then use the construction utilities as follows:

```bash
python process_data.py
python generate_seg_json.py
```

The evaluation scripts under `evalaute/` provide examples for running clip-level and video-level QA evaluation with different model backends.

## Citation

```bibtex
@inproceedings{vidpairhalluc2026,
  title = {No Place to Hide: Benchmarking Video Hallucination with Background-Controlled Pairs},
  author = {Huang, Haojian and Chen, Harold Haodong and Luo, Meng and Du, Junjia and Xu, Shanqing and Chen, Ziheng and Huang, Yanxiang and Li, Yinchuan and Chen, Yingcong},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2026},
  url = {https://github.com/JethroJames/VidPair-Halluc}
}
```
