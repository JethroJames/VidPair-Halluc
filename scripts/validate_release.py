#!/usr/bin/env python3
"""Validate a local VidPair-Halluc Hugging Face dataset checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vidpair_halluc import ReleaseDataset


EXPECTED_COUNTS = {
    "videos": 2000,
    "metadata": 2000,
    "pairs": 1000,
    "binary": 4000,
    "multiple_choice": 4000,
    "open_ended": 3523,
    "all": 11523,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to a local VidPair-Halluc dataset root, e.g. a Hugging Face snapshot.",
    )
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Only validate structure/path links; do not require the v1.0 public counts.",
    )
    parser.add_argument(
        "--probe-mp4-headers",
        action="store_true",
        help="Read each MP4 header and check that it looks like an MP4 file.",
    )
    return parser.parse_args()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def validate_mp4_headers(dataset: ReleaseDataset, errors: list[str]) -> None:
    bad = []
    for path in dataset.video_root.glob("*/*.mp4"):
        try:
            header = path.read_bytes()[:16]
        except OSError:
            bad.append(str(path.relative_to(dataset.root)))
            continue
        if b"ftyp" not in header:
            bad.append(str(path.relative_to(dataset.root)))
    if bad:
        fail(errors, f"{len(bad)} MP4 files have invalid-looking headers; first={bad[:5]}")
    else:
        ok("MP4 header probe passed")


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root)
    dataset = ReleaseDataset(root)
    errors: list[str] = []

    required = [
        "DATA_LAYOUT.md",
        "metadata.csv",
        "pairs/pairs.json",
        "qa/hf/binary.jsonl",
        "qa/hf/multiple_choice.jsonl",
        "qa/hf/open_ended.jsonl",
        "qa/hf/all_qa.jsonl",
        "VidPair-Halluc",
    ]
    for rel in required:
        path = root / rel
        if not path.exists():
            fail(errors, f"missing required path: {rel}")
    if errors:
        return 1

    video_count = dataset.count_videos()
    metadata_count = len(dataset.metadata())
    pairs = dataset.pairs()
    qa_counts = {split: sum(1 for _ in dataset.qa_rows(split)) for split in ("binary", "multiple_choice", "open_ended", "all")}

    observed = {
        "videos": video_count,
        "metadata": metadata_count,
        "pairs": len(pairs),
        **qa_counts,
    }
    print(json.dumps({"observed_counts": observed}, indent=2, sort_keys=True))

    if not args.skip_count_check:
        for key, expected in EXPECTED_COUNTS.items():
            if observed[key] != expected:
                fail(errors, f"{key} count mismatch: expected {expected}, got {observed[key]}")
            else:
                ok(f"{key} count = {expected}")

    missing_qa = dataset.missing_qa_videos("all", limit=10)
    if missing_qa:
        fail(errors, f"QA rows reference missing videos; first={missing_qa}")
    else:
        ok("all QA rows resolve to existing videos")

    missing_pairs = []
    for pair in pairs:
        for key in ("positive_video", "adversarial_video"):
            path = dataset.video_path(str(pair[key]))
            if not path.exists():
                missing_pairs.append(f"{pair.get('pair_id')}:{key}:{pair[key]}")
                if len(missing_pairs) >= 10:
                    break
        if len(missing_pairs) >= 10:
            break
    if missing_pairs:
        fail(errors, f"pair manifest references missing videos; first={missing_pairs}")
    else:
        ok("all pair manifest videos resolve")

    if args.probe_mp4_headers:
        validate_mp4_headers(dataset, errors)

    if errors:
        print(f"\nValidation failed with {len(errors)} issue(s).")
        return 1
    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
