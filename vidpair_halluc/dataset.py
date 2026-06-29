"""Lightweight readers for the VidPair-Halluc Hugging Face layout.

The public dataset stores videos under ``VidPair-Halluc/<id>/<file>.mp4`` and
canonical QA rows under ``qa/hf/*.jsonl``. This module intentionally uses only
the Python standard library so users can validate a downloaded release before
installing model-specific evaluation dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, MutableMapping, Optional


QA_SPLITS = {
    "binary": "qa/hf/binary.jsonl",
    "multiple_choice": "qa/hf/multiple_choice.jsonl",
    "open_ended": "qa/hf/open_ended.jsonl",
    "all": "qa/hf/all_qa.jsonl",
}


class ReleaseDataset:
    """Reader for a local VidPair-Halluc dataset checkout/download."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def video_root(self) -> Path:
        return self.root / "VidPair-Halluc"

    def iter_jsonl(self, relative_path: str | Path) -> Iterator[Dict[str, object]]:
        path = self.root / relative_path
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def qa_rows(self, split: str = "all", parse_json_fields: bool = False) -> Iterator[Dict[str, object]]:
        if split not in QA_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(QA_SPLITS)}")

        for row in self.iter_jsonl(QA_SPLITS[split]):
            if parse_json_fields:
                row = dict(row)
                for key in ("choices_json", "option_map_json", "mask_flags_json"):
                    value = row.get(key)
                    if isinstance(value, str) and value:
                        try:
                            row[key[:-5] if key.endswith("_json") else key] = json.loads(value)
                        except json.JSONDecodeError:
                            row[key[:-5] if key.endswith("_json") else key] = value
            yield row

    def metadata(self) -> List[MutableMapping[str, str]]:
        path = self.root / "metadata.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def pairs(self) -> List[Dict[str, object]]:
        path = self.root / "pairs/pairs.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def video_path(self, row_or_relative_path: Dict[str, object] | str | Path) -> Path:
        if isinstance(row_or_relative_path, dict):
            rel = row_or_relative_path.get("repo_video_path") or row_or_relative_path.get("video_path")
        else:
            rel = row_or_relative_path

        if rel is None:
            raise ValueError("No video path found")

        rel_path = Path(str(rel))
        if rel_path.parts and rel_path.parts[0] == "VidPair-Halluc":
            return self.root / rel_path
        return self.video_root / rel_path

    def count_videos(self) -> int:
        return sum(1 for _ in self.video_root.glob("*/*.mp4"))

    def missing_qa_videos(self, split: str = "all", limit: Optional[int] = None) -> List[str]:
        missing: List[str] = []
        for row in self.qa_rows(split):
            path = self.video_path(row)
            if not path.exists():
                missing.append(str(path.relative_to(self.root)))
                if limit is not None and len(missing) >= limit:
                    break
        return missing
