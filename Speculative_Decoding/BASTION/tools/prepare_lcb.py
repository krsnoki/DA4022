"""Build upstream/cache/lcb.jsonl without a script-based `datasets` loader.

Upstream loads ``load_dataset("livecodebench/code_generation_lite", "release_v2", split="test")``.
That repo ships only a loading script (code_generation_lite.py) plus raw JSONL files, and
``datasets>=4`` refuses dataset scripts ("Dataset scripts are no longer supported").

The script's ``release_v2`` config simply yields every line of ``test.jsonl`` then
``test2.jsonl`` in order (ALLOWED_FILES["release_v2"]). This helper downloads exactly those
two files from a pinned dataset revision, applies upstream's own
``DATASETS["lcb"]["format"]`` and writes the same ``{"turns": [...]}`` JSONL that
``bastion.benchmark._prepare_dataset`` would have produced. No upstream file is modified.

Run with the venv python:  .venv/bin/python tools/prepare_lcb.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream"
sys.path.insert(0, str(UPSTREAM))

from huggingface_hub import hf_hub_download  # noqa: E402

from bastion.benchmark import CACHE_DIR, DATASETS  # noqa: E402

REPO = "livecodebench/code_generation_lite"
REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"  # main as of 2026-11-21
RELEASE_V2_FILES = ["test.jsonl", "test2.jsonl"]


def main() -> None:
    fmt = DATASETS["lcb"]["format"]
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / "lcb.jsonl"
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    n = 0
    with open(tmp_path, "w", encoding="utf-8") as out:
        for name in RELEASE_V2_FILES:
            local = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)
            with open(local, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    out.write(json.dumps({"turns": [fmt(row)]}, ensure_ascii=False) + "\n")
                    n += 1
    os.replace(tmp_path, out_path)
    print(f"[cached] {out_path} ({n} samples) from {REPO}@{REVISION} files={RELEASE_V2_FILES}")


if __name__ == "__main__":
    main()
