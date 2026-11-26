"""Shared helpers for the tempered-remasking experiment scripts.

Covers configuration loading, policy presets, and dataset loading (Hugging Face
``datasets``: prompt and reference extraction for GSM8K, MATH-500, HumanEval
and MBPP), so the experiments are self-contained.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Dict, Iterator, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_ROOT, ".."))
sys.path.insert(0, os.path.join(MODULE_ROOT, "methods"))

from tempered_remasking import SamplerConfig, build_policy  # noqa: E402
from tempered_remasking import tasks  # noqa: E402

DEFAULT_CONFIG = os.path.join(MODULE_ROOT, "configs", "tempered_remasking", "paper.yaml")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def policy_from_preset(cfg: dict, preset: str, tpos: Optional[float] = None, threshold: Optional[float] = None):
    p = dict(cfg["presets"][preset])
    return build_policy(p["policy"], tpos=tpos if tpos is not None else p.get("tpos"),
                        threshold=threshold if threshold is not None else p.get("threshold"))


def sampler_config(cfg: dict, **overrides) -> SamplerConfig:
    s = {k: v for k, v in cfg["sampling"].items() if k != "dtype"}
    s.update({k: v for k, v in overrides.items() if v is not None})
    return SamplerConfig(**s)


def load_problems(cfg: dict, name: str, limit: Optional[int] = None) -> List[dict]:
    """Return ``[{id, prompt, reference, meta}]`` for one of the paper's benchmarks."""
    from datasets import load_dataset

    d = cfg["datasets"][name]
    ds = load_dataset(d["hf_id"], d["config"], split=d["split"]) if d["config"] else load_dataset(d["hf_id"], split=d["split"])
    n = limit if limit is not None else d.get("limit")
    rows = list(ds)[: n] if n else list(ds)
    out = []
    for i, r in enumerate(rows):
        if name == "gsm8k":
            out.append({"id": f"gsm8k/{i}", "prompt": tasks.format_math_prompt(r["question"]),
                        "question": r["question"], "reference": tasks.extract_gsm8k_reference(r["answer"]), "meta": {}})
        elif name == "math500":
            out.append({"id": f"math500/{r.get('unique_id', i)}", "prompt": tasks.format_math_prompt(r["problem"]),
                        "question": r["problem"], "reference": r["answer"], "meta": {}})
        elif name == "humaneval":
            msg = "Complete the following Python function:\n```python\n" + r["prompt"] + "```"
            out.append({"id": r["task_id"], "prompt": tasks.format_code_prompt(msg), "question": r["prompt"], "reference": None,
                        "meta": {"prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]}})
        elif name == "mbpp":
            out.append({"id": f"mbpp/{r['task_id']}", "prompt": tasks.format_code_prompt(r["text"], "\n".join(r["test_list"][:1])),
                        "question": r["text"], "reference": None,
                        "meta": {"test_list": r["test_list"], "test_setup_code": r.get("test_setup_code", "")}})
        else:
            raise ValueError(name)
    return out


def grade(name: str, completion: str, problem: dict):
    """Returns ``(answer, correct)``; answer is ``None`` for code tasks."""
    if name == "gsm8k":
        return tasks.grade_gsm8k(completion, problem["reference"])
    if name == "math500":
        return tasks.grade_math(completion, problem["reference"])
    m = problem["meta"]
    if name == "humaneval":
        return None, tasks.grade_humaneval(completion, m["prompt"], m["test"], m["entry_point"])
    if name == "mbpp":
        return None, tasks.grade_mbpp(completion, m["test_list"], m["test_setup_code"])
    raise ValueError(name)


def environment_metadata() -> Dict[str, object]:
    import torch

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        commit = None
    meta = {
        "commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except ImportError:
        meta["transformers"] = None
    return meta


def read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return iter(())
    with open(path, encoding="utf-8") as f:
        return iter([json.loads(line) for line in f if line.strip()])


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
