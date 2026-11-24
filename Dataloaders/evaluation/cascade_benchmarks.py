"""Evaluation loaders + graders for LLM-cascade experiments (MMLU, TriviaQA, MATH L3-5).

Prompts are the exact formatted prompts released with Bouchard (2026),
"Is Escalation Worth It?" (code: github.com/dylanbouchard/llm-cascade-frontiers,
Apache-2.0), pinned to a commit and verified by SHA-256. The underlying
datasets keep their own licences (MMLU: MIT; TriviaQA: Apache-2.0;
MATH: MIT) - cite the original dataset papers.

Graders follow the paper's released grading code:
* MMLU     - first standalone A-D letter in the response == gold letter.
* TriviaQA - normalised exact match (lowercase, strip punctuation/articles,
             collapse whitespace) against any answer alias.
* MATH     - last \\boxed{} content (fallback: last number), symbolic
             equivalence with SymPy, then normalised string match.

Record contract (one dict per example)::

    {"uid": "mmlu-123", "dataset": "mmlu", "source_id": 123,
     "prompt": str, "ground_truth": str, "aliases": list[str] | None}

Sampling: ``num_examples`` rows drawn without replacement with
``numpy.random.default_rng(seed)`` from the full prompt file, then kept in
sampled order. The same records (same seed) must be used for every model.
"""

from __future__ import annotations

import hashlib
import os
import re
import string
import unicodedata
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

UPSTREAM_COMMIT = "3872d8accca3f16ba19e29cf8eb8af5cb166140f"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/dylanbouchard/llm-cascade-frontiers/"
    f"{UPSTREAM_COMMIT}/src/data/prompts/{{name}}.parquet"
)
SHA256 = {
    "mmlu": "143054e0eb42c3208b70f68a3df353107ea89aa08530ed7e79b5b6b705cd0e7e",
    "triviaqa": "56398f5e3a3955d658be4ccf5d7ef7c38cb58465795a2650106efa8647f076cc",
    "math_hard": "f3be4c4776235d7a2af6d914074ac20e22e7e93ba781e36dfd2125b024d03c85",
}
DATASETS = tuple(SHA256)
# Generation budget (max new tokens) per dataset used by our experiments.
MAX_NEW_TOKENS = {"mmlu": 16, "triviaqa": 48, "math_hard": 1024}
DEFAULT_CACHE = Path(os.environ.get("DA4022_DATA_CACHE", Path.home() / ".cache" / "da4022" / "cascade_prompts"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_prompt_file(name: str, cache_dir: Optional[Path] = None) -> Path:
    """Download (once) and checksum-verify the upstream prompt parquet."""
    if name not in SHA256:
        raise KeyError(f"unknown dataset {name!r}; choose from {DATASETS}")
    cache_dir = Path(cache_dir or DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.parquet"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(UPSTREAM_URL.format(name=name), tmp)
        tmp.replace(path)
    digest = _sha256(path)
    if digest != SHA256[name]:
        raise ValueError(f"checksum mismatch for {path}: {digest}")
    return path


def records_from_frame(df, name: str) -> List[Dict]:
    out = []
    for row in df.itertuples(index=False):
        aliases = None
        if name == "triviaqa":
            aliases = [a for a in str(row.answer_aliases).split(" | ") if a]
        out.append(
            {
                "uid": f"{name}-{int(row.id)}",
                "dataset": name,
                "source_id": int(row.id),
                "prompt": str(row.prompt),
                "ground_truth": str(row.ground_truth),
                "aliases": aliases,
            }
        )
    return out


def load_records(
    name: str,
    num_examples: Optional[int] = 2000,
    seed: int = 2026,
    cache_dir: Optional[Path] = None,
    frame=None,
) -> List[Dict]:
    """Deterministically sampled evaluation records for one dataset.

    ``frame`` lets tests pass an in-memory DataFrame instead of downloading.
    """
    import pandas as pd

    df = frame if frame is not None else pd.read_parquet(fetch_prompt_file(name, cache_dir))
    df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    if num_examples is not None and num_examples < len(df):
        pos = np.random.default_rng(seed).choice(len(df), size=num_examples, replace=False)
        df = df.iloc[pos].reset_index(drop=True)
    return records_from_frame(df, name)


# ---------------------------------------------------------------- graders

def grade_mmlu(response: str, record: Dict) -> int:
    if not isinstance(response, str):
        return 0
    m = re.search(r"\b([A-D])\b", response.strip(), re.IGNORECASE)
    return int(bool(m) and m.group(1).upper() == record["ground_truth"].strip().upper())


def _norm_trivia(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def grade_triviaqa(response: str, record: Dict) -> int:
    if not isinstance(response, str):
        return 0
    aliases = record.get("aliases") or [record["ground_truth"]]
    return int(_norm_trivia(response) in {_norm_trivia(a) for a in aliases})


def extract_boxed(text: str) -> str:
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start = idx + len(r"\boxed{")
        depth, j = 1, start
        limit = min(len(text), start + 500)
        while j < limit and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1])
        i = idx + 1
    if results:
        return results[-1].strip()
    nums = re.findall(r"[-+]?\d+(?:[./]\d+)?", text)
    return nums[-1].strip() if nums else text.strip()


def _strip_latex(s: str) -> str:
    s = re.sub(r"\\left|\\right|\\,|\\!|\\;|\\displaystyle", "", s)
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("\\dfrac", "\\frac")
    return s.strip()


def _norm_str(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def _sympy_equal(pred: str, ref: str, timeout_s: int = 5) -> Optional[bool]:
    """True/False if SymPy can decide equivalence, None if parsing fails."""
    import signal

    use_alarm = hasattr(signal, "SIGALRM")

    def _handler(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _handler) if use_alarm else None
    if use_alarm:
        signal.alarm(timeout_s)
    try:
        from sympy import N, simplify, sympify

        def _parse(s: str):
            try:
                from sympy.parsing.latex import parse_latex

                return parse_latex(s)
            except Exception:
                s2 = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
                s2 = s2.replace("\\cdot", "*").replace("^", "**").replace("\\pi", "pi")
                s2 = s2.replace("\\sqrt", "sqrt").replace("{", "(").replace("}", ")")
                return sympify(s2)

        a, b = _parse(pred), _parse(ref)
        if simplify(a - b) == 0:
            return True
        try:
            if abs(float(N(a)) - float(N(b))) < 1e-9:
                return True
        except Exception:
            pass
        return False
    except Exception:
        return None
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def grade_math(response: str, record: Dict) -> int:
    if not isinstance(response, str):
        return 0
    pred = extract_boxed(response)
    gt = record["ground_truth"]
    ref = extract_boxed(gt) if r"\boxed{" in gt else gt.strip()
    if _norm_str(_strip_latex(pred)) == _norm_str(_strip_latex(ref)):
        return 1
    res = _sympy_equal(pred, ref)
    return int(res is True)


GRADERS: Dict[str, Callable[[str, Dict], int]] = {
    "mmlu": grade_mmlu,
    "triviaqa": grade_triviaqa,
    "math_hard": grade_math,
}


def grade(response: str, record: Dict) -> int:
    return GRADERS[record["dataset"]](response, record)
