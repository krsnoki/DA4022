"""Reusable pieces of the offline replay, shared by ``replay_sweep`` and the analysis scripts.

* ``prepare``: segment, tag and score the cached traces (the segmentation and the
  tagger are parameters, so variants such as a minimal step size or a second
  tagger reuse the same pipeline);
* ``cut_points``: where each configuration stops on each trace;
* ``ForcedAnswerCache``: forced answers keyed by ``(uid, cut)``. They depend only
  on the trace and the cut, not on how the cut was chosen, so every analysis of
  one trace file shares a single cache;
* ``aggregate``: per-configuration accuracy, tokens and stop rate, plus per-problem rows.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from Adaptive_Stopping.common.forced_answer import answer_text, force_answers_shared_prefix
from Adaptive_Stopping.common.replay import st_es_cut_points, step_count_cut_points
from Adaptive_Stopping.common.steps import Step, segment_token_ids
from Adaptive_Stopping.experiments.st_es.io_utils import DEFAULT_RUN_DIR, append_jsonl, read_jsonl
from Adaptive_Stopping.methods.step_tagging.calibration import Config
from Performance_Metrics.quality.math_answer import is_correct

StepsFn = Callable[[Sequence[int], object], list[Step]]


def trace_dir(traces: Path, run_dir: Path = DEFAULT_RUN_DIR) -> Path:
    """``<run_dir>/replay/<dataset>/<traces stem>``: home of every replay output of one trace file."""
    return Path(run_dir) / "replay" / traces.parent.name / traces.stem


def prepare(records: Iterable[dict], tokenizer, tagger, steps_fn: StepsFn | None = None) -> list[dict]:
    steps_fn = steps_fn or segment_token_ids
    prepared = []
    for r in records:
        steps = steps_fn(r["gen_ids"], tokenizer)
        prepared.append({
            "rec": r, "steps": steps, "tags": tagger.tag_all(s.text for s in steps),
            "base_correct": is_correct(r["gen_text"], r["gold"], r["gold_format"]),
            "n_eligible": sum(1 for s in steps if s.closed and s.in_thinking),
        })
    return prepared


def cut_points(prep: dict, configs: Sequence[Config]) -> dict[str, int | None]:
    """``{config key: cut (generated tokens kept) or None (no stop)}`` for one prepared trace."""
    by_tag = defaultdict(list)
    ks = []
    for c in configs:
        if c.kind == "st_es":
            by_tag[c.tag].append(c.delta)
        else:
            ks.append(c.k)
    out = {}
    for tag, deltas in by_tag.items():
        for d, cut in st_es_cut_points(prep["steps"], prep["tags"], tag, deltas).items():
            out[Config("st_es", tag=tag, delta=d).key] = cut
    for k, cut in step_count_cut_points(prep["steps"], ks).items():
        out[Config("step_count", k=k).key] = cut
    return out


class ForcedAnswerCache:
    """Forced answers of one trace file, persisted as JSONL and keyed by ``(uid, cut)``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[tuple[str, int], dict] = {}
        if self.path.exists():
            for row in read_jsonl(self.path):
                self.rows[(row["uid"], row["cut"])] = row

    def __getitem__(self, key: tuple[str, int]) -> dict:
        return self.rows[key]

    def __len__(self) -> int:
        return len(self.rows)

    def missing(self, prepared: Sequence[dict], cuts: dict[str, dict[str, int | None]]) -> dict[str, set[int]]:
        todo = defaultdict(set)
        for p in prepared:
            uid = p["rec"]["uid"]
            for cut in cuts[uid].values():
                if cut is not None and (uid, cut) not in self.rows:
                    todo[uid].add(cut)
        return todo

    def fill(self, prepared: Sequence[dict], todo: dict[str, set[int]], backbone, exit_prompt_id: str,
             log=print) -> float:
        """Compute the missing answers with the (loaded) backbone; returns wall seconds."""
        t0 = time.time()
        tok, eos = backbone.tokenizer, backbone.eos_token_ids
        done = 0
        for p in prepared:
            uid = p["rec"]["uid"]
            if uid not in todo:
                continue
            r = p["rec"]
            plen = len(r["prompt_ids"])
            res = force_answers_shared_prefix(backbone.model, tok, r["prompt_ids"] + r["gen_ids"],
                                              [plen + c for c in sorted(todo[uid])], exit_prompt_id=exit_prompt_id,
                                              eos_ids=eos, pad_id=tok.pad_token_id)
            for ctx_len, ans in res.items():
                cut = ctx_len - plen
                text = answer_text(tok, ans, exit_prompt_id)
                row = {"uid": uid, "cut": cut, "exit_prompt_id": exit_prompt_id, "answer_ids": ans,
                       "answer_text": text, "correct": is_correct(text, r["gold"], r["gold_format"])}
                self.rows[(uid, cut)] = row
                append_jsonl(self.path, row)
            done += 1
            if done % 10 == 0:
                log(f"  forced answers: {done}/{len(todo)} traces, {time.time() - t0:.0f}s")
        return time.time() - t0


def standard_row(prepared: Sequence[dict]) -> dict:
    n = len(prepared)
    return {"kind": "standard", "key": "standard", "tag": None, "delta": None, "k": None, "n": n,
            "accuracy": sum(p["base_correct"] for p in prepared) / n,
            "mean_tokens": sum(p["rec"]["n_gen_tokens"] for p in prepared) / n, "saved_pct": 0.0,
            "stop_rate": 0.0, "mean_answer_tokens": 0.0}


def aggregate(prepared: Sequence[dict], configs: Sequence[Config], cuts: dict[str, dict[str, int | None]],
              forced: ForcedAnswerCache) -> tuple[list[dict], list[dict]]:
    """Rows (standard first, then one per config) and per-problem rows, as written to sweep/per_example CSVs."""
    std = standard_row(prepared)
    n = std["n"]
    rows = [std]
    per_example = [{"key": "standard", "uid": p["rec"]["uid"], "stopped": False, "cut": None,
                    "tokens": p["rec"]["n_gen_tokens"], "correct": int(p["base_correct"])} for p in prepared]
    for c in configs:
        acc = tok_sum = ans_sum = stops = 0
        for p in prepared:
            r = p["rec"]
            cut = cuts[r["uid"]][c.key]
            if cut is None:
                correct, used, ans_len = p["base_correct"], r["n_gen_tokens"], 0
            else:
                f = forced[(r["uid"], cut)]
                correct, ans_len = f["correct"], len(f["answer_ids"])
                used = cut + ans_len
                stops += 1
            acc += correct
            tok_sum += used
            ans_sum += ans_len
            per_example.append({"key": c.key, "uid": r["uid"], "stopped": cut is not None, "cut": cut,
                                "tokens": used, "correct": int(correct)})
        rows.append({"kind": c.kind, "key": c.key, "tag": c.tag, "delta": c.delta, "k": c.k, "n": n,
                     "accuracy": acc / n, "mean_tokens": tok_sum / n,
                     "saved_pct": 100 * (1 - tok_sum / n / std["mean_tokens"]), "stop_rate": stops / n,
                     "mean_answer_tokens": ans_sum / max(stops, 1)})
    return rows, per_example
