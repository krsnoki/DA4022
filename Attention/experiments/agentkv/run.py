"""Reproducible cloud pilot and download-free CPU plumbing smoke run.

Run from repository root with python -m Attention.experiments.agentkv.run.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
import transformers

from Attention.methods.agentkv import AgentKVConfig
from Attention.methods.agentkv.qwen3 import Qwen3Session
from Backbones.Generative_Pretrained.qwen3 import load
from Dataloaders.synthetic.agent_memory import episodes, answer_diagnostics
from Performance_Metrics.inference import summarize, timed


class ByteTokenizer:
    """Only for plumbing tests with random weights; not a model tokenizer."""
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(text.encode("ascii"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("ascii", errors="replace")


def fragments(tokenizer, ids):
    return [tokenizer.decode([i], skip_special_tokens=False) for i in ids]


def generate(session, tokenizer, logits, limit):
    generated = []
    eos = session.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for _ in range(limit):
        token = int(logits.argmax().item())
        generated.append(token)
        logits = session.append([token], fragments(tokenizer, [token]), decode=True)
        if token in eos:
            break
    return tokenizer.decode(generated, skip_special_tokens=False), len(generated), bool(generated and generated[-1] in eos)


def run_episode(model, tokenizer, episode, mode, policy, config, smoke=False):
    device = next(model.parameters()).device
    timing = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    output_tokens = 0
    input_tokens = 0
    texts, truncated = [], []
    events = []
    with Qwen3Session(model, mode, policy) as session:
        for turn, prompt in enumerate((episode.first_prompt, episode.continuation)):
            # Explicitly close a capped assistant turn before the next message.
            # This supplied delimiter is counted as input, not generated output.
            if turn and truncated[-1]:
                prompt = "<|im_end|>" + prompt
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            decoded = fragments(tokenizer, ids)
            input_tokens += len(ids)
            with timed(device, timing, "prefill_seconds"):
                chunk = config["prefill_chunk_size"]
                for start in range(0, len(ids), chunk):
                    logits = session.append(ids[start:start+chunk], decoded[start:start+chunk])
                # Reference adaptation: compress at each supplied turn boundary,
                # in addition to the periodic decode schedule.
                session.compress()
            events.append({"turn": turn, "stage": "after_prefill", **session.report()})
            with timed(device, timing, "decode_seconds"):
                text, count, stopped = generate(session, tokenizer, logits, config["max_new_tokens"])
            output_tokens += count
            texts.append(text)
            truncated.append(not stopped)
            events.append({"turn": turn, "stage": "after_decode", **session.report()})
        report = session.report()
    return {
        "sample_id": episode.sample_id, "method": mode, "budget": policy.budget if mode != "full" else None,
        "status": "smoke_only" if smoke else "partial",
        **({"correct": None, "answer_category": "smoke_only"} if smoke else answer_diagnostics(texts[-1], episode.expected, truncated[-1])),
        "first_response_mentions_expected": episode.expected in texts[0],
        "expected": episode.expected, "responses": texts, "truncated_turns": truncated,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        **timing, **report, "events": events,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None,
    }


def source_hash():
    digest = hashlib.sha256()
    for folder in ("Attention", "Backbones", "Dataloaders", "Performance_Metrics"):
        for path in sorted(Path(folder).rglob("*.py")):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_config(config):
    for name in ("num_examples", "max_new_tokens", "prefill_chunk_size"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["warmup_examples"]) is not int or config["warmup_examples"] < 1:
        raise ValueError("at least one warmup example is required")
    if not config["budgets"] or len(set(config["budgets"])) != len(config["budgets"]):
        raise ValueError("budgets must be nonempty and unique")
    for budget in config["budgets"]:
        AgentKVConfig(budget=budget, scoring_profile=config.get("scoring_profile", "reference_v1"), **{k: config[k] for k in ("query_budget", "interval", "sink_tokens", "recent_tokens", "pool_kernel")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Attention/configs/agentkv/pilot.json")
    parser.add_argument("--out", default="artifacts/agentkv-pilot")
    parser.add_argument("--smoke", action="store_true", help="tiny random CPU model; no downloads or quality claims")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.smoke:
        config.update(num_examples=2, distractors=1, budgets=[64], query_budget=8,
                      sink_tokens=4, recent_tokens=4, max_new_tokens=3, interval=2,
                      dtype="float32", model_id="tiny-random-qwen3", model_revision="seed-42")
    validate_config(config)
    if not args.smoke and not torch.cuda.is_available():
        raise SystemExit("GPU pilot requires CUDA. Use --smoke for local CPU checks; no weights were downloaded.")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise SystemExit("Output directory is not empty; use a new path or --resume.")
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.smoke else "cuda"
    identity = {"config": config, "source_sha256": source_hash(), "torch": torch.__version__,
                "transformers": transformers.__version__, "python": platform.python_version(),
                "device": "CPU" if args.smoke else torch.cuda.get_device_name(0)}
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
        raise SystemExit("Config, source, software, or hardware changed; use a new output directory.")
    if not manifest_path.exists():
        git = subprocess.run(["git", "-c", f"safe.directory={Path.cwd().resolve().as_posix()}", "rev-parse", "HEAD"], capture_output=True, text=True)
        snapshot_path = Path("SOURCE_SNAPSHOT.json")
        snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        manifest_path.write_text(json.dumps({"identity": identity, "commit": git.stdout.strip() or (snapshot or {}).get("base_commit", "unknown"),
                                           "source_snapshot": snapshot, "commit_note": "Source hash identifies working-tree implementation; commit may be its base.", "pip_freeze": freeze.stdout,
                                           "cuda_runtime": torch.version.cuda}, indent=2))
    torch.manual_seed(config["seed"])
    if args.smoke:
        torch.set_num_threads(1)
        tiny = transformers.Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=4096, eos_token_id=0)
        tiny._attn_implementation = "sdpa"
        model, tokenizer = transformers.Qwen3ForCausalLM(tiny).eval(), ByteTokenizer()
    else:
        model, tokenizer = load(device, config["dtype"], config["model_id"], config["model_revision"])
    data = list(episodes(config["num_examples"], config["seed"], config["distractors"]))
    warmups = list(episodes(config["warmup_examples"], config["seed"] + 100000, config["distractors"]))
    raw_path = out / "samples.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
    completed = {(r["method"], r["budget"], r["sample_id"]) for r in rows}
    variants = [("full", config["budgets"][0])] + [(m, b) for b in config["budgets"] for m in ("recent", "agentkv")]
    summaries = []
    for mode, budget in variants:
        policy = AgentKVConfig(budget=budget, scoring_profile=config.get("scoring_profile", "reference_v1"), **{k: config[k] for k in ("query_budget", "interval", "sink_tokens", "recent_tokens", "pool_kernel")})
        key_budget = None if mode == "full" else budget
        pending = [e for e in data if (mode, key_budget, e.sample_id) not in completed]
        if pending:
            for episode in warmups:
                run_episode(model, tokenizer, episode, mode, policy, config, args.smoke)
        for episode in pending:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            torch.manual_seed(config["seed"])
            try:
                row = run_episode(model, tokenizer, episode, mode, policy, config, args.smoke)
            except (RuntimeError, ValueError) as error:
                row = {"method": mode, "budget": key_budget, "sample_id": episode.sample_id,
                       "status": "failed", "error": str(error)}
            rows.append(row)
            with raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"{mode} budget={key_budget} {episode.sample_id}: {row['status']}", flush=True)
        selected = [r for r in rows if r["method"] == mode and r["budget"] == key_budget]
        summary = summarize(selected)
        manifest = json.loads(manifest_path.read_text())
        summaries.append({
            "experiment_id": f"agentkv-{mode}-{key_budget}", "paper_id": "s80e8otSxP", "method": mode,
            "commit": manifest["commit"], "backbone_id": config["model_id"], "backbone_revision": config["model_revision"],
            "dataset_id": "synthetic.agent_memory.v1", "split": "synthetic_evaluation", "seed": config["seed"],
            "device": identity["device"], "framework_version": torch.__version__, "dtype": config["dtype"],
            "batch_size": 1, "input_length": "variable; see samples.jsonl", "output_length": "variable; see samples.jsonl",
            "policy": asdict(policy) if mode != "full" else None, **summary,
        })
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"Results: {out.resolve()}")
    if any(r["status"] == "failed" for r in rows):
        raise SystemExit("Some samples failed; inspect samples.jsonl. Resume preserves failures; use a new run to retry.")


if __name__ == "__main__":
    main()
