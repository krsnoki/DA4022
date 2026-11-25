"""Audit an AgentKV cloud export without importing torch or downloading models."""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(results_zip, source_zip, destination):
    # Read only explicitly named files; do not extract arbitrary archive paths.
    with zipfile.ZipFile(results_zip) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        summaries = json.loads(archive.read("summary.json"))
        raw = archive.read("samples.jsonl")
        rows = [json.loads(line) for line in raw.decode().splitlines()]
    identity = manifest["identity"]
    config = identity["config"]
    digest = hashlib.sha256()
    with zipfile.ZipFile(source_zip) as source:
        names = source.namelist()
        for folder in ("Attention", "Backbones", "Dataloaders", "Performance_Metrics"):
            for name in sorted(n for n in names if n.startswith(folder + "/") and n.endswith(".py")):
                digest.update(name.encode())
                digest.update(source.read(name))
    require(digest.hexdigest() == identity["source_sha256"], "Cloud source differs from supplied source bundle")
    variants = {("full", None)} | {(m, b) for m in ("recent", "agentkv") for b in config["budgets"]}
    expected_ids = {f"agent_memory_v1:{config['seed']}:{i}" for i in range(config["num_examples"])}
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["method"], row["budget"]].append(row)
    require(set(groups) == variants, "Missing or unexpected configurations")
    checks = []
    for (method, budget), group in groups.items():
        require(len(group) == len(expected_ids) and {r["sample_id"] for r in group} == expected_ids,
                f"Missing or duplicate samples: {method}/{budget}")
        require(all(r["status"] == "partial" for r in group), "Failed or non-pretrained samples in export")
        for row in group:
            text = row["responses"][-1]
            answer = text.rsplit("</think>", 1)[-1].replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            correct = "</think>" in text and answer == row["expected"]
            require(correct == row["correct"], "Recorded correctness differs from evaluator")
            require(all(math.isfinite(row[k]) and row[k] >= 0 for k in
                        ("prefill_seconds", "decode_seconds", "peak_memory_mb", "eviction_seconds")), "Invalid measurement")
            if method == "full":
                require(row["cached_tokens"] == row["logical_tokens"] and row["compression_events"] == 0, "Full cache unexpectedly compressed")
            else:
                require(row["compression_events"] > 0, "Compressed policy never evicted")
                for event in row["events"]:
                    if event["stage"] == "after_prefill":
                        require(event["cached_tokens"] <= budget, "Turn-boundary cache exceeds budget")
                require(row["cached_tokens"] < budget + config["interval"], "Final cache violates interval growth bound")
        matches = [s for s in summaries if s["method"] == method and (s["policy"] or {}).get("budget") == budget]
        require(len(matches) == 1, "Missing or duplicate summary")
        summary = matches[0]
        accuracy = sum(r["correct"] for r in group) / len(group)
        throughput = sum(r["output_tokens"] for r in group) / sum(r["decode_seconds"] for r in group)
        latency = statistics.median(r["prefill_seconds"] + r["decode_seconds"] for r in group) * 1000
        for key, value in (("quality_value", accuracy), ("throughput", throughput), ("latency_median_ms", latency),
                           ("peak_memory_mb", max(r["peak_memory_mb"] for r in group)),
                           ("kv_bytes_max_final", max(r["kv_bytes"] for r in group))):
            require(math.isclose(summary[key], value, rel_tol=1e-9, abs_tol=1e-9), f"Summary mismatch: {key}")
        require(summary["failures"] == 0 and summary["num_examples"] == len(group), "Invalid summary counts")
        checks.append({"method": method, "budget": budget, "correct": sum(r["correct"] for r in group),
                       "total": len(group), "truncated_final_turns": sum(r["truncated_turns"][-1] for r in group),
                       "failed_sample_ids": [r["sample_id"] for r in group if not r["correct"]]})
    require(len(summaries) == len(variants), "Unexpected extra summaries")
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (out / "samples.jsonl").write_bytes(raw)
    metadata = {k: v for k, v in manifest.items() if k != "pip_freeze"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    audit_result = {"source_matches_uploaded_bundle": True, "result_zip_sha256": hashlib.sha256(Path(results_zip).read_bytes()).hexdigest(),
                    "record_count": len(rows), "configuration_count": len(variants), "checks": checks,
                    "note": "Validates exported records; does not rerun model inference or certify algorithm fidelity. Original manifest with pip freeze remains in the results ZIP."}
    (out / "audit.json").write_text(json.dumps(audit_result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit_result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="artifacts/agentkv-results.zip")
    parser.add_argument("--source", default="artifacts/agentkv-source.zip")
    parser.add_argument("--out", default="Performance_Metrics/results/agentkv_pilot")
    args = parser.parse_args()
    audit(args.results, args.source, args.out)
