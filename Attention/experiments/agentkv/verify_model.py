"""Pretrained integration gate; tiny random CPU mode available without downloads."""

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from Attention.methods.agentkv import AgentKVConfig
from Attention.methods.agentkv.qwen3 import Qwen3Session
from Backbones.Generative_Pretrained.qwen3 import load
from Backbones.Generative_Pretrained.qwen3 import REVISION


@torch.inference_mode()
def verify(model, chunks, text_chunks, atol=1e-5, rtol=1e-5):
    """Compare same chunk/token forwards, including caches, against untouched HF.

    Fixed teacher-forced tokens isolate numerical integration from diverging
    generated sequences. Tolerances apply to logits and cached K/V, not quality.
    """
    device = next(model.parameters()).device
    total = sum(len(chunk) for chunk in chunks)
    reference = []
    cache = DynamicCache()
    for chunk in chunks:
        result = model(input_ids=torch.tensor([chunk], device=device),
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        reference.append(result.logits[0, -1].clone())
    references = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    del result, cache
    checks = []
    for mode in ("full", "recent", "agentkv"):
        cfg = AgentKVConfig(budget=max(64, total + 1), query_budget=8,
                            sink_tokens=4, recent_tokens=4, interval=2)
        with Qwen3Session(model, mode, cfg) as session:
            maximum_error = 0.0
            for chunk, text, target in zip(chunks, text_chunks, reference, strict=True):
                actual = session.append(chunk, text, decode=len(chunk) == 1)
                torch.testing.assert_close(actual, target, atol=atol, rtol=rtol)
                if int(actual.argmax()) != int(target.argmax()):
                    raise AssertionError(f"{mode}: no-eviction next-token argmax differs")
                maximum_error = max(maximum_error, float((actual.float() - target.float()).abs().max()))
            for layer, (keys, values) in zip(session.cache.layers, references, strict=True):
                torch.testing.assert_close(layer.keys, keys, atol=atol, rtol=rtol)
                torch.testing.assert_close(layer.values, values, atol=atol, rtol=rtol)
            assert session.compressions == 0
            checks.append({"mode": mode, "max_logit_absolute_error": maximum_error, "cache_matches": True})
    # Forced compaction checks physical state and continuation, not equality to full cache.
    cfg = AgentKVConfig(budget=8, query_budget=8, sink_tokens=1, recent_tokens=1,
                        interval=2, scoring_profile="upstream_mean")
    with Qwen3Session(model, "agentkv", cfg) as session:
        for chunk, text in zip(chunks, text_chunks, strict=True):
            logits = session.append(chunk, text, decode=len(chunk) == 1)
            if not torch.isfinite(logits).all():
                raise AssertionError("nonfinite logits after cache compaction")
            session.compress()
        report = session.report()
        assert report["logical_tokens"] == total
        assert report["cached_tokens"] == min(total, cfg.budget)
        assert report["compression_events"] > 0
        for positions in session.positions.values():
            assert (positions[:, -1] == total - 1).all()
            assert (positions[:, 1:] > positions[:, :-1]).all()
        phases = report["representatives_by_phase"]
        assert all(all(count > 0 for count in layer.values()) for layer in phases.values())
    return {"passed": True, "atol": atol, "rtol": rtol, "no_eviction": checks,
            "compressed_continuation": report,
            "scope": "Same-input numerical checks and finite compressed continuation; not pretrained task accuracy or proof of algorithm fidelity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="artifacts/agentkv-next/verification.json")
    args = parser.parse_args()
    if args.smoke:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval()
        chunks = [list(range(1, 13)), [13, 14, 15, 16], [17], [18]]
        texts = [["<think>"] * 3 + ["<tool_call>"] * 3 + ["<tool_response>"] * 3 + ["<|im_end|>"] * 3,
                 ["<think>", "reason", "</think>", "answer"], ["x"], ["y"]]
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required; use --smoke for local checks. No weights downloaded.")
        model, tokenizer = load()
        prompts = ["<|im_start|>system\nRemember the code.<|im_end|>\n<|im_start|>assistant\n<think>Look up the order.</think><tool_call>{\"name\":\"lookup\"}</tool_call><|im_end|>\n<|im_start|>tool\n<tool_response>code: 123456</tool_response><|im_end|>\n",
                   "<|im_start|>user\nWhat was the code?<|im_end|>\n<|im_start|>assistant\n<think>\n"]
        chunks = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
        chunks += [[token] for token in tokenizer.encode("The code", add_special_tokens=False)]
        texts = [[tokenizer.decode([token], skip_special_tokens=False) for token in chunk] for chunk in chunks]
    report = verify(model, chunks, texts, atol=1e-5 if args.smoke else 0.02, rtol=1e-5 if args.smoke else 0.01)
    from Attention.experiments.agentkv.run import source_hash
    report.update(smoke=args.smoke, model_id="tiny-random-qwen3" if args.smoke else model.config._name_or_path,
                  model_revision="seed-42" if args.smoke else REVISION, source_sha256=source_hash(),
                  torch=torch.__version__, device=str(next(model.parameters()).device))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Integration gate passed. Saved {out}")


if __name__ == "__main__":
    main()
