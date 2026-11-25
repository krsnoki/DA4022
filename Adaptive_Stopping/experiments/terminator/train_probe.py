"""Train the TERMINATOR probe on a frozen base model (step C), then write
per-token probabilities for the validation and test traces.

Training follows the paper's appendix B.4 where our data size allows:
AdamW (lr 2e-4, betas 0.9/0.999, eps 1e-8, weight decay 0.01), effective
batch 16 traces, cosine schedule with linear warmup down to 1e-6, 2 epochs,
seed 1337, class-weighted BCE with inverse-frequency weights. Model selection
uses token-level Macro-F1 on the held-out validation traces.

The base model is run once per trace per epoch to get the last-layer hidden
states (no caching to disk); only the probe gets gradients. On GPU the base
runs in fp16 but the probe, its gradients and the optimizer stay in float32
(no autocast): an fp16 probe overflowed at some positions and every step was
skipped. Gradients are clipped to norm 1.0.

Example:
    python Adaptive_Stopping/experiments/terminator/train_probe.py \
        --traces traces_train.jsonl traces_math500.jsonl --labels labels.jsonl \
        --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from Adaptive_Stopping.methods.learned_exit.terminator.labels import token_labels  # noqa: E402
from Adaptive_Stopping.methods.learned_exit.terminator.probe import (  # noqa: E402
    TerminatorProbe,
    base_hidden_states,
    probe_input,
)

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def load_jsonl(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def repeat_kv_in_sdpa():
    """transformers passes enable_gqa=True to torch sdpa when there is no mask.
    Without flash attention (T4) torch then falls back to the math kernel,
    which keeps a T x T matrix per head (about 2.4 GB at 6k tokens in the
    probe's backward pass). Repeating the KV heads instead lets the
    memory-efficient kernel run. The result is the same."""
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:
        return
    if hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


def lr_at(step, total, warmup, peak, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, t)))


def probe_logits(base, probe, trace, device, counter=None):
    """Probe logits over the thinking tokens. The base runs in its own dtype
    (fp16 on GPU); the probe always runs in float32 without autocast."""
    ids = torch.tensor([trace["prompt_ids"] + trace["think_ids"]], device=device)
    h, n_bad = probe_input(base_hidden_states(base, ids))
    if counter is not None:
        counter["traces"] += 1
        counter["bad_positions"] += n_bad
        counter["traces_with_bad"] += int(n_bad > 0)
    logits = probe(h)
    p0 = len(trace["prompt_ids"])
    return logits[0, p0 : p0 + len(trace["think_ids"])]


def weighted_loss(z, y, w0, w1):
    w = torch.where(y > 0.5, torch.full_like(y, w1), torch.full_like(y, w0))
    return (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).mean()


def train_step(base, probe, trace, w0, w1, device, counter=None):
    """Forward and backward for one trace (gradients accumulate). Returns the
    loss as a float; skips the backward pass and returns NaN if not finite."""
    y = torch.tensor(trace["labels"], device=device, dtype=torch.float32)
    loss = weighted_loss(probe_logits(base, probe, trace, device, counter), y, w0, w1)
    if not torch.isfinite(loss):
        return float("nan")
    loss.backward()
    return loss.item()


def macro_f1(tp, fp, fn, tn):
    def f1(tp, fp, fn):
        return 2 * tp / max(1, 2 * tp + fp + fn)

    return 0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp))


@torch.no_grad()
def evaluate(base, probe, traces, device):
    probe.eval()
    tp = fp = fn = tn = 0
    loss_sum, n_tok = 0.0, 0
    for tr in traces:
        y = torch.tensor(tr["labels"], device=device, dtype=torch.float32)
        z = probe_logits(base, probe, tr, device)
        loss_sum += F.binary_cross_entropy_with_logits(z, y, reduction="sum").item()
        n_tok += len(y)
        pred = (torch.sigmoid(z) > 0.5).float()
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
    probe.train()
    return {"macro_f1": macro_f1(tp, fp, fn, tn), "bce": loss_sum / max(1, n_tok),
            "acc": (tp + tn) / max(1, tp + fp + fn + tn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--check-traces", type=int, default=32,
                    help="stop with an error if any of the first N training losses is not finite")
    ap.add_argument("--log-every", type=int, default=5, help="optimizer steps between log lines")
    ap.add_argument("--skip-train", action="store_true", help="only write probabilities")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repeat_kv_in_sdpa()

    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    probe = TerminatorProbe(base).float().to(device).train()
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"probe parameters: {n_params}", flush=True)

    traces = {r["uid"]: r for r in load_jsonl(args.traces)}
    labels = {r["uid"]: r for r in load_jsonl([args.labels])}
    folds = {"train": [], "val": [], "test": []}
    for uid, lab in labels.items():
        tr = traces.get(uid)
        if tr is None:
            continue
        tr["fold"] = lab["fold"]
        tr["label_pos"] = lab["label_pos"]
        if lab["label_pos"] is not None and len(tr["think_ids"]) > 0:
            tr["labels"] = token_labels(len(tr["think_ids"]), lab["label_pos"])
        folds[lab["fold"]].append(tr)
    train = [t for t in folds["train"] if "labels" in t]
    val_lab = [t for t in folds["val"] if "labels" in t]
    if args.max_train:
        train = train[: args.max_train]
    n1 = sum(sum(t["labels"]) for t in train)
    n0 = sum(len(t["labels"]) for t in train) - n1
    w0, w1 = (n0 + n1) / (2 * max(1, n0)), (n0 + n1) / (2 * max(1, n1))
    print(f"train traces {len(train)}, val {len(val_lab)}, n0 {n0}, n1 {n1}, "
          f"w0 {w0:.3f}, w1 {w1:.3f}", flush=True)

    log = {"n_params": n_params, "n_train": len(train), "n_val": len(val_lab),
           "n0": n0, "n1": n1, "w0": w0, "w1": w1, "evals": [], "args": vars(args)}
    best_path = out / "probe.pt"

    if not args.skip_train:
        opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=args.weight_decay)
        total = math.ceil(len(train) / args.accum) * args.epochs
        warmup = min(args.warmup, max(1, total // 10))
        log["total_steps"], log["warmup_steps"] = total, warmup
        step, best = 0, -1.0
        oom, nonfinite_loss, skipped_steps = [], [], 0
        counter = {"traces": 0, "bad_positions": 0, "traces_with_bad": 0}
        eval_every = max(1, len(train) // args.evals_per_epoch)
        t0 = time.time()
        seen = 0
        for epoch in range(args.epochs):
            order = list(range(len(train)))
            random.shuffle(order)
            run_loss, run_n = 0.0, 0
            for k, i in enumerate(order):
                tr = train[i]
                try:
                    loss = train_step(base, probe, tr, w0, w1, device, counter)
                    if math.isfinite(loss):
                        run_loss += loss
                        run_n += 1
                    else:
                        nonfinite_loss.append(tr["uid"])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    oom.append(tr["uid"])
                    print(f"out of memory on {tr['uid']} ({len(tr['think_ids'])} thinking "
                          f"tokens), skipped", flush=True)
                seen += 1
                # fail fast: a non-finite loss early on means the setup is broken
                if nonfinite_loss and seen <= args.check_traces:
                    raise RuntimeError(f"non-finite loss on {nonfinite_loss} in the first "
                                       f"{seen} traces")
                if len(nonfinite_loss) > 0.01 * seen + 5:
                    raise RuntimeError(f"too many non-finite losses: {len(nonfinite_loss)} "
                                       f"of {seen}")
                if k == 0 and epoch == 0:
                    msg = f"first trace loss {loss:.4f}"
                    if device.type == "cuda":
                        msg += f", peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                    print(msg, flush=True)
                last = k == len(order) - 1
                if (k + 1) % args.accum == 0 or last:
                    for g in opt.param_groups:
                        g["lr"] = lr_at(step, total, warmup, args.lr, args.min_lr)
                    gnorm = torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                    if torch.isfinite(gnorm):
                        opt.step()
                    else:
                        skipped_steps += 1
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_every == 0:
                        print(f"epoch {epoch} step {step}/{total} "
                              f"loss {run_loss / max(1, run_n):.4f} grad_norm {gnorm.item():.3g} "
                              f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s",
                              flush=True)
                        run_loss, run_n = 0.0, 0
                if (k + 1) % eval_every == 0 or last:
                    ev = evaluate(base, probe, val_lab, device)
                    ev.update({"epoch": epoch, "step": step, "time_s": time.time() - t0})
                    log["evals"].append(ev)
                    print("val", json.dumps(ev), flush=True)
                    if ev["macro_f1"] > best:
                        best = ev["macro_f1"]
                        torch.save(probe.state_dict(), best_path)
        log["train_time_s"] = time.time() - t0
        log["oom_skipped"] = oom
        log["nonfinite_loss_skipped"] = nonfinite_loss
        log["skipped_optimizer_steps"] = skipped_steps
        log["base_hidden_nonfinite"] = counter
        log["best_val_macro_f1"] = best
        print("base hidden states with non-finite values:", json.dumps(counter), flush=True)

    probe.load_state_dict(torch.load(best_path, map_location=device))
    probe.eval()
    for fold in ("val", "test"):
        with open(out / f"probs_{fold}.jsonl", "w") as f, torch.no_grad():
            for tr in folds[fold]:
                if not tr["think_ids"]:
                    probs = []
                else:
                    probs = torch.sigmoid(probe_logits(base, probe, tr, device)).tolist()
                f.write(json.dumps({"uid": tr["uid"], "probs": [round(p, 4) for p in probs]}) + "\n")
        print(f"wrote probabilities for {len(folds[fold])} {fold} traces", flush=True)
    with open(out / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
