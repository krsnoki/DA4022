# Short GPU check of where non-finite values first appear when training the
# probe (base in fp16 on a T4). Prepended by build_kernel.py. Env:
#   DEBUG_UIDS: comma list of trace uids to inspect
#   N_STEPS: number of optimiser steps in the mini training run (accum 16)
import glob
import subprocess
import time

# the Kaggle CLI log comes back empty here, so keep our own log in the output
sys.stdout = sys.stderr = open("/kaggle/working/log_debug.txt", "w", buffering=1)
subprocess.run("pip install -q transformers==4.57.0 2>&1 | tail -n 2", shell=True,
               stdout=sys.stdout, stderr=sys.stdout)

import torch  # noqa: E402

from Adaptive_Stopping.experiments.terminator.train_probe import (  # noqa: E402
    MODEL, REVISION, load_jsonl, repeat_kv_in_sdpa, train_step)
from Adaptive_Stopping.methods.learned_exit.terminator.probe import (  # noqa: E402
    TerminatorProbe, base_hidden_states)

T0 = time.time()
dev = torch.device("cuda")
repeat_kv_in_sdpa()
from transformers import AutoModelForCausalLM  # noqa: E402

base = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
for p in base.parameters():
    p.requires_grad_(False)
inner = base.model
print(f"loaded {time.time() - T0:.0f}s", flush=True)

tr_path = sorted(glob.glob("/kaggle/input/**/traces_train.jsonl", recursive=True))[0]
traces = {r["uid"]: r for r in load_jsonl([tr_path])}
uids = os.environ.get("DEBUG_UIDS", "").split(",")


def stats(t):
    t = t.detach()
    fin = torch.isfinite(t)
    bad_tok = (~fin).reshape(t.shape[0], t.shape[1], -1).any(-1)[0].nonzero().flatten().tolist()
    amax = t[fin].abs().max().item() if fin.any() else float("nan")
    return {"max_abs": round(amax, 1), "n_bad_tok": len(bad_tok), "bad_tok": bad_tok[:6]}


rec = {}


def hook(name):
    def fn(m, i, o):
        o = o[0] if isinstance(o, tuple) else o
        rec[name] = stats(o)
        if name.endswith("down_proj"):
            rec[name + ".in"] = stats(i[0])
    return fn


hs = [l.register_forward_hook(hook(f"base.layer{k}")) for k, l in enumerate(inner.layers)]
fresh = TerminatorProbe(base).float().to(dev)
for name, m in [("probe.attn", fresh.layer.self_attn), ("probe.mlp", fresh.layer.mlp),
                ("probe.mlp.down_proj", fresh.layer.mlp.down_proj), ("probe.layer", fresh.layer)]:
    hs.append(m.register_forward_hook(hook(name)))

saved = sorted(glob.glob("/kaggle/input/**/probe.pt", recursive=True))
if saved:
    sd = torch.load(saved[0], map_location=dev)
    diff = max((sd[k].float() - v.float()).abs().max().item() for k, v in fresh.state_dict().items())
    print(f"saved probe.pt vs fresh copy of the last layer: max abs diff {diff:.3g}", flush=True)

for uid in uids:
    tr = traces[uid]
    ids = torch.tensor([tr["prompt_ids"] + tr["think_ids"]], device=dev)
    p0 = len(tr["prompt_ids"])
    rec.clear()
    h = base_hidden_states(base, ids)
    base_rec = {k: v for k, v in rec.items() if k.startswith("base")}
    worst = sorted(base_rec.items(), key=lambda kv: -kv[1]["max_abs"])[:3]
    n_bad_base = sum(v["n_bad_tok"] for v in base_rec.values())
    print(f"\n{uid} T={ids.shape[1]} prompt={p0} base non-finite tokens (all layers) {n_bad_base}; "
          f"largest |h|: {worst}", flush=True)
    for mode in ("fp16_autocast", "fp32"):
        rec.clear()
        x = h if mode == "fp16_autocast" else h.float()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=mode == "fp16_autocast"):
            z = fresh(x)
        zs = stats(z.unsqueeze(-1))
        think_bad = [t - p0 for t in zs["bad_tok"]]
        print(f"  {mode}: logits non-finite tokens {zs['n_bad_tok']} (thinking positions "
              f"{think_bad})", flush=True)
        for k in ("probe.attn", "probe.mlp.down_proj.in", "probe.mlp.down_proj", "probe.mlp",
                  "probe.layer"):
            print(f"    {k}: {rec.get(k)}", flush=True)

# mini training run with the fixed recipe (fp32 probe, fp32 input, no autocast)
for x in hs:
    x.remove()
probe = TerminatorProbe(base).float().to(dev).train()
counter = {"traces": 0, "bad_positions": 0, "traces_with_bad": 0}
opt = torch.optim.AdamW(probe.parameters(), lr=2e-4, weight_decay=0.01)
pool = [t for t in traces.values() if t["think_ids"]][:16 * int(os.environ.get("N_STEPS", "3"))]
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
for k, tr in enumerate(pool):
    n = len(tr["think_ids"])
    tr["labels"] = [0] * (n // 3) + [1] * (n - n // 3)
    loss = train_step(base, probe, tr, 1.0, 1.0, dev, counter)
    assert loss == loss, (tr["uid"], loss)
    if (k + 1) % 16 == 0:
        gn = torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"fp32 step {(k + 1) // 16}: last loss {loss:.4f} grad norm {gn.item():.3g} "
              f"{time.time() - t1:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
              flush=True)
ok = all(torch.isfinite(p).all() for p in probe.parameters())
print(f"weights finite after training: {ok}; hidden-state check {counter}; total {time.time() - T0:.0f}s", flush=True)
