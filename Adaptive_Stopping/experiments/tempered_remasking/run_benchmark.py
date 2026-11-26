"""CPU benchmark on the synthetic anchor-fork model: quality, cost, latency, throughput, memory, GRPO.

Measures, for AR, LC, random, TLC, CT and TCT under identical settings:

A. Quality and cost (Fig. 2/3/4/9 analogues): pass@k, pass@NFE, Best@k with
   self-consistency, Entropy@k, mean NFE, tokens committed per step.
   "Correct" is the minority fork value the anchors bias against (Fig. 1).
B. Systems metrics: single-sample latency (median / p95), batched throughput
   (tokens/s, samples/s), and peak resident memory (one fresh process per method).
   The anchor-fork model's logits are analytic, so each forward pass additionally runs a
   randomly initialised bidirectional transformer over the sequence. Wall-clock
   cost then scales with NFEs, as it does for a real dLLM.
   Energy is read from Linux RAPL counters when the OS exposes them; otherwise it
   is reported as ``not_measured`` (never estimated).
C. Ablations (Fig. 6/7 analogues): TLC ``T_pos`` sweep; ``T_token`` sweep for AR/LC/TLC.
D. Fixed-wall-clock diffu-GRPO (Fig. 5 analogue): a small masked-diffusion
   transformer is warm-started on a bimodal task, then trained with TCT, CT and AR
   rollouts for the same time budget; steps completed, rollout NFE, group reward
   std and greedy-CT evaluation reward are reported.

    python Adaptive_Stopping/experiments/tempered_remasking/run_benchmark.py --out benchmark_results.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

import torch

import common  # noqa: F401  (sets sys.path)
from tempered_remasking import SamplerConfig, build_policy, generate
from tempered_remasking import metrics as M
from tempered_remasking.grpo import DiffuGRPOConfig, DiffuGRPOTrainer
from tempered_remasking.anchor_fork import FORK_B, AnchorForkModel

METHODS = {
    "ar": lambda: build_policy("ar"),
    "lc": lambda: build_policy("lc"),
    "random": lambda: build_policy("random"),
    "tlc": lambda: build_policy("tlc", tpos=1.0),
    "ct": lambda: build_policy("ct", threshold=0.6),
    "tct": lambda: build_policy("tct", threshold=0.6, tpos=0.1),
}
LABELS = {"ar": "AR", "lc": "LC", "random": "Random", "tlc": "TLC (T_pos=1)", "ct": "CT (λ=0.6)",
          "tct": "TCT (T_pos=0.1, λ=0.6)"}


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def build_anchor_fork() -> AnchorForkModel:
    anchors = {6: (0.99, 0.12), 11: (0.985, 0.12), 17: (0.98, 0.1), 23: (0.975, 0.1), 29: (0.97, 0.08)}
    return AnchorForkModel(length=32, fork_position=2, anchors=anchors, filler_confidence=0.9)


class ComputeBoundAnchorFork(torch.nn.Module):
    """Anchor-fork logits plus the forward compute of a small bidirectional transformer."""

    def __init__(self, base: AnchorForkModel, layers: int = 4, d_model: int = 256, heads: int = 4, vocab: int = 32000):
        super().__init__()
        torch.manual_seed(0)
        self.base = base
        self.mask_token_id = base.mask_token_id
        self.emb = torch.nn.Embedding(vocab, d_model)
        layer = torch.nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, batch_first=True, dropout=0.0)
        self.body = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = torch.nn.Linear(d_model, vocab)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        self.head(self.body(self.emb(x)))  # cost of one NFE; output unused
        return self.base(x)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sampler_cfg(model, token_temperature=0.8, block_length=8, confidence_mode="max"):
    return SamplerConfig(gen_length=model.length, block_length=block_length, token_temperature=token_temperature,
                         confidence_mode=confidence_mode)


def sample_problems(base, policy, problems, n, token_temperature, block_length, mode):
    """Per-"problem" (seed) correctness, answers and paper-convention NFEs."""
    correct, answers, nfes, tps = [], [], [], []
    for p in range(problems):
        res = generate(base, base.prompt(n), policy, sampler_cfg(base, token_temperature, block_length, mode),
                       base.mask_token_id, generator=torch.Generator().manual_seed(1000 + p))
        fv = res.generated[:, base.fork_position]
        correct.append((fv == FORK_B).tolist())
        answers.append([str(int(v)) for v in fv])
        nfes.append(res.nfe_paper_convention.tolist())
        tps.append(base.length / res.nfe.double().mean().item())
    return correct, answers, nfes, statistics.mean(tps)


def rapl_energy_uj():
    files = sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"))
    if not files:
        return None
    try:
        return sum(int(open(f).read()) for f in files)
    except OSError:
        return None


def time_generate(model, policy, batch, repeats, cfg, seed=0):
    times = []
    for r in range(repeats):
        g = torch.Generator().manual_seed(seed + r)
        t0 = time.perf_counter()
        res = generate(model, model.base.prompt(batch), policy, cfg, model.mask_token_id, generator=g)
        times.append(time.perf_counter() - t0)
    return times, res


# --------------------------------------------------------------------------- #
# A + C: quality, cost, ablations
# --------------------------------------------------------------------------- #
def quality_table(base, problems, n, mode):
    out = {}
    for name, mk in METHODS.items():
        correct, answers, nfes, tps = sample_problems(base, mk(), problems, n, 0.8, 8, mode)
        curve = M.pass_at_nfe_curve(correct, nfes)
        out[name] = {
            "pass_at_k": {c["k"]: c["pass"] for c in curve},
            "pass_at_nfe": curve,
            "mean_nfe_per_sample": statistics.mean(x for row in nfes for x in row),
            "tokens_per_step": tps,
            "best_at_16_self_consistency": M.best_at_k(answers, [str(FORK_B)] * problems, 16, "self_consistency"),
            "entropy_at_16": M.entropy_at_k(answers, 16),
        }
    return out


def ablations(base, problems, n, mode):
    tpos = {}
    for T in [0.1, 0.5, 1.0, 2.0]:
        c, a, nf, _ = sample_problems(base, build_policy("tlc", tpos=T), problems, n, 0.8, 8, mode)
        tpos[T] = {"pass_at_k": M.pass_at_k_curve(c), "entropy_at_16": M.entropy_at_k(a, 16)}
    ttok = {}
    for name in ["ar", "lc", "tlc"]:
        ttok[name] = {}
        for tt in [0.3, 0.6, 0.8, 1.0, 1.2, 1.5]:
            c, a, nf, _ = sample_problems(base, METHODS[name](), problems, n, tt, 8, mode)
            ttok[name][tt] = {"pass_at_k": M.pass_at_k_curve(c), "entropy_at_16": M.entropy_at_k(a, 16)}
    return {"tlc_tpos": tpos, "token_temperature": ttok}


# --------------------------------------------------------------------------- #
# B: systems metrics
# --------------------------------------------------------------------------- #
def systems_table(latency_repeats, throughput_batch, throughput_repeats):
    model = ComputeBoundAnchorFork(build_anchor_fork())
    cfg = sampler_cfg(model.base)
    out = {}
    for name, mk in METHODS.items():
        pol = mk()
        time_generate(model, pol, 1, 2, cfg)  # warm-up
        lat, _ = time_generate(model, pol, 1, latency_repeats, cfg, seed=10)
        lat_ms = sorted(1000 * t for t in lat)
        e0 = rapl_energy_uj()
        thr, res = time_generate(model, pol, throughput_batch, throughput_repeats, cfg, seed=100)
        e1 = rapl_energy_uj()
        tot = sum(thr)
        n_samples = throughput_batch * throughput_repeats
        out[name] = {
            "latency_median_ms": statistics.median(lat_ms),
            "latency_p95_ms": lat_ms[min(len(lat_ms) - 1, int(math.ceil(0.95 * len(lat_ms))) - 1)],
            "throughput_tokens_per_s": n_samples * model.base.length / tot,
            "throughput_samples_per_s": n_samples / tot,
            "mean_nfe_per_sample_batched": res.nfe.double().mean().item(),
            "energy_j_per_sample": ((e1 - e0) / 1e6 / n_samples) if (e0 is not None and e1 is not None and e1 >= e0) else None,
            "peak_memory": memory_in_subprocess(name, throughput_batch),
        }
        print(f"  {name:>7}: latency {out[name]['latency_median_ms']:.1f} ms (p95 {out[name]['latency_p95_ms']:.1f}), "
              f"{out[name]['throughput_tokens_per_s']:.0f} tok/s, peak RSS {out[name]['peak_memory']['peak_rss_mb']:.0f} MB "
              f"(+{out[name]['peak_memory']['generation_delta_mb']:.1f} MB)")
    return out


def memory_in_subprocess(name, batch):
    cmd = [sys.executable, os.path.abspath(__file__), "--memory-worker", name, "--memory-batch", str(batch),
           "--threads", str(torch.get_num_threads())]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def _current_rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def memory_worker(name, batch):
    """Peak RSS while generating, sampled every 1 ms in a background thread."""
    import threading

    model = ComputeBoundAnchorFork(build_anchor_fork())
    pol = METHODS[name]()
    cfg = sampler_cfg(model.base)
    baseline = _current_rss_mb()
    peak = [baseline]
    done = threading.Event()

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], _current_rss_mb())
            time.sleep(0.001)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    generate(model, model.base.prompt(batch), pol, cfg, model.mask_token_id, generator=torch.Generator().manual_seed(0))
    done.set()
    th.join()
    peak[0] = max(peak[0], _current_rss_mb())
    lifetime = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB -> MiB (Linux)
    print(json.dumps({"peak_rss_mb": max(lifetime, peak[0]), "generation_delta_mb": peak[0] - baseline,
                      "baseline_rss_mb": baseline, "batch": batch}))


# --------------------------------------------------------------------------- #
# D: fixed-budget diffu-GRPO
# --------------------------------------------------------------------------- #
G_MASK, G_EOS, G_VOCAB, G_PROMPT, G_LEN = 0, 1, 10, 4, 16
PATTERN_A = [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, G_EOS, G_EOS, G_EOS, G_EOS]   # majority mode (p = 0.8)
PATTERN_B = [6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, G_EOS, G_EOS, G_EOS, G_EOS]   # minority mode = rewarded
G_PROMPT_IDS = torch.tensor([9, 8, 7, 6])


class SmallDLM(torch.nn.Module):
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.emb = torch.nn.Embedding(G_VOCAB, d)
        self.pos = torch.nn.Parameter(torch.randn(G_PROMPT + G_LEN, d) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0)
        self.body = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = torch.nn.Linear(d, G_VOCAB)

    def forward(self, x):
        logits = self.head(self.body(self.emb(x) + self.pos[: x.shape[1]]))
        return logits.masked_fill(torch.arange(G_VOCAB, device=x.device) == G_MASK, -1e4)


def warm_start(steps, seed):
    """Masked-diffusion pre-training on the bimodal data (80% pattern A, 20% B)."""
    torch.manual_seed(seed)
    model = SmallDLM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    pa, pb = torch.tensor(PATTERN_A), torch.tensor(PATTERN_B)
    for _ in range(steps):
        B = 64
        comp = torch.where((torch.rand(B, 1) < 0.2), pb, pa)
        x0 = torch.cat([G_PROMPT_IDS.expand(B, -1), comp], 1)
        t = torch.rand(B, 1).clamp_min(1e-3)
        m = torch.rand(B, G_LEN) < t
        m = torch.cat([torch.zeros(B, G_PROMPT, dtype=torch.bool), m], 1)
        xt = x0.masked_fill(m, G_MASK)
        ce = torch.nn.functional.cross_entropy(model(xt).transpose(1, 2), x0, reduction="none")
        loss = ((ce * m) / t).sum() / (B * G_LEN)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def reward_b(tokens, ex):
    return sum(int(a == b) for a, b in zip(tokens, PATTERN_B)) / G_LEN


@torch.no_grad()
def eval_greedy_ct(model, threshold, n=32):
    cfg = SamplerConfig(gen_length=G_LEN, block_length=8, token_temperature=0.0)
    res = generate(model.eval(), G_PROMPT_IDS.expand(n, -1), build_policy("ct", threshold=threshold), cfg, G_MASK, [G_EOS])
    return statistics.mean(reward_b(c.tolist(), None) for c in res.generated)


@torch.no_grad()
def eval_sampled(model, mode, n=256, seed=0):
    cfg = SamplerConfig(gen_length=G_LEN, block_length=8, token_temperature=0.8, confidence_mode=mode)
    res = generate(model.eval(), G_PROMPT_IDS.expand(n, -1), build_policy("tct", threshold=0.6, tpos=0.1), cfg, G_MASK,
                   [G_EOS], torch.Generator().manual_seed(seed))
    return statistics.mean(reward_b(c.tolist(), None) for c in res.generated)


def grpo_budget(budget_s, warm_steps, seed, mode):
    base = warm_start(warm_steps, seed)
    init = {"greedy_ct": {lam: eval_greedy_ct(base, lam) for lam in (0.6, 0.8, 1.0)}, "sampled_tct_reward": eval_sampled(base, mode)}
    out = {"initial": init}
    for sampler in ["tct", "ct", "ar"]:
        model = copy.deepcopy(base)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = DiffuGRPOConfig(num_generations=8, num_iterations=1, time_budget_s=budget_s, rollout_batch_size=8,
                              beta=0.04)
        tr = DiffuGRPOTrainer(model, None, opt, METHODS[sampler](),
                              SamplerConfig(gen_length=G_LEN, block_length=8, token_temperature=0.8, confidence_mode=mode),
                              reward_b, G_MASK,
                              [G_EOS], cfg, torch.Generator().manual_seed(seed))
        hist = tr.train(iter(lambda: {"prompt_ids": G_PROMPT_IDS}, None))
        out[sampler] = {
            "steps_completed": len(hist),
            "rollout_nfe_mean": statistics.mean(h["rollout_nfe_mean"] for h in hist),
            "rollout_time_fraction": sum(h["rollout_time_s"] for h in hist) / hist[-1]["wall_time_s"],
            "reward_std_mean": statistics.mean(h["reward_std"] for h in hist),
            "frac_zero_std_groups": statistics.mean(h["frac_zero_std_groups"] for h in hist),
            "greedy_ct_reward": {lam: eval_greedy_ct(model, lam) for lam in (0.6, 0.8, 1.0)},
            "sampled_tct_reward": eval_sampled(model, mode),
        }
        r = out[sampler]
        print(f"  [{mode}] {sampler:>4}: {r['steps_completed']} steps in {budget_s:.0f}s, rollout NFE {r['rollout_nfe_mean']:.1f}, "
              f"reward std {r['reward_std_mean']:.3f}, greedy-CT reward {r['greedy_ct_reward']}")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=50)
    ap.add_argument("--n", type=int, default=64, help="samples per problem")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--latency-repeats", type=int, default=30)
    ap.add_argument("--throughput-batch", type=int, default=64)
    ap.add_argument("--throughput-repeats", type=int, default=3)
    ap.add_argument("--grpo-budget-s", type=float, default=60.0)
    ap.add_argument("--grpo-warm-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sections", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--memory-worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--memory-batch", type=int, default=64, help=argparse.SUPPRESS)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.memory_worker:
        memory_worker(args.memory_worker, args.memory_batch)
        return

    base = build_anchor_fork()
    out = {"env": {"torch": torch.__version__, "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
                   "threads": args.threads, "rapl_available": rapl_energy_uj() is not None},
           "settings": {"gen_length": base.length, "block_length": 8, "token_temperature": 0.8, "problems": args.problems,
                        "samples_per_problem": args.n}}
    if "A" in args.sections:
        print("A. quality and cost")
        out["quality"] = {mode: quality_table(base, args.problems, args.n, mode) for mode in ("max", "sampled")}
        for mode, tab in out["quality"].items():
            for name, r in tab.items():
                print(f"  [{mode}] {name:>7}: pass@1 {r['pass_at_k'].get(1, math.nan):.3f} pass@4 {r['pass_at_k'].get(4, math.nan):.3f} "
                      f"pass@16 {r['pass_at_k'].get(16, math.nan):.3f} pass@64 {r['pass_at_k'].get(64, math.nan):.3f} "
                      f"NFE {r['mean_nfe_per_sample']:.1f} tok/step {r['tokens_per_step']:.2f} "
                      f"Best@16 {r['best_at_16_self_consistency']:.3f} Entropy@16 {r['entropy_at_16']:.3f}")
    if "B" in args.sections:
        print("B. systems metrics")
        out["systems"] = systems_table(args.latency_repeats, args.throughput_batch, args.throughput_repeats)
    if "C" in args.sections:
        print("C. ablations")
        out["ablations"] = ablations(base, args.problems, args.n, "max")
        for T, r in out["ablations"]["tlc_tpos"].items():
            print(f"  TLC T_pos={T}: pass@1 {r['pass_at_k'].get(1, math.nan):.3f} pass@16 {r['pass_at_k'].get(16, math.nan):.3f} "
                  f"Entropy@16 {r['entropy_at_16']:.3f}")
        for name, d in out["ablations"]["token_temperature"].items():
            print(f"  {name}: " + "  ".join(f"T={tt}: p@1 {r['pass_at_k'].get(1, math.nan):.3f} p@16 {r['pass_at_k'].get(16, math.nan):.3f}"
                                         for tt, r in d.items()))
    if "D" in args.sections:
        print("D. fixed-budget diffu-GRPO")
        out["grpo"] = {mode: grpo_budget(args.grpo_budget_s, args.grpo_warm_steps, args.seed, mode) for mode in ("max", "sampled")}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
