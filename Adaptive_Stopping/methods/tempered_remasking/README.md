# Paper implementation card: A Tale of Two Temperatures — Simple, Efficient, and Diverse Sampling from Diffusion Language Models

- **Paper / OpenReview:** Olausson, Jazbec, Wang, Solar-Lezama, Naesseth, Mandt, Nalisnick — *Proceedings of the 43rd ICML*, PMLR 306, 2026. PDF: `20_A_Tale_of_Two_Temperatures_.pdf` (repository root). Listed in the [AdaptFM accepted-paper index](https://adaptfm.gitlab.io/accepted-papers).
- **Primary module:** `Adaptive_Stopping/` (adaptive decoding policy: what to commit at each denoising step, and how many NFEs a sample costs)
- **Secondary modules:** `Performance_Metrics/` (pass@k, pass@NFE, Best@NFE), `Backbones/Generative_Pretrained/` (LLaDA-8B-Instruct, Dream-v0-Instruct-7B), `LoRA/` (d1 trains LoRA adapters for GRPO), `Dataloaders/` (GSM8K, MATH-500, HumanEval, MBPP)
- **Owner / reviewer:** `Adaptive_Stopping/` module
- **Status:** `tested`. All algorithms, the Appendix D theory, metrics, grading and the GRPO loop are covered by 55 CPU tests; the theory is reproduced end to end on an exact anchor-fork model. LLaDA/Dream numbers in this card are paper-reported.
- **Upstream code / licence:** no official code release found in the paper. LLaDA's `generate.py` and Fast-dLLM are the reference behaviour for LC/CT; d1 is the reference for diffu-GRPO.
- **Backbone IDs:** `GSAI-ML/LLaDA-8B-Instruct`, `Dream-org/Dream-v0-Instruct-7B` (the `revision` field in `paper.yaml` is passed to `from_pretrained`)
- **Dataset IDs and split:** `openai/gsm8k` main/test (first 300, see below), `HuggingFaceH4/MATH-500` test, `openai/openai_humaneval` test, `google-research-datasets/mbpp` full/test
- **Hardware assumptions:** 8B dLLM in bf16, about 16 GB of accelerator memory plus activations. The paper's GRPO runs used 4×A100 for 72 h per run. Everything in `tests/` and `run_toy_theory.py` runs on CPU.

## Research summary

### Problem
Masked diffusion LMs choose which masked positions to unmask at each step. Greedy confidence heuristics (low-confidence remasking, LC; confidence thresholding, CT/Fast-dLLM) give good single samples and, for CT, adaptive speed, but they **collapse diversity** (flat pass@k). They defer high-entropy *fork* tokens until confident *anchor* tokens have already fixed the solution, a phenomenon Ni et al. (2026) call entropy degradation. Autoregressive (left-to-right) decoding recovers diversity but gives up parallel decoding.

### Method
A second temperature, the **position temperature `T_pos`**, controls how greedily the unmasking *order* is chosen. It complements `T_token`, which controls *what* is sampled at each position.

| Policy | Rule | Limits |
|---|---|---|
| **TLC** (Alg. 2) | Draw `K` masked positions without replacement from `Cat(c̃)`, where `c̃_k ∝ c_k^{1/T_pos}` | `T_pos→0`: LC; `T_pos→∞`: random remasking |
| **TCT** (Alg. 4) | Include each masked position independently with `b_k ~ Ber(σ((c_k − λ)/T_pos))` | `T_pos→0`: CT (`c > λ`); `T_pos→∞`: Ber(½) |

The confidence is `c_k = q_θ^k(x_k | x_t)`, the untempered probability of the sampled token (footnote 1).

**Theory (App. D).** Fork tokens are defined operationally (Def. D.1). Assumption D.2 states the anchor-fork degeneracy. Lemma D.3 shows `σ(δ_a/T_pos) ≤ P(T(a) < T(ℓ)) ≤ σ(Δ_a/T_pos)` under TLC with `K = 1`. From this, Prop. D.4 proves that any `T'_pos > T_pos·Δ/δ` strictly increases expected fork entropy. Remark D.5 notes that LC is the minimum and any `T_pos > 0` beats it. Cor. D.7 says semantic entropy also rises when the gain exceeds `ε + δ`. Prop. D.8 shows `T_token` cannot fix LC's order, and Prop. D.9 shows AR beats LC when some anchor follows the fork.

### Expected behaviour (paper-reported)
- TLC (`T_pos = 1`) closely tracks AR pass@k on GSM8K/MATH/HumanEval/MBPP for LLaDA and Dream, except Dream on HumanEval (Fig. 2).
- TCT (`T_pos = 0.1, λ = 0.6`) has slightly lower pass@k than AR but **higher pass@NFE**. On MATH with LLaDA it needs about 4000 NFEs to exceed 70%, versus about 8000 for AR (Fig. 3).
- Test-time compute: TCT loses to AR under self-consistency at high NFE but improves the frontier with an ORM (AceMath-7B) (Fig. 4).
- GRPO (d1) at a fixed 72 h budget: TCT completes about 44K steps versus about 21K for AR on GSM8K, keeps the highest within-group reward std, and evaluates comparably to or better than AR/Fast-dLLM (Fig. 5).

## Implementation

| File | Paper content |
|---|---|
| `remasking.py` | Alg. 1–4 (LC, TLC, CT, TCT) plus random and AR baselines; token sampling with `T_token`; the confidence definitions; LLaDA's fixed-`K` schedule. TLC uses the exact Gumbel-top-`k` form of sampling without replacement. |
| `sampler.py` | First-hit sampler with semi-AR blocks (L = 256, blocks of 32); NFE accounting per footnote 3; AR EOS stopping; optional NFE budget; `DecodingStopPolicy` implementing the module's `reset / should_stop / report` contract. |
| `theory.py` | Def. D.1, Eq. (5), Assump. D.2, Lemma D.3, Prop. D.4, Remark D.5, Cor. D.7, Prop. D.8, Prop. D.9 as executable functions. |
| `toy.py` | A toy dLLM that satisfies the anchor-fork degeneracy *exactly* (the Fig. 1 scenario), with an exact dynamic program for TLC fork statistics. |
| `metrics.py` | pass@k (unbiased), pass@NFE, Best@k / Best@NFE (self-consistency and ORM), Entropy@k, GRPO group-reward stats. |
| `tasks.py` | Prompt formats; GSM8K/MATH answer extraction and grading; HumanEval/MBPP execution with timeouts. |
| `orm.py` | ORM scorer (`nvidia/AceMath-7B-RM`). |
| `grpo.py` | d1 diffu-GRPO: one-step log-likelihood with prompt masking (`p_mask = 0.15`), clipped surrogate with k3 KL, pluggable rollout sampler, wall-clock budget. |
| `models.py` | LLaDA / Dream adapters (Dream's one-position logit shift) and a Hugging Face loader. |

- **Input contract:** `model(ids[B, N]) -> logits[B, N, V]` where `logits[:, k]` parameterises `q_θ^k(·|x_t)`; prompt ids `[P]` or `[B, P]`; `mask_token_id`; optional EOS ids.
- **Output contract:** `GenerationResult` contains `sequences [B, P+L]` (long), `nfe [B]`, `nfe_paper_convention [B]`, `stop_reasons`, `unmask_step [B, P+L]` (gives `T(j)`), `confidence_at_unmask`, optional `entropy_at_unmask`, `tokens_per_step [S, B]`, stats, and config.
- **Configuration:** `Adaptive_Stopping/configs/tempered_remasking/paper.yaml`. It contains every paper setting, with assumptions marked.
- **Determinism:** stochastic whenever `T_pos > 0` or `T_token > 0`, and reproducible with a seeded `torch.Generator`. It is deterministic only for LC/CT/AR with `T_token = 0` (`stats["stop_policy"]["deterministic"]`). The policies need **no extra forward pass and no trained weights**; confidences come from the forward pass the sampler already runs.

### Usage

```python
import sys; sys.path.insert(0, "Adaptive_Stopping/methods")
import torch
from tempered_remasking import SamplerConfig, build_policy, generate
from tempered_remasking.models import load_hf_dllm

h = load_hf_dllm("llada")                                  # needs transformers + GPU
prompt = h.encode_chat("What is 17 * 23? Put the answer in \\boxed{}.", torch.device("cuda"))
res = generate(h.model_fn, prompt.unsqueeze(0).expand(8, -1),   # 8 samples
               build_policy("tct", tpos=0.1, threshold=0.6),
               SamplerConfig(gen_length=256, block_length=32, token_temperature=0.8),
               h.mask_token_id, h.eos_token_ids)
texts = [h.decode(s.tolist()) for s in res.generated]; print(res.nfe)
```

### Reproduction commands

```bash
# CPU: tests (55) and the formal model end to end (~3 min)
python -m pytest Adaptive_Stopping/tests -q
python Adaptive_Stopping/experiments/tempered_remasking/run_toy_theory.py --out toy_results.json

# GPU: Fig. 2/3/6/7/8 data (one file per model × dataset × preset)
for p in ar lc random tlc ct tct; do
  python Adaptive_Stopping/experiments/tempered_remasking/run_sampling.py \
    --model llada --dataset gsm8k --preset $p --n-samples 96 --out results/llada_gsm8k_$p.jsonl
done
# Fig. 6 (T_pos ablation): --preset tlc --tpos {0.1,0.5,1,2} on humaneval
# Fig. 7 (T_token ablation): --token-temperature {0.3,0.6,0.8,1.0,1.2,1.5}
# Fig. 4/9/10: ORM scores, then Best@NFE / Entropy@k
python Adaptive_Stopping/experiments/tempered_remasking/score_orm.py --in results/llada_gsm8k_tct.jsonl --out results/llada_gsm8k_tct.orm.jsonl
python Adaptive_Stopping/experiments/tempered_remasking/analyze.py results/*.jsonl --json summary.json --csv rows.csv
# Fig. 5: diffu-GRPO at a fixed wall-clock budget, then greedy CT eval at λ ∈ {0.6, 0.8, 1.0}
python Adaptive_Stopping/experiments/tempered_remasking/run_grpo.py --dataset gsm8k --rollout tct --hours 72 --out runs/gsm8k_tct
```

`analyze.py --csv` writes rows in the `Performance_Metrics/README_metrics.md` schema, with `energy_j` left empty (not measured) and a `status` label chosen with `--status` (default `reproduced`).

## Results

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported (LLaDA-8B, MATH-500) | pass@NFE > 70% at ≈4000 NFEs (TCT) vs ≈8000 (AR) | — | — | — | — | Fig. 3; L = 256, blocks of 32, `T_token = 0.8` |
| Paper-reported (LLaDA-8B, GSM8K GRPO) | ≈44K (TCT) vs ≈21K (AR) GRPO steps in 72 h | — | — | — | — | Fig. 5; 4×A100 |
| Local baseline / method (LLaDA, Dream) | not reproduced | — | — | — | not_measured | 8B benchmarks require a GPU; produced by the commands above |
| Local, toy anchor-fork model (CPU) | see below | — | — | — | not_measured | `run_toy_theory.py`; measured, toy |

**Toy model results (locally measured, `run_toy_theory.py`, L = 32, fork at position 2, 5 anchors, `H₀ = ln 2`, `Ση = 0.52`):**

- Expected fork entropy under TLC (exact DP vs Monte Carlo through `generate`) rises monotonically with `T_pos`: 0.180 (`T_pos = 0.02`), 0.248 (0.1), 0.391 (1), 0.433 (∞). This lies strictly between LC (0.173, the Remark D.5 minimum) and AR (0.693, Prop. D.9 with all anchors after the fork). The Lemma D.3 bounds hold at every `T_pos`.
- Without blocks and with max-prob confidence: CT gives fork entropy 0.173 and semantic entropy 0.173 in 2 NFEs, while TCT (`T_pos = 0.1`, `λ = 0.6`) gives 0.320 and 0.455 in 2 NFEs.
- Toy pass@k, where the "correct" answer is the minority solution the anchors bias against (blocks of 8, `T_token = 0.8`):
  - **max-prob confidence** (the setting of the formal analysis): pass@4 is LC 0.62, TLC 0.79, CT 0.64 at 20 NFEs, TCT 0.76 at 23 NFEs, and AR 0.93 at 128 NFEs. At a matched cost of about 23–32 NFEs, TCT pass@4 (0.76) beats AR pass@1 (0.50).
  - **sampled-token confidence** (footnote 1, the paper's experimental setting): TLC still beats LC (pass@4 0.72 vs 0.61), but **TCT trails CT** (0.48 vs 0.53). In this toy, a fork that drew its minority value has lower *sampled-token* confidence, so TCT's Bernoulli inclusion commits it less often. CT is less exposed because it defers the fork until the argmax fallback. This is a property of the toy and need not transfer to real dLLMs; `confidence_mode` (`sampled` or `max`) selects between the two settings on LLaDA/Dream.

## Differences and limitations

- **8B-model numbers are paper-reported.** The implementation environment was CPU-only, so the LLaDA/Dream/GRPO rows above come from the paper. The full experiment pipeline for them is included, and its analysis stage is tested on synthetic run files.
- **Settings the paper does not state** (all configurable, marked `assumption` in `paper.yaml`):
  - `K = 1` token per step for LC/TLC/random (`steps = L`).
  - Which 300 GSM8K test problems were used (first 300 here).
  - Prompts and answer-matching code (MATH grading here is string-normalised, not symbolic).
  - The MBPP split.
  - d1 hyper-parameters: `G`, `μ`, learning rate, LoRA rank/targets, and d1's format rewards (only the correctness reward is implemented).
- **Empty TCT steps.** If TCT's Bernoulli draws select nothing, the default (`empty_selection="resample"`) redraws from the *same* logits. Because the model is deterministic given `x_t`, this is distributionally identical to literal Algorithm 4 conditioned on progress, but it wastes no NFE. `"skip"` reproduces the literal algorithm, including the wasted NFEs. CT uses the Fast-dLLM argmax fallback, so with `λ = 1.0` it decodes one token per step.
- **AR NFEs** follow footnote 3 (`nfe_paper_convention`: tokens before `<eos>`); `nfe` is the forward passes actually run (one more when stopped at EOS).
- **No KV caching** (Fast-dLLM's cache is orthogonal; NFE counts do not depend on it). Latency from `run_sampling.py` is therefore not a Fast-dLLM speed claim.
- **Theory scope:** as in the paper, the formal results cover TLC at `K = 1` without blocks. The toy uses max-prob confidence so that Assumption D.2's confidence gap holds deterministically.
- **Code execution:** HumanEval/MBPP grading runs model-generated code (`python -I`, timeout). Run it only inside a sandbox.
- **Dataset loading** is self-contained in `experiments/tempered_remasking/common.py` (Hugging Face `datasets`), so the module has no external data dependencies beyond the dataset IDs above.

## Reproduction checklist

- [x] Paper and version identified (ICML 2026, PMLR 306; AdaptFM accepted-paper index)
- [x] Base models and datasets specified by Hugging Face ID, with a `revision` field forwarded to the loaders
- [x] Baselines (AR, LC, random, CT) implemented on the same sampler and settings
- [x] Tests and seed handling added (55 CPU tests; seeded `torch.Generator`)
- [x] Quality and NFE metrics measured on the toy model; real-model runs write rows in the `Performance_Metrics/` schema
- [x] No weights or data stored in the repository; models and datasets are used under the terms on their Hugging Face cards
