# Paper implementation card: Rethinking Layer Redundancy in Large Language Models

- **Paper / OpenReview:** https://openreview.net/forum?id=QuWWYol2nP
- **Primary module:** `Quantization_and_Pruning/pruning/layer_depth/`
- **Secondary modules:** none
- **Owner / reviewer:** @satviksingh121
- **Status:** implementing
- **Upstream code / licence:** not found (anonymous ICML submission)
- **Backbone IDs:** meta-llama/Llama-2-7b-hf
- **Dataset IDs and split:** allenai/c4 (calibration), EleutherAI/wikitext (evaluation), HellaSwag eval split, BoolQ eval split
- **Hardware assumptions:** CUDA GPU with ≥ 16 GB VRAM recommended; CPU-only smoke tests are supported

## Research summary

### Problem
Depth pruning (removing entire Transformer blocks) reduces LLM inference cost proportionally to the number of pruned layers. Prior work treats layer redundancy as an **intrinsic structural property** of the pretrained model — e.g. ShortGPT ranks layers by cosine similarity independent of any downstream task. This paper questions whether that assumption holds.

### Method
The paper formulates depth pruning as a **subset selection problem**:

> S* = argmin_{|S|=k} L(D, θ ∘ m_S)

where S is the set of removed layers, L is a calibration loss, D is a calibration dataset, and θ is the pretrained model.

**Two calibration objectives are compared:**
1. **Perplexity** — negative log-likelihood on the C4 dataset (task-agnostic fluency signal)
2. **Task Likelihood Margin (TLM)** — for each multiple-choice example, `log P(correct) - max(log P(wrong choices))` on Commonsense 170k (discriminative, task-specific signal)

**Seven search algorithms are compared** (we implement 2):
- One-shot top-k: score each layer independently by its solo removal impact, remove k worst
- Greedy iterative: remove one layer at a time, always picking the least harmful

**Main finding:** Changing the calibration objective produces qualitatively different pruning patterns. Changing the search algorithm (under a fixed objective) converges to similar solutions. → Objective matters more than algorithm.

### Expected behaviour
- Perplexity-guided pruning removes contiguous mid-to-late layers
- TLM-guided pruning removes more scattered/distributed layers
- Spearman correlation between perplexity rank and accuracy rank is near zero or negative
- Accuracy drop: ~15–20 pp at k=7, ~20–25 pp at k=9 (from paper Table 1)

## Implementation

- **Input contract:** HuggingFace `AutoModelForCausalLM`, tokenizer, calibration dataset as HF `Dataset`, pruning budget k (int)
- **Output contract:** pruned model (in-place), list of removed layer indices, calibration score
- **Configuration:** `config.yaml` — model ID, revision, dataset, seed, batch size, dtype, pruning budget
- **Reproduction command:** `python run_experiment.py --config config.yaml`
- **Tests:** `python -m pytest tests/test_smoke.py -v`

## Results

| Source | Quality (HellaSwag %) | Quality (BoolQ %) | Perplexity (WikiText-2) | Peak memory | Status |
|---|---:|---:|---:|---:|---|
| Paper-reported (LLaMA3-8B, k=7, TLM, Greedy) | 60.47 | 67.49 | — | — | paper-reported |
| Local baseline (unpruned) | TBD | TBD | TBD | TBD | pending |
| Local method (one-shot + perplexity) | TBD | TBD | TBD | TBD | pending |
| Local method (one-shot + TLM) | TBD | TBD | TBD | TBD | pending |
| Local method (greedy + perplexity) | TBD | TBD | TBD | TBD | pending |
| Local method (greedy + TLM) | TBD | TBD | TBD | TBD | pending |

## Differences and limitations

- Using 1 LLM family (LLaMA-2-7B) instead of 3 (LLaMA3, LLaMA3.1, Qwen3)
- Implementing 2 search algorithms instead of 7 (one-shot and greedy)
- Using 1 pruning budget (k=7) instead of 2 (k=7 and k=9)
- Evaluating on 2 downstream tasks instead of 6
- No access to Commonsense 170k training split; using publicly available commonsense benchmarks for TLM calibration

## Reproduction checklist

- [x] Exact paper link and version verified
- [x] Base model and dataset revisions pinned in config.yaml
- [ ] Baseline runs on the same setup
- [ ] Tests and seed handling added
- [ ] Resource and quality metrics reported
- [ ] Licence/data terms checked
