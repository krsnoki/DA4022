# Paper implementation card: TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning

- **Paper / OpenReview:** https://openreview.net/forum?id=GwTFovD6SM (AdaptFM workshop, ICML 2026); arXiv https://arxiv.org/abs/2603.12529 (v3 read for this card)
- **Primary module:** Adaptive_Stopping (learned exit)
- **Secondary modules:** Dataloaders (MATH loader), Backbones (Qwen3-1.7B card), Performance_Metrics (result rows)
- **Owner / reviewer:** Arnav Kshirsagar, Pranav Shalya, Amit Kumar, Ashmit Satpute, Atharv Wankhede (issue #12)
- **Status:** partial-reproduction
- **Upstream code / licence:** https://huggingface.co/acnagle/Terminator-Qwen3-8B, Apache-2.0. Inference code and a trained probe for Qwen3-8B only. The labelling and training code is not released; we wrote it from the paper.
- **Backbone IDs:** `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, float16, thinking mode
- **Dataset IDs and split:** probe training: `EleutherAI/hendrycks_math` (7 configs, split `train`, 2000 problems sampled with seed 0, of which 1,280 got a trace before the Kaggle time limit; 10% held out as validation); evaluation: `HuggingFaceH4/MATH-500` (split `test`, all 500)
- **Hardware assumptions:** Kaggle Tesla T4 16 GB (compute capability 7.5, no bf16, no FlashAttention 2). vLLM 0.34 runs with the Triton attention backend.

## Research summary

### Problem
Reasoning models keep thinking after they have already written the answer they
will finally give. Those extra tokens cost time and memory but rarely change
the answer.

### Method
A small probe reads the base model's hidden states during the thinking phase
and predicts, at every thinking token, whether the final answer has already
appeared in the chain of thought.

- Probe input: output of the last decoder layer before the final RMSNorm
  (the released code captures it with a forward hook on `model.layers[-1]`).
- Probe: one extra decoder layer, initialised as a copy of the base model's
  last decoder layer, run over all positions (prompt and thinking), followed
  by RMSNorm and a linear layer to one logit. The base model stays frozen.
- Labels: 0 for thinking tokens before the first appearance of the final
  answer, 1 from that token on. The paper finds that position with a second
  reasoning model (extract, identify, verify).
- Loss: binary cross-entropy averaged over thinking tokens, with
  inverse-frequency class weights w0 = (n0+n1)/(2 n0), w1 = (n0+n1)/(2 n1).
- Stopping rule: keep the last 10 probe outputs; a token counts as 1 when its
  probability is above 0.7; stop when more than half of the window is 1
  (6 of 10). Then the text "\nI've run out of thinking tokens. I need to
  commit to a final answer." and `</think>` are forced, and the model writes
  its answer.

### Expected behaviour
Paper-reported, Qwen3-8B on MATH-500: accuracy 91.1% to 90.7% while mean
thinking tokens drop from 5,037 to 2,425 (compression 45.1%, mean of
per-sample ratios). Latency on a GH200 with vLLM, batch size 1: 32.7 s to
14.1 s per problem; throughput 151.5 to 135.2 tokens/s (10.8% probe overhead).
We expect a similar trade-off on Qwen3-1.7B, but with a weaker probe because
we train on about 1,000 labelled traces from one dataset instead of about 110k traces from four.

## Implementation

- **Input contract:** `TerminatorProbe.forward(hidden[B, T, H]) -> logits[B, T]` where `hidden` is the pre-norm output of the base model's last decoder layer (fp16 or fp32). Optional `past_key_values` (a transformers `Cache`) and `position_offset` for token-by-token use. `TerminatorPolicy.should_stop(state, metrics, budget)` expects `metrics["terminator_prob"]` for the newest thinking token; `state["thinking_tokens"]` and `budget` (max thinking tokens) are optional. The probability comes from the probe layer on the hidden state the base model already computed, so there is no extra base-model pass.
- **Output contract:** `should_stop` returns a bool; `report()` returns the threshold, window, stop reason (`probe` or `budget`), exit position, number of predictions and of missing telemetry steps. `generate_with_exit` returns thinking and answer token ids, the stop reason, per-token probabilities and timings. `exit_index(probs, threshold, window)` gives the number of thinking tokens kept, or None.
- **Configuration:** `Adaptive_Stopping/configs/terminator/qwen3_1_7b_math500.yaml` (real run), `tiny_smoke.yaml` (CPU tests)
- **Reproduction command:** see `Adaptive_Stopping/experiments/terminator/README.md` (one Kaggle kernel per stage, built with `kaggle/build_kernel.py`)
- **Tests:** `python -m pytest Adaptive_Stopping/tests -q`

## Results

Full tables, plots and commands: `Adaptive_Stopping/experiments/terminator/README.md`
and `.../experiments/terminator/results/`. Qwen3-1.7B, fp16, Tesla T4, one
trace per problem (seed 0).

**Probe.** Best validation token macro F1 0.892 (110 traces); on the 432
labelled MATH-500 test traces token macro F1 0.848, accuracy 0.863.

**Offline, MATH-500 (n = 500), window 10:**

| method | accuracy | mean thinking tokens | per-sample compression |
|---|---|---|---|
| no stop | 0.870 | 3,474 | 1.000 |
| TERMINATOR, threshold 0.7 (paper default) | 0.838 | 1,472 | 0.390 |
| TERMINATOR, threshold 0.8 | 0.850 | 1,634 | 0.428 |
| TERMINATOR, threshold 0.9 | 0.850 | 1,848 | 0.487 |
| fixed budget 1,472 tokens | 0.804 | 1,427 | 0.555 |
| oracle cut at the labelled answer position | 0.852 | 1,724 | 0.442 |

**Online latency, batch 1, HF loop, 25 problems:** no stop 0.92 accuracy,
median 304.5 s; TERMINATOR (0.7) 0.84 accuracy, median 107.6 s (2.8x
faster), probe 2.55 ms per thinking token, +241 MB peak memory.

**Compared with the paper** (Qwen3-8B, MATH-500: 91.1% to 90.7%, per-sample
compression 45.1%): we keep a similar or larger share of thinking tokens
removed (39-49% kept) but lose 2.0-3.2 points of accuracy (thresholds 0.7-0.9) instead of 0.4. The
validation rule (fewest thinking tokens with val accuracy at most 1 point
below no stop) picked no threshold. Likely reasons: about 1,000 training
traces instead of about 110k, a string-matching labeller, a smaller
backbone, and re-sampled answers after the exit (the oracle cut also loses
1.8 points). Status stays partial reproduction.

## Differences and limitations

- **Backbone.** Qwen3-1.7B instead of the paper's Qwen3-8B/14B. A 1.7B
  model fits on one T4 with room for the probe; the 8B model alone needs
  about 16 GB in fp16, which a T4 does not have. The released 8B probe cannot
  be reused because the hidden size differs.
- **Labeller.** The paper finds the first answer position with a second
  reasoning model. We use string matching on a normalised view of the
  thinking text (`labels.py`). It misses answers written in another form
  (for example "one half" for 1/2) and can match too early when a short
  answer also appears in an intermediate step; the problem-statement and
  context checks reduce, but do not remove, the second case. Traces whose
  final answer is never found in the thinking text are dropped from training.
- **Training data.** We planned 2000 MATH train problems, one trace each,
  but the trace kernel hit Kaggle's 12 h limit after 1,280 of them. These
  were split into 1,152 training and 128 validation traces; after labelling
  969 and 110 were kept (the rest had no boxed answer or the answer was not
  found in the thinking text). The paper uses about 110k traces from four
  datasets. One sampled trace per problem.
- **Evaluation data.** MATH-500 only. The paper also reports other
  benchmarks.
- **Thinking cap.** 6000 thinking tokens. Traces that hit the cap get the
  exit message and `</think>`, like an early exit.
- **Accuracy runs are offline.** The probe is causal, so the exit position is
  read from the probe probabilities on the full no-stop trace and only the
  answer is generated again (vLLM). This gives the same exit point as online
  generation for the same sampled thinking, but the answer is a new sample
  (different seed), so a small part of any accuracy difference is sampling
  noise. `test_cached_probe_matches_full_pass` checks that the online probe
  with its own KV cache gives the same probabilities as the offline pass.
- **Latency runs are online, but not with vLLM.** The paper measures latency
  with a vLLM integration on a GH200. We use a plain HF token loop at batch
  size 1 on a T4, so absolute latencies are much higher and the probe
  overhead is a larger fraction of the step time. The no-stop and TERMINATOR
  runs use the same loop on two T4s of the same machine, started together.
- **Precision.** fp16 base model (T4 has no bf16). The probe, its gradients
  and AdamW are kept in float32, without autocast, for training, offline
  probabilities and the latency runs. The first training run used fp16
  autocast for the probe and was broken: the copied last layer's MLP output
  (`down_proj`) goes above the fp16 maximum of 65504 at some positions
  (73k-89k in float32 on the traces we checked), so it became inf and the
  head's RMSNorm turned it into NaN. About 0.4% of thinking tokens were
  affected, in about 74% of the traces, so nearly every batch of 16 had a
  NaN loss and the gradient scaler skipped every optimizer step: the saved
  probe was still its initialisation. The base model's own hidden states
  were finite (largest value about 14.9k). `train_probe.py` now also stops
  with an error if a loss in the first 32 traces is not finite, replaces
  non-finite base hidden states (counted in `train_log.json`; none were
  found) and clips gradients to norm 1.0.
- **Single seed.** One trace per problem, seed 0. No confidence intervals
  over seeds.
- **Energy** was not measured.

## Reproduction checklist

- [x] Exact paper link and version verified (arXiv 2603.12529v3)
- [x] Base model and dataset revisions pinned (model revision pinned; datasets loaded from the hub default branch)
- [x] Baseline runs on the same setup (same traces, prompts and seeds)
- [x] Tests and seed handling added
- [x] Resource and quality metrics reported
- [x] Licence/data terms checked (Qwen3: Apache-2.0; upstream code: Apache-2.0; MATH: MIT; MATH-500: from MATH)
