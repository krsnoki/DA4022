# DIPA Implementation Notes & Empirical Fixes

Because the official repository referenced in the paper (`https://github.com/allen4747/DIPA`) contains only a `README.md` with no published code, several implementation details not fully specified in [62_DIPA_Difficulty_Informed_Pr.pdf](../../../docs/62_DIPA_Difficulty_Informed_Pr.pdf) had to be resolved and validated against real GPU runs on `Qwen/Qwen2.5-Math-1.5B-Instruct` (`MATH-500`).

---

## 1. Pairing `DIPA(MGC)` with `input_proxy="const"` (`0.133` Single-Arm Lock-In)

### Symptom observed (Kaggle T4, `N = 20`, `pool_size = 8`)
`dipa_mgc` stalled at `0.133` coverage across every budget multiplier (`b=0.5` through `b=8`), wrapping the pool up to `150` times on `b=8`:

| Method | `b=0.5` | `b=1` | `b=2` | `b=4` | `b=8` |
|---|---:|---:|---:|---:|---:|
| `uniform` | 0.217 | 0.450 | 0.500 | 0.633 | 0.700 |
| `dipa_mgl` | 0.317 | 0.450 | 0.600 | 0.683 | 0.700 |
| `dipa_mgc` (with `mql` prior) | **0.133** | **0.133** | **0.133** | **0.133** | **0.133** |

### Root cause
1. In Table 1 of the paper, every generation proxy has an input-based counterpart on a comparable scale (`MQL -> MGL`, `MEnt(x) -> MEnt(O|x)`, `MGN(x) -> MGN(O|x)`, `MVoG(x) -> MVoG(O|x)`) **except** `MGC(O|x)`, which has no input-based proxy.
2. When `dipa_mgc` was initialized with `input_proxy="mql"`, untouched instances received initial difficulties in the `20` to `100` range (prompt token/word length).
3. On the very first failed attempt (`m = 1`), generation consensus over one attempt is `MGC = 1 / 1 = 1.0`, so its inverted difficulty (Appendix D.3) becomes `1 / 1.0 = 1.0`.
4. Raising `1 / M` to `lambda` gave that single failed arm ~100% of all future sampling probability (`1^lambda` vs `(1/30)^lambda`), trapping `dipa_mgc` on one failed problem forever.
5. By contrast, `MGL` (mean generation length, typically `100..500` tokens) is naturally larger than `MQL` (`20..100`), so failed arms move upward in difficulty instead of collapsing below untouched arms.

### Resolution & verification (`2f3a06f`)
In `Adaptive_Stopping/experiments/run_dipa_benchmark.py` (`_build_allocator`), `dipa_mgc` overrides `input_proxy="const"` (`1.0`). Untouched arms start at `1.0`; a first failed attempt stays at `1.0` (tied with untouched arms), and subsequent disagreeing failed attempts rise to `2.0`, `3.0`, etc., shifting compute onto other problems.

Re-running the exact same `N = 20` Kaggle setup immediately resolved the stall:

| Method | `b=0.5` | `b=1` | `b=2` | `b=4` | `b=8` |
|---|---:|---:|---:|---:|---:|
| `dipa_mgc` (with `const` prior) | **0.183** | **0.417** | **0.533** | **0.667** | **0.683** |

---

## 2. Normalizing Dynamic `lambda` by Batch Size (`0.430` Collapse at `N = 100`)

### Symptom observed (Colab T4, `N = 100`, `pool_size = 8`)
When scaling from `N = 20` to `N = 100`, `dipa_mgl` matched `easy2hard` almost identically (`0.430` flat from `b=1` to `b=4`) and trailed `uniform` at `b=2` and `b=4`:

| Method | `b=0.5` | `b=1` | `b=2` | `b=4` |
|---|---:|---:|---:|---:|
| `uniform` | 0.220 | 0.430 | 0.523 | 0.560 |
| `easy2hard` | 0.280 | 0.430 | 0.430 | 0.430 |
| `hard2easy` | 0.000 | 0.000 | 0.000 | 0.000 |
| `dipa_mgl` (unnormalized `lambda = |X|`) | 0.283 | **0.430** | **0.430** | **0.430** |
| `dipa_mgc` (unnormalized `lambda = |X|`) | 0.183 | 0.353 | **0.533** | **0.533** |

*(Note: `hard2easy = 0.000` is expected because greedy `argmax M_k` deterministically pulls the single hardest problem with `0 / 8` correct answers in the pool for all `T` steps.)*

### Root cause
1. Section 4.2 and Appendix E.7 specify a dynamic active sampling temperature `lambda proportional to |X|` (where `|X|` is the count of remaining unsolved arms), compared against fixed `lambda = 1` and `lambda = 10` in Figure 8, without stating the proportionality constant.
2. Implementing `lambda_t = 1.0 * |X|` directly meant `lambda_t` started at `20` when `N = 20` (still stochastic), but started at `100` when `N = 100` (and remained `57` after `43` solves; it would start at `500` for full `MATH-500`).
3. At `lambda_t >= 57`, `P_k proportional to 1 / M_k^lambda_t` concentrates ~100% probability mass on `argmin M_k`, collapsing probabilistic DIPA into deterministic greedy `easy2hard`. Any single short-but-unsolvable problem traps the entire remaining budget.

### Resolution
Normalize dynamic `lambda` by total batch size `N` in `Adaptive_Stopping/methods/dipa/allocator.py`:

$$\lambda_t = c \cdot \frac{|X|}{N}$$

where `c = config.lambda_value` (default `5.0` in `Adaptive_Stopping/configs/dipa.yaml`).

Why this stays faithful to the paper:
- For any fixed batch of size `N`, `lambda_t` remains **strictly proportional to `|X|`** (`|X|` is the only runtime variable).
- It keeps `lambda_t` in the `O(1)` regime (`[0, 5.0]`, directly between Appendix E.7's `lambda = 1` and `lambda = 10` ablations) regardless of whether `N = 20`, `100`, or `500`.
- It preserves DIPA's intended dynamic behavior: stronger exploitation early when `|X| / N` is near `1.0`, shifting smoothly toward exploration (`lambda_t -> 0`) as easier arms are eliminated.
- `--lambda-mode` and `--lambda-value` flags in `run_dipa_benchmark.py` expose both schedules cleanly for `--eval-only` sensitivity sweeps without re-running GPU generation.
