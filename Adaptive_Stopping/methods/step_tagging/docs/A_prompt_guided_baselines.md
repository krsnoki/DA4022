# A: Prompt-guided baselines

Part of the ST-ES implementation ([paper card](../PAPER_CARD.md)).

## What

The paper compares ST-ES with four *prompt-guided* baselines (Sec. 4, App. D, Fig. 12). In each, the model is told to be brief and to skip over-verification; nothing is stopped.

| Variant | Paper | Prompt |
|---|---|---|
| `pguided_user_zs.v1` | P(0)user | User: "Please do not reason extensively, be succinct, and put your final answer within \boxed{}. {question}" |
| `pguided_system_zs.v1` | P(0)system | System: "Respond concisely and confidently. Skip validations and over-verification steps." User: our standard prompt |
| `pguided_system_fs1.v1` | P(1)system | The same system prompt + "Here is an example: Example 1: …" (one verification step) |
| `pguided_system_fs3.v1` | P(3)system | The same system prompt + three example verification steps |

The instruction texts follow Fig. 12. The example steps are taken from the authors' MIT-licensed implementation of the same baselines (`yannisbelkhiter/TRACES`, `src/lrm_inference/utils.py`).

## Setup

```bash
python -m Adaptive_Stopping.experiments.st_es.run_prompt_guided --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
```

- **Everything but the prompt matches the standard run:** the same 200 GSM8K test problems, backbone, seed, sampling (T=0.6, top-p 0.95, top-k 50), 4096-token cap and batched cached generation.
- **System variants** keep our standard user prompt (question + "Please reason step by step, and put your final answer within \boxed{}."), so they differ from standard generation by the system instruction alone.
- **P(0)user** replaces that directive with the paper's instruction, which asks for a boxed answer itself.
- **Scoring:** Math-Verify on the answer section. Δ is a paired bootstrap (2000 resamples) against the standard traces on the same problems.
- **Model-card caveat:** the DeepSeek-R1 model card advises against system prompts; the paper uses them for these baselines, so we do too.
- **Outputs:** `experiments/st_es/results/text.gsm8k.test/prompt_guided.csv` and `.meta.json` (trace checksums), plus rows `st_es_pguided-*` in `Performance_Metrics/results/st_es_prompt_guided.csv` (repository metrics schema).
- **Tests:** `tests/test_prompt_guided.py`, 6 CPU tests covering the exact prompt texts, the message structure and rendering through the backbone's chat template.

## Results (GSM8K test, 200 problems)

| Config | Accuracy | Δ vs standard [95% CI] | Mean generated tokens | Tokens saved | Traces hitting the 4096 cap |
|---|---:|---|---:|---:|---:|
| Standard | 0.830 | – | 1786 | – | 28 |
| P(0)user | 0.820 | −1.0 [−6.5, +4.5] | 1037 | 41.9% | 12 |
| P(0)system | 0.840 | +1.0 [−4.0, +6.5] | 1244 | 30.3% | 14 |
| P(1)system | 0.855 | +2.5 [−2.5, +8.0] | 1230 | 31.1% | 9 |
| P(3)system | 0.825 | −0.5 [−6.0, +5.0] | 1273 | 28.7% | 17 |
| *ST-ES 95% (exploration, δ=0), for reference* | 0.830 | 0.0 [−3.0, +3.5] | 922 | 48.4% | – |
| *ST-ES 85/90% (verification, δ=0), for reference* | 0.815 | −1.5 [−5.5, +2.0] | 779 | 56.4% | – |

- **Effect of the prompts.** All four cut 29–42% of the tokens without a significant change in accuracy. Part of the saving comes from fewer runaway traces that hit the token cap (9–17 against 28).
- **Comparison with ST-ES.** At about the same accuracy, ST-ES saves more: 48–56% of tokens against 29–42%. The paper reports the same picture on its DeepSeek models ("ST-ES outperforms most Pguided baselines for both Deepseek models", Sec. 7).
- **Comparison with the paper's own numbers.** There (DS-R1-Distill-Llama-8B, GSM8K) the prompt-guided baselines saved 13–52% while losing 6–8 points. Ours lose nothing measurable, and on GSM8K neither does ST-ES. The 1.5B model and our prompt differ from the paper's setup (see the paper card).

## Not done

The prompt-guided baselines were not run on MATH-500, because each variant takes about 35 minutes of generation there. They have no live latency run either; their token counts come from cached generation, like the replay rows.
