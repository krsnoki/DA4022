# XShare routing simulations

Status: `approximate`

The three CSVs (`overlap.csv`, `speculative.csv`, `expert_parallel.csv`) come from `simulate_variants.py`, run offline on router logits captured during real decode steps of OLMoE-1B-7B-0125-Instruct on GSM8K.

Limits:

- No draft model was run. Speculative windows group consecutive true tokens of one request, which a real draft model may not produce.
- Expert-to-GPU placement is assumed (expert index mod G) on virtual GPUs. There is one physical GPU, so no latency or communication cost is measured.
- The numbers describe routing behaviour only: overlap, experts kept and activated, gate weight retained, and per-group load.

Regenerate with:

```bash
python -m Mixture_of_Experts.experiments.xshare.simulate_variants --logits Mixture_of_Experts/experiments/xshare/results/gsm8k/router_logits_c8.pt Mixture_of_Experts/experiments/xshare/results/gsm8k/router_logits_c32.pt --out-dir Mixture_of_Experts/experiments/xshare/results/simulations
```
