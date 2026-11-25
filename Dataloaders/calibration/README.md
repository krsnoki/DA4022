# AAAC calibration inputs

The Qwen2.5-0.5B base loader lives in `Backbones/Generative_Pretrained/qwen2_5.py`. C4 calibration data is kept here because it is a data-role concern, not part of the backbone or AAAC algorithm.

`iter_c4_calibration_batches` reads only the English C4 train split at a pinned dataset revision. It streams documents in a seeded shuffle order, selects one seeded 2,048-token window per sufficiently long document, and defaults to four windows with seed 42. Each returned batch has shape `[1, 2048]`, so these calibration windows contain no padding. The dataset license is ODC-BY; follow the dataset card's terms when using the data.

`collect_activation_column_energy` includes two-dimensional `torch.nn.Linear` weights whose module path has a `layers` component. For Qwen2.5, that includes attention and MLP projections in each decoder block. It excludes embeddings, the output `lm_head`, and linear modules outside decoder blocks. RTN and AAAC should use this same eligible-layer set. The helper runs in evaluation mode, uses each batch's attention mask to ignore padding, and accumulates the sum of squared inputs per feature in FP32 on CPU. It keeps no activation history.

## Hand-off contract (Member 1 → Members 2, 3, 4)

```python
from Backbones.Generative_Pretrained import load_qwen25_05b
from Dataloaders.calibration import (
    collect_activation_column_energy, iter_c4_calibration_batches, iter_quantization_targets,
)

model, tok = load_qwen25_05b(device="cpu", dtype=torch.float32)       # pass dtype explicitly in benchmarks
batches = iter_c4_calibration_batches(tok, num_sequences=4, sequence_length=2048, seed=42)
energy = collect_activation_column_energy(model, batches)              # {layer_name: FP32 CPU tensor [in]}

for t in iter_quantization_targets(model, energy):                     # one layer at a time
    t.weight      # FP32 copy of W, shape [out, in]
    t.importance  # FP32 I[k] = sum_t X[t,k]^2, shape [in]
    t.module      # live nn.Linear (read t.module.bias; never quantized)
```

* Eligible set for Qwen2.5-0.5B: 24 decoder blocks × 7 projections = **168** layers (q/k/v/o, gate/up/down); 357.8M weights. `embed_tokens` and the tied `lm_head` stay in full precision. RTN and AAAC must both use exactly this set.
* In-feature widths are 896 (q/k/v/o, gate/up) and 4864 (down); both are divisible by 128, so g = 128 and S = 16 need no padding.
* q/k/v carry a bias; it is never quantized and must be preserved by the integration step.
* Importance comes from a single forward pass of the unmodified model, so every layer sees full-precision inputs (no sequential re-calibration through already-quantized layers).
* Calibration is deterministic for a fixed `datasets` version, seed and pinned revisions; record `datasets`/`transformers` versions with results.
