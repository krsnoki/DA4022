"""End-to-end CPU smoke test for the whole AAAC pipeline.

calibration hooks -> AAAC fit -> 4-bit packing -> AAACLinear model -> perplexity,
with the asymmetric (zero-point) INT4 RTN baseline on the same layers.

Uses a tiny randomly initialised Qwen2 model, so it needs no downloads and runs
in a few seconds on CPU. It checks that the pieces connect, not paper accuracy.
"""

import copy
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from Dataloaders.calibration.activation_energy import (
    collect_activation_column_energy,
    eligible_linear_modules,
)
from Dataloaders.wikitext2 import EvalWindows
from Quantization_and_Pruning.quantization.aaac import (
    AAACConfig,
    AAACLinear,
    RTNLinear,
    fit_aaac_model,
    measure_model_storage_bytes,
    quantize_model_rtn,
    replace_linear_layers_with_aaac,
)
from Quantization_and_Pruning.quantization.aaac.benchmark import evaluate_wikitext2_perplexity

GROUP = 128
VOCAB = 512
SEQ = 64


def _tiny_qwen2():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=256,          # multiple of g = 128
        intermediate_size=512,    # multiple of g = 128
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=SEQ,
        tie_word_embeddings=True,
    )
    return transformers.Qwen2ForCausalLM(cfg).eval()


def _calibration_batches():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (4, SEQ), generator=g)
    mask = torch.ones_like(ids)
    mask[1, -8:] = 0  # some padding, which the hooks must skip
    return [{"input_ids": ids[i : i + 2], "attention_mask": mask[i : i + 2]} for i in (0, 2)]


def _eval_windows():
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, VOCAB, (3, SEQ), generator=g)
    return EvalWindows(
        input_ids=ids, total_available_windows=3, seq_len=SEQ, is_subset=False,
        dataset_name="random", dataset_config="smoke", split="test",
    )


def _weighted_error(base, quant, energy):
    """sum over layers of sum_k I[k] * (W_hat - W)^2, the objective AAAC minimises."""
    base_mods = dict(base.named_modules())
    total = 0.0
    for name, mod in quant.named_modules():
        if isinstance(mod, (AAACLinear, RTNLinear)):
            diff = (mod.dequantize_weight(torch.float32) - base_mods[name].weight.float()).detach()
            total += float((diff.square() * energy[name]).sum())
    return total


@pytest.fixture(scope="module")
def setup():
    model = _tiny_qwen2()
    energy = collect_activation_column_energy(model, _calibration_batches())
    return model, energy


def test_calibration_covers_every_eligible_layer(setup):
    model, energy = setup
    layers = eligible_linear_modules(model)
    assert set(energy) == set(layers)
    assert len(layers) == 2 * 7  # q, k, v, o, gate, up, down per decoder layer
    for name, e in energy.items():
        assert e.shape == (layers[name].in_features,)
        assert e.dtype == torch.float32
        assert torch.isfinite(e).all() and (e > 0).all()


@pytest.mark.parametrize("selection_size", [128, 16])
def test_end_to_end_pipeline(setup, selection_size):
    model, energy = setup
    targets = eligible_linear_modules(model)
    windows = _eval_windows()
    cpu = torch.device("cpu")

    # fit -> pack -> swap in AAACLinear
    fits = fit_aaac_model(model, energy, AAACConfig(group_size=GROUP, selection_size=selection_size))
    aaac = replace_linear_layers_with_aaac(copy.deepcopy(model), fits)
    rtn = quantize_model_rtn(copy.deepcopy(model), targets.keys(), group_size=GROUP, asymmetric=True)

    # every target layer swapped, nothing else touched
    aaac_mods = dict(aaac.named_modules())
    for name, mod in targets.items():
        assert isinstance(aaac_mods[name], AAACLinear)
        if mod.bias is not None:  # Qwen q/k/v projections have a bias
            assert torch.equal(aaac_mods[name].bias, mod.bias)
    assert torch.equal(aaac.model.embed_tokens.weight, model.model.embed_tokens.weight)
    assert torch.equal(aaac.model.norm.weight, model.model.norm.weight)

    # packed model reconstructs exactly what the fit says
    for name, fit in fits.items():
        assert torch.equal(aaac_mods[name].dequantize_weight(torch.float32), fit.dequantize())

    # exact packed byte counts
    storage = measure_model_storage_bytes(aaac)
    weights = sum(m.out_features * m.in_features for m in targets.values())
    assert storage["quantized_layer_count"] == len(targets)
    assert storage["packed_indices_bytes"] == weights // 2              # two 4-bit indices per byte
    assert storage["codebooks_bytes"] == 64 * len(targets)              # 2 tables x 16 x BF16
    assert storage["scales_bytes"] == (weights // GROUP) * 2            # FP16 scale per group
    expected_selector = 0 if selection_size == GROUP else math.ceil(weights / selection_size / 8)
    assert storage["selector_bitset_bytes"] == expected_selector

    # AAAC beats RTN on the activation-weighted error it optimises (same scales and layers)
    err_aaac, err_rtn = _weighted_error(model, aaac, energy), _weighted_error(model, rtn, energy)
    print(f"S={selection_size} weighted error aaac={err_aaac:.4g} rtn={err_rtn:.4g}")
    assert err_aaac < err_rtn

    # shared perplexity runner works on all three variants
    with torch.no_grad():
        results = {
            name: evaluate_wikitext2_perplexity(m, windows, cpu)
            for name, m in {"unmodified": model, "rtn": rtn, "aaac": aaac}.items()
        }
    for r in results.values():
        assert math.isfinite(r["perplexity"])
        assert r["evaluated_tokens"] == 3 * (SEQ - 1)
    # 4-bit should not wreck the model: perplexity stays within 10% of the original
    base = results["unmodified"]["perplexity"]
    print({k: round(v["perplexity"], 3) for k, v in results.items()})
    assert abs(results["aaac"]["perplexity"] / base - 1) < 0.10
