import pytest

torch = pytest.importorskip("torch")
from torch import nn

from Quantization_and_Pruning.quantization.aaac import (
    AAACConfig,
    assign_selection_groups,
    encode_indices,
    fit_aaac_layer,
    fit_aaac_model,
    initial_codebooks,
    normalize_weight,
)

from aaac_helpers import keep_tables, round_tables

STUBS = dict(update_fn=keep_tables, finalize_fn=round_tables)


def _layer(out=24, inp=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inp, generator=g) * 0.02
    w[3, 7] = 0.5          # an outlier
    w[5, 128:256] = 0.0    # an all-zero scale group
    imp = torch.rand(inp, generator=g) * 1000
    imp[:8] = 0.0          # some dead columns
    return w, imp


@pytest.mark.parametrize("selection_size", [128, 16])
def test_fit_output_shapes_dtypes_and_ranges(selection_size):
    w, imp = _layer()
    cfg = AAACConfig(group_size=128, selection_size=selection_size, outer_iters=3, inner_iters=10)
    fit = fit_aaac_layer(w, imp, cfg, **STUBS)

    assert fit.scales.shape == (24, 2) and fit.scales.dtype == torch.float16
    assert (fit.scales > 0).all()
    assert fit.codebooks.shape == (2, 16) and fit.codebooks.dtype == torch.bfloat16
    assert fit.selectors.shape == (24, 256 // selection_size) and fit.selectors.dtype == torch.bool
    assert fit.indices.shape == (24, 256) and fit.indices.dtype == torch.uint8
    assert int(fit.indices.max()) < 16
    assert fit.group_size == 128 and fit.selection_size == selection_size
    assert len(fit.objective_history) == cfg.outer_iters + 1
    assert fit.shape == (24, 256)


def test_fit_is_bit_identical_across_runs_and_chunk_sizes():
    w, imp = _layer(seed=4)
    base = fit_aaac_layer(w, imp, AAACConfig(selection_size=16), **STUBS)
    for chunk in (1 << 24, 1, 5000):
        again = fit_aaac_layer(w.clone(), imp.clone(), AAACConfig(selection_size=16, chunk_elements=chunk), **STUBS)
        for name in ("scales", "codebooks", "selectors", "indices"):
            assert torch.equal(getattr(base, name), getattr(again, name)), name
        assert base.objective_history == again.objective_history


def test_final_selectors_and_indices_are_recomputed_with_the_bf16_tables():
    w, imp = _layer(seed=1)
    cfg = AAACConfig(selection_size=16)
    fit = fit_aaac_layer(w, imp, cfg, **STUBS)
    u = normalize_weight(w, fit.scales, 128)
    tables = fit.codebooks.float()
    sel, err = assign_selection_groups(u, imp, tables, 16)
    idx, _ = encode_indices(u, tables, sel, 16)
    assert torch.equal(fit.selectors, sel) and torch.equal(fit.indices, idx)
    assert fit.objective_history[-1] == pytest.approx(err.double().sum().item())
    # with the identity update the BF16 tables are exactly the rounded initial tables
    assert torch.equal(fit.codebooks, initial_codebooks(u, imp, 16).to(torch.bfloat16))


def test_update_fn_is_called_outer_times_inner_with_current_selectors():
    w, imp = _layer(seed=2)
    calls = []

    def recording_update(normalized, importance, codebooks, selectors, selection_size, *, chunk_elements):
        calls.append((selectors.clone(), selection_size, chunk_elements, codebooks.dtype))
        return codebooks + 0.0

    cfg = AAACConfig(selection_size=16, outer_iters=3, inner_iters=4, chunk_elements=12345)
    fit_aaac_layer(w, imp, cfg, update_fn=recording_update, finalize_fn=round_tables)
    assert len(calls) == 12
    assert all(c[1] == 16 and c[2] == 12345 and c[3] == torch.float32 for c in calls)
    assert all(c[0].shape == (24, 16) for c in calls)


def test_dequantize_is_scale_times_selected_table_entry():
    w, imp = _layer(seed=3)
    fit = fit_aaac_layer(w, imp, AAACConfig(selection_size=16), **STUBS)
    w_hat = fit.dequantize()
    o, k = 7, 200
    table = int(fit.selectors[o, k // 16])
    expected = fit.scales[o, k // 128].float() * fit.codebooks[table, int(fit.indices[o, k])].float()
    assert w_hat[o, k] == expected
    assert w_hat.shape == w.shape and w_hat.dtype == torch.float32
    assert torch.equal(w_hat[5, 128:], torch.zeros(128)) or w_hat[5, 128:].abs().max() < 1e-4
    # 4-bit reconstruction of a Gaussian layer: clearly better than zeroing it
    assert (w_hat - w).square().mean() < 0.05 * w.square().mean()


def test_zero_outer_iterations_and_all_zero_importance_work():
    w, _ = _layer(seed=5)
    fit = fit_aaac_layer(w, torch.zeros(256), AAACConfig(outer_iters=0), **STUBS)
    assert len(fit.objective_history) == 1 and fit.objective_history[0] > 0  # uniform fallback


def test_input_tensors_are_not_modified():
    w, imp = _layer(seed=6)
    w0, imp0 = w.clone(), imp.clone()
    fit_aaac_layer(w, imp, AAACConfig(selection_size=16), **STUBS)
    assert torch.equal(w, w0) and torch.equal(imp, imp0)


def test_validation_errors():
    w, imp = _layer()
    with pytest.raises(ValueError, match="divide"):
        AAACConfig(group_size=128, selection_size=48)
    with pytest.raises(ValueError, match="non-negative"):
        AAACConfig(outer_iters=-1)
    with pytest.raises(ValueError, match="positive int"):
        AAACConfig(group_size=0)
    with pytest.raises(ValueError, match="importance shape"):
        fit_aaac_layer(w, imp[:10], **STUBS)
    with pytest.raises(ValueError, match="non-negative"):
        fit_aaac_layer(w, -imp - 1, **STUBS)
    with pytest.raises(ValueError, match="divisible"):
        fit_aaac_layer(w[:, :200], imp[:200], **STUBS)
    with pytest.raises(ValueError, match="update_fn"):
        fit_aaac_layer(w, imp, update_fn=lambda *a, **k: torch.zeros(2, 8), finalize_fn=round_tables)
    with pytest.raises(ValueError, match="finalize_fn"):
        fit_aaac_layer(w, imp, update_fn=keep_tables, finalize_fn=lambda c, d: c)  # still float32


def test_default_functions_point_at_member_2b_stubs_until_implemented():
    w, imp = _layer()
    try:
        fit_aaac_layer(w, imp, AAACConfig(outer_iters=1, inner_iters=1))
    except NotImplementedError as exc:
        assert "Member 2" in str(exc)


def test_end_to_end_with_real_lloyd_update():
    """Activates automatically once Member 2B implements lloyd.py."""
    w, imp = _layer(out=64, inp=512, seed=7)
    cfg = AAACConfig(selection_size=16)
    try:
        fit = fit_aaac_layer(w, imp, cfg)
    except NotImplementedError:
        pytest.skip("lloyd.py is not implemented yet (Member 2B)")
    init_only = fit_aaac_layer(w, imp, cfg, **STUBS)
    hist = fit.objective_history
    # Lloyd steps and re-assignment never increase the objective (before BF16 rounding).
    assert all(b <= a * (1 + 1e-6) for a, b in zip(hist[:-2], hist[1:-1]))
    assert hist[-1] <= init_only.objective_history[-1] * (1 + 1e-6)
    assert fit.codebooks.dtype == torch.bfloat16


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 128)
        self.layers = nn.ModuleList([nn.Linear(128, 256, bias=True), nn.Linear(256, 128, bias=False)])
        self.lm_head = nn.Linear(128, 10, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.lm_head(self.layers[1](self.layers[0](self.embed_tokens(input_ids))))


def test_fit_aaac_model_covers_eligible_layers_without_touching_the_model():
    from Dataloaders.calibration import collect_activation_column_energy

    torch.manual_seed(0)
    model = _TinyLM()
    energy = collect_activation_column_energy(
        model, [{"input_ids": torch.randint(0, 10, (1, 12)), "attention_mask": torch.ones(1, 12, dtype=torch.long)}]
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fits = fit_aaac_model(model, energy, AAACConfig(selection_size=16), **STUBS)
    assert list(fits) == ["layers.0", "layers.1"]
    assert fits["layers.0"].indices.shape == (256, 128) and fits["layers.1"].indices.shape == (128, 256)
    assert all(f.indices.device.type == "cpu" for f in fits.values())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
