"""Integration tests on tiny randomly initialised Hugging Face models (CPU, no downloads).

Covers: exact V/O invariance under GQA (with and without v-bias), every method end to end,
target selection (Table 3 ablation), save/load drift, and the reproduction script.
"""
import csv

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import CONFIG_DIR  # noqa: E402

from Quantization_and_Pruning.quantization.weight_only.cafeq import (  # noqa: E402
    CafeQConfig,
    CafeQLinear,
    apply_vo_transform,
    attention_geometry,
    decoder_layers,
    load_quantized,
    quantize_model,
    save_quantized,
)

SMALL = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
)


def _llama(seed=0):
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(transformers.LlamaConfig(**SMALL)).eval()


def _qwen2(seed=0):  # q/k/v projections carry biases
    torch.manual_seed(seed)
    return transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**SMALL)).eval()


def _gemma2(seed=0):  # explicit head_dim, tied embeddings
    if not hasattr(transformers, "Gemma2ForCausalLM"):
        pytest.skip("transformers too old for Gemma 2")
    torch.manual_seed(seed)
    return transformers.Gemma2ForCausalLM(transformers.Gemma2Config(**SMALL, head_dim=16)).eval()


BUILDERS = {"llama": _llama, "qwen2": _qwen2, "gemma2": _gemma2}


def _tokens(n=2, t=16, seed=0):
    return torch.randint(0, SMALL["vocab_size"], (n, t), generator=torch.Generator().manual_seed(seed))


def _cfg(method="cafeq", bits=4, **targets):
    cfg = CafeQConfig.from_yaml(CONFIG_DIR / "tiny_smoke.yaml")
    cfg.quantization.method = method
    cfg.quantization.bits = bits
    for k, v in targets.items():
        setattr(cfg.quantization.targets, k, v)
    cfg.block_transform.iterations = 20
    cfg.paired_transform.iterations = 20
    cfg.adaptive_rounding.iterations = 2
    return cfg


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("arch", list(BUILDERS))
def test_vo_transform_is_output_preserving(arch):
    model = BUILDERS[arch]()
    x = _tokens()
    with torch.no_grad():
        ref = model(x).logits
    geom = attention_geometry(model.config)
    g = torch.Generator().manual_seed(1)
    for layer in decoder_layers(model):
        ms = [torch.eye(geom.head_dim) + 0.3 * torch.randn(geom.head_dim, geom.head_dim, generator=g)
              for _ in range(geom.n_kv)]
        apply_vo_transform(layer.self_attn, ms, geom)
    with torch.no_grad():
        out = model(x).logits
    assert _rel(out, ref) < 1e-4


@pytest.mark.parametrize("arch", list(BUILDERS))
@pytest.mark.parametrize("method", ["rtn", "random_hadamard", "cafeq"])
def test_methods_run_and_stay_close_at_8_bits(arch, method):
    model = BUILDERS[arch]()
    x = _tokens()
    with torch.no_grad():
        ref = model(x).logits
    report = quantize_model(model, _cfg(method, bits=8), log=None)
    with torch.no_grad():
        out = model(x).logits
    assert out.shape == ref.shape and torch.isfinite(out).all()
    assert _rel(out, ref) < 0.05
    assert len(report["layers"]) == SMALL["num_hidden_layers"]
    for layer in decoder_layers(model):
        for mod in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                    layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            assert isinstance(mod, CafeQLinear)
    assert not isinstance(model.lm_head, CafeQLinear)  # embeddings / head stay unquantized


def test_cafeq_report_contents_and_vo_improvement():
    model = _qwen2()
    report = quantize_model(model, _cfg("cafeq", bits=3), log=None)
    vo = report["layers"][0]["self_attn.vo"]
    assert vo["rel_pqe_final"] <= vo["rel_pqe_learned_independent"] + 1e-6  # keep_best in Algorithm 1
    q = report["layers"][0]["self_attn.q_proj"]
    assert q["rel_l2_final"] <= q["rel_l2_init"] + 1e-9
    assert report["estimated_extra_flops_pct"] > 0
    assert report["calibration_dataset"] is None


def test_targets_select_weight_sets():
    model = _llama()
    quantize_model(model, _cfg("rtn", ff=True, qk=False, vo=False), log=None)
    layer = decoder_layers(model)[0]
    assert isinstance(layer.mlp.down_proj, CafeQLinear)
    assert not isinstance(layer.self_attn.q_proj, CafeQLinear)
    assert not isinstance(layer.self_attn.v_proj, CafeQLinear)


def test_method_none_is_identity():
    model = _llama()
    x = _tokens()
    with torch.no_grad():
        ref = model(x).logits
    quantize_model(model, _cfg("none"), log=None)
    with torch.no_grad():
        assert torch.equal(model(x).logits, ref)


def test_save_load_roundtrip(tmp_path):
    model = _qwen2()
    quantize_model(model, _cfg("cafeq"), log=None)
    x = _tokens()
    with torch.no_grad():
        ref = model(x).logits
    save_quantized(model, tmp_path)
    fresh = load_quantized(_qwen2(seed=123), tmp_path)
    with torch.no_grad():
        out = fresh(x).logits
    assert torch.allclose(out, ref, atol=1e-6)


def test_run_script_tiny(tmp_path):
    import importlib.util

    script = CONFIG_DIR.parents[1] / "experiments" / "cafeq" / "run_cafeq.py"
    spec = importlib.util.spec_from_file_location("run_cafeq", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--config", str(CONFIG_DIR / "tiny_smoke.yaml"), "--tiny", "--baseline",
                   "--output-dir", str(tmp_path)])
    assert rc == 0
    rows = list(csv.DictReader(open(tmp_path / "cafeq_tiny_smoke" / "results.csv")))
    assert [r["method"] for r in rows] == ["none", "cafeq"]
    assert all(r["status"] == "smoke_test" and r["energy_j"] == "" for r in rows)
    assert (tmp_path / "cafeq_tiny_smoke" / "report_cafeq.json").exists()
