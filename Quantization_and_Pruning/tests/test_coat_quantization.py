from contextlib import ExitStack

import pytest
import torch

from Backbones.Generative_Pretrained.llama import TINY_LLAMA_ID, load_llama
from Dataloaders.causal_lm import TokenBatch
from Dataloaders.synthetic.causal_lm import make_synthetic_batches
from Quantization_and_Pruning.quantization.int4 import (
    affine_int4,
    fake_quantize_int4,
    pack_nibbles,
    unpack_nibbles,
)
from Quantization_and_Pruning.quantization.ptq import gptq
from Quantization_and_Pruning.quantization.ptq.coat import pipeline
from Quantization_and_Pruning.quantization.ptq.coat.calibration import collect_covariances
from Quantization_and_Pruning.quantization.ptq.coat.llama import fuse_llama
from Quantization_and_Pruning.quantization.ptq.coat.rotations import RotationDimensions, build_rotations
from Quantization_and_Pruning.quantization.ptq.coat.runtime import PackedKVCache, PackedLinear


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_affine_int4_edge_cases_and_even_rounding():
    x = torch.tensor([[0] * 5, [3] * 5, [-3] * 5, [0, 0.5, 1.5, 2.5, 15],
                      [-4, -2.5, -1.5, 0, 11]], dtype=torch.float16)
    codes, scale, zero = affine_int4(x)
    assert codes.dtype == zero.dtype == torch.uint8
    assert scale.dtype == torch.float32
    torch.testing.assert_close(scale[:, 0], torch.tensor([1, 0.2, 0.2, 1, 1]))
    assert zero.flatten().tolist() == [0, 0, 15, 0, 4]
    assert codes.tolist() == [[0] * 5, [15] * 5, [0] * 5, [0, 0, 2, 2, 15], [0, 2, 2, 4, 15]]
    torch.testing.assert_close(fake_quantize_int4(x)[:3], x[:3])


@pytest.mark.parametrize("width", [1, 15, 16])
def test_nibble_round_trip(width):
    codes = (torch.arange(width * 3, dtype=torch.uint8) % 16).view(width, 3).T
    packed = pack_nibbles(codes)
    assert packed.shape == (3, (width + 1) // 2)
    torch.testing.assert_close(unpack_nibbles(packed, width), codes)
    if width % 2:
        assert (packed[:, -1] >> 4).eq(0).all()
    assert pack_nibbles(torch.tensor([1, 2, 3], dtype=torch.uint8)).tolist() == [0x21, 0x03]


def test_gptq_compensates_across_columns_and_blocks():
    w = torch.tensor([[0.48, 0.45, 7.0], [-0.48, -0.45, -7.0], [0.0, 0.0, 0.0]])
    gram = torch.tensor([[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 1.0]])
    before = w.clone(), gram.clone()
    for block_size in (1, 2, 128):
        packed = gptq.gptq_quantize(w, gram, gptq.GPTQConfig(block_size=block_size))
        assert packed.scale.tolist() == [1.0, 1.0, 1.0]
        assert unpack_nibbles(packed.codes, 3).tolist() == [[7, 8, 14], [7, 6, 0], [7, 7, 7]]
    torch.testing.assert_close(w, before[0])
    torch.testing.assert_close(gram, before[1])


@torch.no_grad()
def test_gptq_grams_match_actual_rotated_projection_inputs():
    baseline = load_llama(TINY_LLAMA_ID)
    target = load_llama(TINY_LLAMA_ID)
    batches = list(make_synthetic_batches(count=4, seq_len=17, batch_size=2))
    mask = batches[0].attention_mask.clone()
    mask[0, -4:] = False
    batches[0] = TokenBatch(batches[0].input_ids, mask)
    cov = collect_covariances(baseline.adapter, batches)
    dims = RotationDimensions.from_config(baseline.model.config)
    rotated = fuse_llama(target.adapter, build_rotations(cov, dims))
    captured = {}
    valid = None

    def capture(key):
        def hook(_, args):
            captured.setdefault(key, []).append(args[0][valid])

        return hook

    with rotated.online_rotations(), ExitStack() as hooks:
        for i in range(dims.num_layers):
            for name, proj in rotated.adapter.projections(i).items():
                hooks.enter_context(proj.register_forward_pre_hook(capture((i, name))))
        for batch in batches:
            valid = batch.attention_mask
            rotated(input_ids=batch.input_ids, attention_mask=valid, use_cache=False)

    for i, grams in gptq.collect_input_grams(rotated, iter(batches)):
        assert set(grams) == set(rotated.adapter.projections(i))
        for name, gram in grams.items():
            x = torch.cat(captured[i, name]).double()
            torch.testing.assert_close(gram.double(), x.T @ x, atol=1e-5, rtol=1e-5)
        x = torch.cat(captured[i, "q_proj"])
        centered = x - x.mean(dim=0)
        assert not torch.allclose(grams["q_proj"], centered.T @ centered)


def test_packed_kv_cache_append_and_reload(tmp_path):
    gen = torch.Generator().manual_seed(3)
    k = torch.randn((2, 2, 4, 7), generator=gen).half()
    v = (4 * torch.randn(k.shape, generator=gen)).half()
    cache = PackedKVCache()
    k0, v0 = cache.update(k[:, :, :2], v[:, :, :2], 0)
    torch.testing.assert_close(k0, fake_quantize_int4(k[:, :, :2]))
    torch.testing.assert_close(v0, fake_quantize_int4(v[:, :, :2]))
    cache.update(k[:, :, :1], v[:, :, :1], 2)
    path = tmp_path / "kv-snapshot"
    torch.save(cache.export_state(), path)
    cache = PackedKVCache.from_state(torch.load(path, weights_only=True))
    full_k, full_v = cache.update(k[:, :, 2:], v[:, :, 2:], 0)
    torch.testing.assert_close(full_k, fake_quantize_int4(k), atol=0, rtol=0)
    torch.testing.assert_close(full_v, fake_quantize_int4(v), atol=0, rtol=0)
    assert cache.get_seq_length() == 4 and cache.get_seq_length(2) == 1
    keys, values = cache.layers[0]
    assert keys.codes.shape == values.codes.shape == (2, 2, 4, 4)
    assert keys.codes.dtype == values.codes.dtype == torch.uint8
    assert keys.scale.dtype == values.scale.dtype == torch.float32
    assert not torch.equal(keys.scale, values.scale)
    cache.reorder_cache(torch.tensor([1, 0, 0]))
    torch.testing.assert_close(cache.layers[0][0].dequantize(k.dtype), full_k[[1, 0, 0]])


@torch.no_grad()
def test_artifact_preserves_runtime_outputs_and_cache(monkeypatch, tmp_path):
    expected = []
    batch = make_synthetic_batches(split="evaluation", count=1, seq_len=9)[0]
    save = pipeline.save_coat

    def capture_and_save(model, path, **kwargs):
        expected.append(model(batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).logits)
        save(model, path, **kwargs)

    monkeypatch.setattr(pipeline, "save_coat", capture_and_save)
    report = pipeline.run_coat(pipeline.CoatConfig(calibration_count=4, calibration_seq_len=16), tmp_path)
    assert report.max_abs_logit_error < 1e-4
    artifact = torch.load(report.artifact_path, weights_only=True)
    assert artifact["version"] == pipeline.ARTIFACT_VERSION
    expected_batch = make_synthetic_batches(count=4, seq_len=16)
    torch.testing.assert_close(artifact["calibration"]["input_ids"], torch.cat([b.input_ids for b in expected_batch]))
    assert artifact["provenance"]["gptq"]["block_size"] == 128
    assert artifact["provenance"]["gptq"]["damp_percent"] == 0.01
    model = pipeline.load_coat(report.artifact_path)
    assert not model.config.tie_word_embeddings
    assert all(isinstance(proj, PackedLinear) and not hasattr(proj, "weight")
               for i in range(2) for proj in model.adapter.projections(i).values())

    args = dict(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    no_cache = model(**args, use_cache=False)
    with_cache = model(**args, use_cache=True)
    cache = with_cache.past_key_values
    assert no_cache.past_key_values is None
    assert isinstance(cache, PackedKVCache) and cache.get_seq_length() == 9
    assert all(k.codes.shape == (1, 2, 9, 4) for k, _ in cache.layers.values())
    torch.testing.assert_close(no_cache.logits, expected[0], atol=0, rtol=0)
    torch.testing.assert_close(with_cache.logits, no_cache.logits, atol=0, rtol=0)

    resumed = model(batch.input_ids[:, -1:], past_key_values=cache, use_cache=True)
    prefix = torch.cat((batch.input_ids, batch.input_ids[:, -1:]), dim=1)
    direct = model(prefix, use_cache=False)
    assert cache.get_seq_length() == 10
    torch.testing.assert_close(resumed.logits[:, -1], direct.logits[:, -1], atol=1e-6, rtol=1e-5)

    generated = model.generate(**args, use_cache=True, max_new_tokens=2, pad_token_id=0,
                               eos_token_id=None, return_dict_in_generate=True)
    plain = model.generate(**args, use_cache=False, max_new_tokens=2, pad_token_id=0, eos_token_id=None)
    assert isinstance(generated.past_key_values, PackedKVCache)
    torch.testing.assert_close(generated.sequences, plain)

    artifact["version"] += 1
    bad = tmp_path / "future-artifact"
    torch.save(artifact, bad)
    with pytest.raises(ValueError, match="artifact version"):
        pipeline.load_coat(bad)
