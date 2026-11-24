from contextlib import ExitStack
from dataclasses import replace
from math import sqrt

import pytest
import torch

from Backbones.Generative_Pretrained.llama import TINY_LLAMA_ID, load_llama
from Dataloaders.causal_lm import TokenBatch
from Dataloaders.synthetic.causal_lm import make_synthetic_batches
from Quantization_and_Pruning.quantization.ptq.coat.calibration import (
    WelfordCovariance,
    collect_covariances,
)
from Quantization_and_Pruning.quantization.ptq.coat.llama import fuse_llama
from Quantization_and_Pruning.quantization.ptq.coat.rotations import (
    Hadamard,
    RotationDimensions,
    build_rotations,
    load_hadamard44,
    load_hadamard172,
)

TOL = 1e-9


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def calibrated():
    backbone = load_llama(TINY_LLAMA_ID, seed=0)
    batches = make_synthetic_batches(seed=0, count=8, seq_len=32, batch_size=2)
    cov = collect_covariances(backbone.adapter, batches, chunk_size=17)
    dims = RotationDimensions.from_config(backbone.model.config)
    return cov, dims


def _direct_cov(x):
    x = x.double()
    centered = x - x.mean(dim=0)
    return centered.T @ centered / (len(x) - 1)


def _sylvester(width):
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < width:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)))
    return h / sqrt(width)


def test_chunked_covariance():
    gen = torch.Generator().manual_seed(5)
    x = 1e5 + torch.randn((37, 8), generator=gen, dtype=torch.float64)
    stats = WelfordCovariance(8, chunk_size=5)
    for rows in x.split(11):
        stats.update(rows)
    assert stats.count == len(x)
    torch.testing.assert_close(stats.covariance(), _direct_cov(x), atol=TOL, rtol=TOL)


def test_collection_pools_only_valid_rows_from_expected_sites():
    adapter = load_llama(TINY_LLAMA_ID).adapter
    batch = make_synthetic_batches(batch_size=2)[0]
    mask = batch.attention_mask.clone()
    mask[0, 20:] = False
    mask[1, 25:] = False
    batch = TokenBatch(batch.input_ids, mask)
    norm_rows = []
    head_rows = [[] for _ in adapter.layers]

    def keep_norm(norm, args):
        x = args[0][mask].float()
        norm_rows.append(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon))

    def keep_heads(i):
        def collect(_, args):
            p = adapter.layers[i].self_attn.head_dim
            head_rows[i].append(args[0][mask].reshape(-1, p))

        return collect

    with ExitStack() as hooks:
        for i in range(len(adapter.layers)):
            for norm in adapter.norms(i):
                hooks.enter_context(norm.register_forward_pre_hook(keep_norm))
            hooks.enter_context(adapter.projections(i)["o_proj"].register_forward_pre_hook(keep_heads(i)))
        cov = collect_covariances(adapter, [batch], chunk_size=3)

    assert len(norm_rows) == 2 * len(adapter.layers)
    torch.testing.assert_close(cov.r1, _direct_cov(torch.cat(norm_rows)), atol=TOL, rtol=TOL)
    for cov_i, rows in zip(cov.r2, head_rows, strict=True):
        torch.testing.assert_close(cov_i, _direct_cov(torch.cat(rows)), atol=TOL, rtol=TOL)


def test_invalid_covariance():
    stats = WelfordCovariance(1)
    stats.update(torch.ones((1, 1)))
    with pytest.raises(ValueError, match="at least two"):
        stats.covariance()
    stats.update(torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="nonfinite"):
        stats.covariance()


def test_coat_orthogonality_and_uniform_variance(calibrated):
    cov, dims = calibrated
    plan = build_rotations(cov, dims, seed=0)
    for cov_i, q_mat in zip((cov.r1, *cov.r2), (plan.r1, *plan.r2), strict=True):
        d = q_mat.shape[0]
        h_mat = _sylvester(d)
        _, eigvecs = torch.linalg.eigh(cov_i)
        torch.testing.assert_close(q_mat, eigvecs @ h_mat, atol=TOL, rtol=TOL)
        torch.testing.assert_close(q_mat.T @ q_mat, torch.eye(d, dtype=q_mat.dtype), atol=TOL, rtol=TOL)
        cov_i = cov_i / torch.linalg.matrix_norm(cov_i)
        diag = (q_mat.T @ cov_i @ q_mat).diagonal()
        torch.testing.assert_close(diag, (cov_i.trace() / d).expand_as(diag), atol=TOL, rtol=TOL)


def test_structured_online_transforms(calibrated):
    cov, dims = calibrated
    plan = build_rotations(cov, dims, seed=42)
    repeat = build_rotations(cov, dims, seed=42)
    gen = torch.Generator().manual_seed(7)
    for name, width in (("r3", dims.head_dim), ("r4", dims.intermediate_size)):
        for i, transform in enumerate(getattr(plan, name)):
            x = torch.randn((2, 3, width), generator=gen, dtype=torch.float64)
            key = f"{name}.{i}.signs"
            torch.testing.assert_close(plan.state_dict()[key], repeat.state_dict()[key])
            h_mat = transform.signs[:, None] * _sylvester(width)
            torch.testing.assert_close(transform(x), x @ h_mat, atol=TOL, rtol=TOL)

    h_tail = _sylvester(4)[[1, 2, 0, 3]]
    x = torch.randn((2, 3, 16), generator=gen, dtype=torch.float64).transpose(0, 1)
    torch.testing.assert_close(Hadamard(4, h_tail)(x), x @ torch.kron(_sylvester(4), h_tail), atol=TOL, rtol=TOL)


def test_order_172_checksum_rejects_different_signs(tmp_path):
    path = tmp_path / "had.172.will.txt"
    path.write_text(("+" * 172 + "\n") * 172 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_hadamard172(path)


def test_order_44_factor_and_structured_5632_transform(calibrated, tmp_path):
    # Sloane's Paley factor uses a circulant block over the quadratic residues mod 43.
    residues = {i * i % 43 for i in range(1, 43)}
    rows = ["+" + "-" * 43] + [
        "+" + "".join("-" if (j - i) % 43 in residues else "+" for j in range(43))
        for i in range(43)
    ]
    path = tmp_path / "had.44.pal.txt"
    path.write_text("\n".join(rows) + "\n", encoding="ascii")
    h44 = load_hadamard44(path)
    torch.testing.assert_close(h44.T @ h44, torch.eye(44, dtype=h44.dtype), atol=TOL, rtol=TOL)

    cov, dims = calibrated
    plan = build_rotations(cov, replace(dims, intermediate_size=5632), seed=7, factors={44: path})
    r4 = plan.r4[0]
    x = torch.randn((3, 5632), dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    expected = _sylvester(128).T @ (x * r4.signs).reshape(3, 128, 44) @ h44
    torch.testing.assert_close(r4(x), expected.reshape_as(x), atol=TOL, rtol=TOL)

    rows[0] = "-" * 44
    path.write_text("\n".join(rows), encoding="ascii")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_hadamard44(path)


@torch.inference_mode()
def _logits(model, batch, prompt_len=32, decode_steps=4):
    out = model(
        input_ids=batch.input_ids[:, :prompt_len],
        attention_mask=batch.attention_mask[:, :prompt_len],
        use_cache=True,
    )
    logits = [out.logits]
    for i in range(prompt_len, prompt_len + decode_steps):
        out = model(
            input_ids=batch.input_ids[:, i:i + 1],
            attention_mask=batch.attention_mask[:, :i + 1],
            past_key_values=out.past_key_values,
            use_cache=True,
        )
        logits.append(out.logits)
    return torch.cat(logits, dim=1)


def _probe_cpu_fp16(model, batch):
    try:
        _logits(model, batch, decode_steps=1)
    except (RuntimeError, NotImplementedError) as exc:
        msg = str(exc).lower()
        if ("half" in msg or "float16" in msg) and any(
            reason in msg for reason in ("not implemented", "not supported", "unsupported")
        ):
            pytest.skip(f"CPU FP16 Llama runtime is unsupported: {exc}")
        raise


@pytest.mark.parametrize(("dtype", "tol"), [(torch.float32, 1e-4), (torch.float16, 5e-3)])
def test_fp_logits_preserved_on_held_out_tokens(calibrated, dtype, tol, record_property):
    cov, dims = calibrated
    baseline = load_llama(TINY_LLAMA_ID, seed=0, dtype=dtype)
    target = load_llama(TINY_LLAMA_ID, seed=0, dtype=dtype)
    batch = make_synthetic_batches(split="evaluation", seed=0, count=2, seq_len=36, batch_size=2)[0]
    if dtype == torch.float16:
        _probe_cpu_fp16(baseline.model, batch)
    rotated = fuse_llama(target.adapter, build_rotations(cov, dims, seed=0))
    expected = _logits(baseline.model, batch)
    actual = _logits(rotated, batch)
    error = (expected.float() - actual.float()).abs()
    max_error, mean_error = error.max().item(), error.mean().item()
    record_property("max_abs_logit_error", max_error)
    record_property("mean_abs_logit_error", mean_error)
    print(f"{dtype}: max absolute logit error {max_error:.6g}, mean {mean_error:.6g}")
    assert actual.shape == expected.shape == (2, 36, 128)
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    assert baseline.adapter.lm_head.weight is baseline.adapter.embeddings.weight
    assert target.adapter.lm_head.weight is not target.adapter.embeddings.weight
    assert not target.model.config.tie_word_embeddings
