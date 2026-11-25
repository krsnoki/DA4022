import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from Adaptive_Stopping.methods.learned_exit.terminator.generation import (  # noqa: E402
    EXIT_MESSAGE,
    exit_ids,
    generate_with_exit,
    sample_token,
)
from Adaptive_Stopping.methods.learned_exit.terminator.policy import (  # noqa: E402
    FixedBudgetPolicy,
    TerminatorPolicy,
    exit_index,
)
from Adaptive_Stopping.methods.learned_exit.terminator.probe import (  # noqa: E402
    TerminatorProbe,
    base_hidden_states,
    probe_input,
)


def test_probe_shapes_and_dtype(tiny_qwen3):
    probe = TerminatorProbe(tiny_qwen3)
    ids = torch.randint(4, 90, (2, 17))
    h = base_hidden_states(tiny_qwen3, ids)
    assert h.shape == (2, 17, 32)
    logits = probe(h)
    assert logits.shape == (2, 17)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_probe_copies_last_layer(tiny_qwen3):
    probe = TerminatorProbe(tiny_qwen3)
    src = tiny_qwen3.model.layers[-1].state_dict()
    for k, v in probe.layer.state_dict().items():
        assert torch.equal(v, src[k])
    # the copy is independent of the frozen base layer
    with torch.no_grad():
        next(probe.layer.parameters()).add_(1.0)
    assert not torch.equal(next(probe.layer.parameters()), next(tiny_qwen3.model.layers[-1].parameters()))


def test_probe_is_causal(tiny_qwen3):
    probe = TerminatorProbe(tiny_qwen3)
    ids = torch.randint(4, 90, (1, 20))
    ids2 = ids.clone()
    ids2[0, 12:] = torch.randint(4, 90, (8,))
    a = probe(base_hidden_states(tiny_qwen3, ids))
    b = probe(base_hidden_states(tiny_qwen3, ids2))
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-5)


def test_probe_backward(tiny_qwen3):
    probe = TerminatorProbe(tiny_qwen3)
    h = base_hidden_states(tiny_qwen3, torch.randint(4, 90, (1, 9)))
    y = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(h), y)
    loss.backward()
    assert probe.head.network.weight.grad is not None
    assert all(p.grad is None for p in tiny_qwen3.parameters())


def test_sample_token_greedy_and_topk():
    logits = torch.tensor([0.0, 5.0, 1.0, 4.0])
    assert sample_token(logits, 0.0, 1.0, 0, None) == 1
    g = torch.Generator().manual_seed(0)
    picks = {sample_token(logits, 1.0, 1.0, 2, g) for _ in range(50)}
    assert picks <= {1, 3}


def _prompt(tok):
    return tok("Q: 1+1?\n")["input_ids"] + [tok.convert_tokens_to_ids("<think>")]


def test_cached_probe_matches_full_pass(tiny_qwen3, char_tokenizer):
    """Online probe with its own KV cache gives the same probabilities as one
    pass over the finished trace, so exits can be computed offline."""
    probe = TerminatorProbe(tiny_qwen3).eval()
    prompt = _prompt(char_tokenizer)
    out = generate_with_exit(tiny_qwen3, char_tokenizer, prompt, probe=probe, policy=None,
                             max_think=25, max_answer=3, temperature=0.0)
    assert len(out["think_ids"]) == 25 and len(out["probs"]) == 25
    ids = torch.tensor([prompt + out["think_ids"]])
    with torch.no_grad():
        full = torch.sigmoid(probe(base_hidden_states(tiny_qwen3, ids)))[0, len(prompt):]
    assert torch.allclose(full, torch.tensor(out["probs"]), atol=1e-4)


def test_end_to_end_early_exit(tiny_qwen3, char_tokenizer):
    probe = TerminatorProbe(tiny_qwen3).eval()
    with torch.no_grad():
        probe.head.network.weight.zero_()
        probe.head.network.bias.fill_(10.0)  # p ~ 1 at every token
    prompt = _prompt(char_tokenizer)
    out = generate_with_exit(tiny_qwen3, char_tokenizer, prompt, probe=probe,
                             policy=TerminatorPolicy(0.7, 10), max_think=50, max_answer=4,
                             temperature=0.0)
    assert out["stopped_early"] and out["think_end"] == "probe"
    assert out["exit_pos"] == 10 == len(out["think_ids"])
    assert exit_index(out["probs"], 0.7, 10) == 10
    assert len(out["answer_ids"]) == 4

    with torch.no_grad():
        probe.head.network.bias.fill_(-10.0)  # p ~ 0, never fires
    out = generate_with_exit(tiny_qwen3, char_tokenizer, prompt, probe=probe,
                             policy=TerminatorPolicy(0.7, 10), max_think=30, max_answer=2,
                             temperature=0.0)
    assert not out["stopped_early"] and out["think_end"] == "cap"
    assert len(out["think_ids"]) == 30


def test_budget_edges_in_loop(tiny_qwen3, char_tokenizer):
    prompt = _prompt(char_tokenizer)
    kw = dict(max_think=20, max_answer=2, temperature=0.0)
    out = generate_with_exit(tiny_qwen3, char_tokenizer, prompt, policy=FixedBudgetPolicy(), budget=0, **kw)
    assert out["think_ids"] == [] and out["think_end"] == "budget"
    out = generate_with_exit(tiny_qwen3, char_tokenizer, prompt, policy=FixedBudgetPolicy(), budget=1, **kw)
    assert len(out["think_ids"]) == 1 and out["stopped_early"]
    out = generate_with_exit(tiny_qwen3, char_tokenizer, prompt, policy=FixedBudgetPolicy(), budget=10**6, **kw)
    assert len(out["think_ids"]) == 20 and out["think_end"] == "cap"
    # TERMINATOR policy without a probe: missing telemetry, only the budget stops
    out = generate_with_exit(tiny_qwen3, char_tokenizer, prompt, policy=TerminatorPolicy(), budget=7, **kw)
    assert len(out["think_ids"]) == 7 and out["policy_report"]["n_missing"] == 7


def test_exit_ids(char_tokenizer):
    ids = exit_ids(char_tokenizer)
    assert ids[-1] == char_tokenizer.convert_tokens_to_ids("</think>")
    assert char_tokenizer.decode(ids[:-1]) == EXIT_MESSAGE


def test_fp32_probe_avoids_fp16_overflow(tiny_qwen3):
    """With large MLP weights (like the last layer of a real Qwen3) the probe
    overflows in fp16 and the head's RMSNorm turns inf into NaN. The float32
    probe used for training stays finite on the same fp16 hidden states."""
    import copy

    probe = TerminatorProbe(tiny_qwen3)
    with torch.no_grad():
        probe.layer.mlp.down_proj.weight.mul_(3000.0)
        probe.layer.mlp.up_proj.weight.mul_(100.0)
        probe.layer.mlp.gate_proj.weight.mul_(100.0)
    torch.manual_seed(0)
    h16 = base_hidden_states(tiny_qwen3, torch.randint(4, 90, (1, 40))).half()
    with torch.no_grad():
        z16 = copy.deepcopy(probe).half()(h16)
        z32 = probe(probe_input(h16)[0])
    assert not torch.isfinite(z16).all()
    assert torch.isfinite(z32).all() and z32.dtype == torch.float32


def test_probe_input_replaces_non_finite():
    h = torch.ones(1, 4, 3, dtype=torch.float16)
    h[0, 1, 0] = float("inf")
    h[0, 2, 2] = float("nan")
    x, n_bad = probe_input(h)
    assert n_bad == 2 and x.dtype == torch.float32 and torch.isfinite(x).all()
    assert probe_input(torch.ones(1, 4, 3))[1] == 0


def test_train_step_gives_finite_loss_and_grads(tiny_qwen3):
    from Adaptive_Stopping.experiments.terminator.train_probe import train_step

    probe = TerminatorProbe(tiny_qwen3).train()
    trace = {"prompt_ids": [5, 6, 7, 1], "think_ids": list(range(10, 30)),
             "labels": [0] * 8 + [1] * 12}
    counter = {"traces": 0, "bad_positions": 0, "traces_with_bad": 0}
    loss = train_step(tiny_qwen3, probe, trace, 1.25, 0.83, torch.device("cpu"), counter)
    assert loss == loss and loss > 0
    grads = [p.grad for p in probe.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for p in tiny_qwen3.parameters())
    assert counter == {"traces": 1, "bad_positions": 0, "traces_with_bad": 0}
