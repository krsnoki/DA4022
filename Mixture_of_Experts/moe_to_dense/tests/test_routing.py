import torch

from Mixture_of_Experts.moe_to_dense.synthetic import TinyMoELayer


def test_sparse_dispatch_matches_masked_reference_and_only_runs_selected_tokens():
    torch.manual_seed(13)
    model = TinyMoELayer(8, 4, 4, 2)
    hidden = torch.randn(3, 5, 8)
    probs, selected = model.route(hidden)
    mask = torch.zeros_like(probs).scatter_(-1, selected, 1.0)
    weights = probs * mask
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = (model.all_expert_outputs(hidden) * weights.unsqueeze(-1)).sum(dim=-2)
    counts = []
    hooks = [expert.register_forward_pre_hook(lambda module, args: counts.append(args[0].shape[0])) for expert in model.experts]
    try:
        actual = model(hidden)
    finally:
        for hook in hooks:
            hook.remove()
    assert sum(counts) == hidden.shape[0] * hidden.shape[1] * model.top_k
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model.router.weight.grad is not None
