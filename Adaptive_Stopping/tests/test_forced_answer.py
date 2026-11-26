import torch

from Adaptive_Stopping.common.forced_answer import BoxCloseCriteria, _batches, brace_delta


class CharTok:
    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)


def test_brace_delta_ignores_escaped_braces():
    assert brace_delta("{") == 1 and brace_delta("}}") == -2
    assert brace_delta("\\{1,2\\}") == 0
    assert brace_delta("\\frac{1}{2}") == 0


def test_box_close_criteria_per_row():
    vocab = ["1", "}", "{", "x", "<eos>"]
    tok = CharTok(vocab)
    crit = BoxCloseCriteria(tok, eos_ids=[4], start_len=2, batch_size=3)
    seqs = [[3, 3], [3, 3], [3, 3]]  # two prompt tokens per row
    steps = [[0, 1, 0], [2, 1, 4], [3, 3, 3]]
    # row 0: "1" "}"       -> closes at 2nd new token
    # row 1: "{" "}" EOS   -> depth 2,1 then EOS at 3rd token (answer = 2 tokens)
    # row 2: "x" "x" "x"   -> never closes
    for t in range(3):
        for r in range(3):
            seqs[r].append(steps[r][t])
        done = crit(torch.tensor(seqs))
    assert crit.stop_len == [2, 2, None]
    assert done.tolist() == [True, True, False]


def test_batches_respect_budgets():
    lengths = [100, 90, 80, 70, 10, 10]
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out = _batches(order, lengths, max_batch=2, kv_budget=10_000, prefill_budget=10_000, extra=5)
    assert all(len(b) <= 2 for b in out) and sorted(i for b in out for i in b) == list(range(6))
    out = _batches(order, lengths, max_batch=10, kv_budget=10_000, prefill_budget=150, extra=5)
    assert all(len(b) * max(lengths[i] for i in b) <= 150 or len(b) == 1 for b in out)
