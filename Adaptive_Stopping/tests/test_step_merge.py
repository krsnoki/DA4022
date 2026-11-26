import random

import pytest

from Adaptive_Stopping.common.replay import replay_policy
from Adaptive_Stopping.common.step_merge import StepMerger, iter_merged, merge_steps
from Adaptive_Stopping.common.steps import Step, segment_text_pieces
from Adaptive_Stopping.methods.step_tagging.policy import StepTagEarlyStopping

TRACE = ("Okay, so I need x.\n\nOk.\n\nFirst, 2 + 3 = 5 and then more words here.\n\nHm.\n\n"
         "Wait, let me check this carefully again.\n\nYes.\n</think>\n\nThe answer is 5.\n\nDone")


def steps_for(text):
    return segment_text_pieces(list(text))  # one character per "token"


@pytest.mark.parametrize("k", [None, 0])
def test_no_merge_when_k_is_none_or_zero(k):
    raw = steps_for(TRACE)
    assert [s.text for s in merge_steps(raw, k)] == [s.text for s in raw]


@pytest.mark.parametrize("k", [0, 5, 10, 25, 60, 10_000])
def test_merging_preserves_text_and_respects_k(k):
    raw = steps_for(TRACE)
    merged = merge_steps(raw, k)
    assert "\n\n".join(s.text for s in merged) == TRACE      # nothing lost or duplicated
    prev = 0
    for s in merged[:-1]:
        assert s.closed and s.end_token - prev > k            # |s_i| > k for every closed step
        prev = s.end_token
    assert not merged[-1].closed and merged[-1].end_token == len(TRACE)
    assert [s.index for s in merged] == list(range(len(merged)))


def test_short_pieces_merge_into_the_next_step():
    raw = steps_for("Ok.\n\nHm.\n\nA much longer step of text.\n\nend")
    # "Ok.\n\n" = 5 tokens, "Ok.\n\nHm.\n\n" = 10 tokens: with k = 10 neither is > k, so both merge forward
    merged = merge_steps(raw, 10)
    assert merged[0].text == "Ok.\n\nHm.\n\nA much longer step of text."
    assert merged[1].text == "end"
    # with k = 8 the second piece already makes the step 10 > 8 tokens long
    assert merge_steps(raw, 8)[0].text == "Ok.\n\nHm."


def test_thinking_flag_is_false_if_any_part_left_thinking():
    raw = steps_for("Yes.\n</think>\n\nAnswer here.\n\nx")
    merged = merge_steps(raw, 100)
    assert len(merged) == 1 and merged[0].in_thinking is False


@pytest.mark.parametrize("seed", range(5))
def test_online_merger_equals_offline(seed):
    rng = random.Random(seed)
    pos, raw = 0, []
    for i in range(30):
        pos += rng.randint(0, 12)
        raw.append(Step(i, f"s{i}", pos, in_thinking=True))
    tail = Step(30, "tail", pos + 3, True, closed=False)
    k = rng.randint(0, 20)
    offline = merge_steps(raw + [tail], k)
    online = list(iter_merged(raw, k))
    m = StepMerger(k)
    for s in raw:
        m.push(s)
    assert online == offline[:-1]
    assert m.flush(tail) == offline[-1]


def test_policy_on_merged_steps_can_only_stop_at_original_boundaries():
    raw = steps_for(TRACE)
    boundaries = {s.end_token for s in raw}
    for k in (0, 5, 20):
        merged = merge_steps(raw, k)
        out = replay_policy(StepTagEarlyStopping("verification", 0), merged, len(TRACE))
        if out.stopped:
            assert out.cut_tokens in boundaries


@pytest.mark.parametrize("bad", [-1, 2.5, "3"])
def test_invalid_k_rejected(bad):
    with pytest.raises(ValueError):
        StepMerger(bad)
