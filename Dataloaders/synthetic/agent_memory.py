"""Seeded scripted tool-observation trajectories; not BFCL or tau2-bench.

Tool calls and observations are supplied fixtures. The model generates live
answers on two turns. This measures retention across scripted tool interactions,
not autonomous tool selection, API execution, or general agent capability.
"""

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class MemoryEpisode:
    sample_id: str
    first_prompt: str
    continuation: str
    expected: str


def episodes(count=20, seed=42, distractors=48):
    if type(count) is not int or count <= 0 or type(distractors) is not int or distractors < 0:
        raise ValueError("count must be positive and distractors nonnegative")
    rng = random.Random(seed)
    for index in range(count):
        code = str(rng.randrange(100000, 1000000))
        noise = "\n".join(f"Unrelated order {j}: reference {rng.randrange(100000, 1000000)}; status archived." for j in range(distractors))
        prompt = (
            "<|im_start|>system\nYou are a careful assistant. Remember tool results across turns. "
            "When asked for a code, answer with only its six digits.<|im_end|>\n"
            "<|im_start|>user\nLook up the collection code for order ORION.<|im_end|>\n"
            '<|im_start|>assistant\n<think>I need the order lookup result.</think>\n'
            '<tool_call>{"name":"lookup_order","arguments":{"order":"ORION"}}</tool_call><|im_end|>\n'
            f'<|im_start|>tool\n<tool_response>{{"order":"ORION","collection_code":"{code}"}}</tool_response><|im_end|>\n'
            "<|im_start|>user\nKeep that code for later. Say OK.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        continuation = (
            "\n<|im_start|>user\nCheck the archive, then give me ORION's collection code from the earlier lookup.<|im_end|>\n"
            '<|im_start|>assistant\n<think>I will check the archive.</think>\n'
            '<tool_call>{"name":"list_archive","arguments":{}}</tool_call><|im_end|>\n'
            f"<|im_start|>tool\n<tool_response>{noise}</tool_response><|im_end|>\n"
            "<|im_start|>user\nWhat is ORION's collection code? Answer only the six digits.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        yield MemoryEpisode(f"agent_memory_v1:{seed}:{index}", prompt, continuation, code)


def exact_code_match(text, expected):
    # Strip a completed reasoning span, but do not credit an answer appearing
    # only in an unfinished reasoning trace or buried in arbitrary prose.
    if "</think>" not in text:
        return False
    answer = text.rsplit("</think>", 1)[-1]
    answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return answer == expected


def answer_diagnostics(text, expected, truncated):
    """Explain strict scoring without changing it or crediting reasoning text."""
    correct = exact_code_match(text, expected)
    reason = "correct" if correct else (
        "truncated" if truncated else (
            "missing_final_delimiter" if "</think>" not in text else "wrong_final_answer"
        )
    )
    return {"correct": correct, "answer_category": reason,
            "truncated": bool(truncated), "has_final_delimiter": "</think>" in text}
