"""Prompt templates for MoA aggregation (paper Table 2) and prefix caching (Table 6).

Provenance:

* ``MOA_AGGREGATOR_SYSTEM``: identical, sentence by sentence, to the paper's Table 2
  system message (checked against arXiv 2510.05059v1 HTML, Appendix C). This text is
  also the aggregator prompt of the original MoA work (Wang et al., 2024).
* ``PREFIX_CACHE_SYSTEM``: **meaning-level reconstruction** of Table 6, not a verbatim
  copy. Sentences 1, 2, 3, 5 and 6 follow the paper; sentence 4 (the "can be easily
  updated or expanded" instruction) is paraphrased. Recorded in PAPER_CARD
  "Differences".

Messages are OpenAI-style ``[{"role": ..., "content": ...}]`` lists so that any chat
template (vLLM server side, or Hugging Face tokenizer) can render them.
"""

from __future__ import annotations

from collections.abc import Sequence

Messages = list[dict[str, str]]

MOA_AGGREGATOR_SYSTEM = (
    "You have been provided with a set of responses from various open-source models to "
    "the latest user query. Your task is to synthesize these responses into a single, "
    "high-quality response. It is crucial to critically evaluate the information "
    "provided in these responses, recognizing that some of it may be biased or "
    "incorrect. Your response should not simply replicate the given answers but should "
    "offer a refined, accurate, and comprehensive reply to the instruction. Ensure your "
    "response is well-structured, coherent, and adheres to the highest standards of "
    "accuracy and reliability."
)

PREFIX_CACHE_SYSTEM = (
    "You have been provided with a set of responses from various open-source models to "
    "the latest user query in chunks. Your task is to synthesize these response chunks "
    "into a single, high-quality response. It is crucial to critically evaluate the "
    "information provided in these responses, recognizing that some of it may be biased "
    "or incorrect. Since some of the responses may not be complete yet, write your "
    "response so that it can be easily updated or expanded as new information arrives. "
    "Your response should not simply replicate the given answers but should offer a "
    "refined, accurate, and comprehensive reply to the instruction. Ensure your response "
    "is well-structured, coherent, and adheres to the highest standards of accuracy and "
    "reliability."
)


def proposer_messages(query: str, system: str | None = None) -> Messages:
    """Messages sent to a proposer: just the user query (optional system prompt)."""
    msgs: Messages = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": query})
    return msgs


def aggregator_messages(query: str, responses: Sequence[str]) -> Messages:
    """Table 2: numbered list of each proposer's (cumulative) response, then the query.

    Empty responses (e.g. proposers skipped by redundancy in the first chunk) are
    left out and the remaining ones are numbered consecutively.
    """
    items = [r for r in responses if r.strip()]
    listing = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(items))
    system = f"{MOA_AGGREGATOR_SYSTEM}\n\nResponses from models:\n{listing}"
    return [{"role": "system", "content": system}, {"role": "user", "content": query}]


def prefix_cache_messages(query: str, chunk_blocks: Sequence[Sequence[str]]) -> Messages:
    """Table 6: responses grouped by chunk index; new chunks are appended at the end.

    ``chunk_blocks[j][i]`` is the new text of proposer ``i`` in chunk ``j``. Because
    earlier blocks never change, the system prompt of call ``j`` is a strict string
    prefix of the system prompt of call ``j + 1``, so a prefix cache can reuse it.
    Models with no new text in a block are omitted from that block.
    """
    parts = [PREFIX_CACHE_SYSTEM, "", "Responses from models:"]
    for j, block in enumerate(chunk_blocks):
        parts.append(f"Chunk {j + 1}:")
        for i, text in enumerate(block):
            if text:
                parts.append(f"Model {i + 1}: {text}")
    system = "\n".join(parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": query}]
