"""vLLM wrapper implementing ``OptionScoringBackbone`` (paper serving stack).

Status: written against the public vLLM offline ``LLM`` API but **not run in
this repository's CI**, which has no GPU. Verify on your hardware with
``Adaptive_Stopping/tests/test_backbones_optional.py`` before reporting.

* Prefix caching is enabled so repeated root prompts and scoring suffixes
  reuse the KV cache, matching the paper's token accounting.
* ``score_next_options`` requests the top ``max_logprobs`` next-token
  log-probabilities; a label outside that list gets the smallest returned
  log-probability minus ``log(10)``. Increase ``max_logprobs`` if this
  happens often (it is logged in ``self.missing_label_events``).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .interface import Generation, OptionScores, log_softmax, logsumexp


class VLLMCausalLM:
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: str = "bfloat16",
        seed: int = 13,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_logprobs: int = 64,
        use_chat_template: bool = True,
    ):
        from vllm import LLM, SamplingParams  # noqa: F401

        self._SamplingParams = SamplingParams
        self.model_id, self.revision, self.dtype = model_id, revision, dtype
        self.max_logprobs = max_logprobs
        self.llm = LLM(
            model=model_id, revision=revision, dtype=dtype, seed=seed, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True, max_logprobs=max_logprobs,
        )
        self.tok = self.llm.get_tokenizer()
        self.use_chat_template = use_chat_template and getattr(self.tok, "chat_template", None) is not None
        self.missing_label_events = 0

    def render_prompt(self, user_message: str, assistant_prefix: str = "") -> str:
        if self.use_chat_template:
            msgs = [{"role": "user", "content": user_message}]
            try:
                head = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                head = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return head + assistant_prefix
        return user_message + "\n" + assistant_prefix

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def get_model_metadata(self) -> dict:
        import vllm

        return {"id": self.model_id, "revision": self.revision, "dtype": self.dtype, "backend": "vllm",
                "vllm_version": vllm.__version__}

    def generate(self, prompt: str, n: int, max_tokens: int, temperature: float, seed: Optional[int] = None) -> List[Generation]:
        if n <= 0 or max_tokens <= 0:
            return []
        sp = self._SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=seed)
        res = self.llm.generate([prompt], sp, use_tqdm=False)[0]
        return [
            Generation(text=o.text, num_tokens=len(o.token_ids), finished=o.finish_reason != "length",
                       token_ids=list(o.token_ids))
            for o in res.outputs
        ]

    def _label_token_ids(self, label: str) -> List[int]:
        ids = set()
        for variant in (label, " " + label):
            enc = self.tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def score_next_options(self, prompt: str, labels: Sequence[str]) -> OptionScores:
        sp = self._SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=self.max_logprobs)
        res = self.llm.generate([prompt], sp, use_tqdm=False)[0]
        top = res.outputs[0].logprobs[0]  # dict token_id -> Logprob
        vals = {tid: lp.logprob for tid, lp in top.items()}
        floor = min(vals.values()) - math.log(10.0)
        raw = {}
        for lab in labels:
            got = [vals[t] for t in self._label_token_ids(lab) if t in vals]
            if not got:
                self.missing_label_events += 1
                got = [floor]
            raw[lab] = logsumexp(got)
        return OptionScores(log_softmax(raw), aux_tokens=len(res.prompt_token_ids))


__all__ = ["VLLMCausalLM"]
