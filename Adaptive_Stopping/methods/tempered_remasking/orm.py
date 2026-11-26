"""Outcome reward model (ORM) scoring for Best@k selection (Sec. 4.2).

The paper uses AceMath-7B (Liu et al., 2024) as ORM; the reward-model
checkpoint is ``nvidia/AceMath-7B-RM``. The usage below follows that model
card (system prompt + user question + assistant response through the chat
template; the scalar score is ``outputs[0][0]``).
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import torch

__all__ = ["ORMScorer", "HFRewardModelScorer", "ACEMATH_RM_ID"]

ACEMATH_RM_ID = "nvidia/AceMath-7B-RM"
_ACEMATH_SYSTEM = "Please reason step by step, and check your final answer within \\boxed{}."


class ORMScorer(Protocol):
    def score(self, question: str, responses: Sequence[str]) -> List[float]:
        """Return one scalar score per response (higher = more likely correct)."""


class HFRewardModelScorer:
    """Sequence-classification reward model with a chat template."""

    def __init__(
        self,
        model_id: str = ACEMATH_RM_ID,
        revision: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        system_prompt: Optional[str] = _ACEMATH_SYSTEM,
        max_length: int = 4096,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_id, revision=revision, num_labels=1, torch_dtype=dtype, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.system_prompt = system_prompt
        self.max_length = max_length

    @torch.no_grad()
    def score(self, question: str, responses: Sequence[str]) -> List[float]:
        out = []
        for resp in responses:
            chat = []
            if self.system_prompt:
                chat.append({"role": "system", "content": self.system_prompt})
            chat += [{"role": "user", "content": question}, {"role": "assistant", "content": resp}]
            text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)[:, -self.max_length :]
            logits = self.model(input_ids=ids.to(self.device))[0]
            out.append(float(logits.reshape(-1)[0]))
        return out
