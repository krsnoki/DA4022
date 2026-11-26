"""Hugging Face Transformers causal-LM wrapper implementing ``OptionScoringBackbone``.

Intended for CPU smoke runs with small checkpoints and for GPU runs when
vLLM is unavailable. Pin ``revision`` for any reported result.

Notes
-----
* ``render_prompt`` uses the tokenizer chat template when one exists
  (``enable_thinking=False`` is passed for templates that support it, so
  Qwen-style models answer directly), then appends ``assistant_prefix``.
* ``num_tokens`` counts generated tokens up to and including EOS.
* ``score_next_options`` returns the next-token log-probability of each
  label (log-sum-exp over the "A" and " A" first-token variants),
  renormalized over the labels. Its ``aux_tokens`` reports the full prompt
  length because this wrapper does not reuse a KV cache across calls; the
  controllers charge the paper-protocol estimate instead (see
  ``Adaptive_Stopping/common/rollout.py``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .interface import Generation, OptionScores, log_softmax, logsumexp


class HFCausalLM:
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        use_chat_template: bool = True,
        model_class: str = "AutoModelForCausalLM",
        model=None,
        tokenizer=None,
    ):
        import torch
        import transformers

        self.torch = torch
        self.model_id, self.revision, self.device, self.dtype = model_id, revision, device, dtype
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        if model is None:
            cls = getattr(transformers, model_class)
            model = cls.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype))
        self.tok = tokenizer
        self.model = model.to(device).eval()
        self.use_chat_template = use_chat_template and getattr(tokenizer, "chat_template", None) is not None
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self._eos = self._eos_ids()

    def _eos_ids(self) -> set:
        ids = set()
        for src in (self.tok.eos_token_id, getattr(self.model.generation_config, "eos_token_id", None)):
            if isinstance(src, int):
                ids.add(src)
            elif isinstance(src, (list, tuple)):
                ids.update(int(x) for x in src)
        return ids

    # ------------------------------------------------------------ interface
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
        n_params = sum(p.numel() for p in self.model.parameters())
        return {"id": self.model_id, "revision": self.revision, "dtype": self.dtype, "device": self.device,
                "num_parameters": n_params, "backend": "transformers",
                "transformers_version": __import__("transformers").__version__,
                "torch_version": self.torch.__version__}

    def _encode(self, prompt: str):
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=not self.use_chat_template)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def generate(self, prompt: str, n: int, max_tokens: int, temperature: float, seed: Optional[int] = None) -> List[Generation]:
        torch = self.torch
        if n <= 0 or max_tokens <= 0:
            return []
        ids, mask = self._encode(prompt)
        if seed is not None:
            torch.manual_seed(seed)
        do_sample = temperature is not None and temperature > 0
        kwargs = dict(max_new_tokens=max_tokens, num_return_sequences=n, do_sample=do_sample,
                      pad_token_id=self.tok.pad_token_id)
        if do_sample:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        with torch.no_grad():
            seqs = self.model.generate(input_ids=ids, attention_mask=mask, **kwargs)
        out = []
        for row in seqs[:, ids.shape[1]:].tolist():
            toks, finished = [], False
            for t in row:
                toks.append(t)
                if t in self._eos:
                    finished = True
                    break
            text = self.tok.decode(toks, skip_special_tokens=True)
            out.append(Generation(text=text, num_tokens=len(toks), finished=finished, token_ids=toks))
        return out

    def _label_token_ids(self, label: str) -> List[int]:
        ids = set()
        for variant in (label, " " + label):
            enc = self.tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def score_next_options(self, prompt: str, labels: Sequence[str]) -> OptionScores:
        torch = self.torch
        ids, mask = self._encode(prompt)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        raw = {lab: logsumexp([lp[i].item() for i in self._label_token_ids(lab)]) for lab in labels}
        return OptionScores(log_softmax(raw), aux_tokens=int(ids.shape[1]))


__all__ = ["HFCausalLM"]
