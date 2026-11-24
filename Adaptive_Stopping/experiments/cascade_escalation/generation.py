"""Greedy generation with top-K log-probs, for Hugging Face and vLLM backends.

Both backends return, per prompt::

    {"response": str, "n_in": int, "n_out": int, "finish": "stop"|"length",
     "chosen_logprobs": [L floats], "topk_logprobs": [L lists of K floats]}

Decoding is greedy (temperature 0) with repetition penalty explicitly 1.0, so
the model's ``generation_config`` sampling defaults do not apply. Log-probs
are from the raw (unprocessed) next-token distribution. End-of-sequence
tokens are not counted in ``n_out`` and do not contribute to scores.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence


def chat_token_ids(tokenizer, prompt: str) -> List[int]:
    """Token ids of a single-turn chat prompt (falls back to plain text)."""
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
        if hasattr(ids, "keys"):  # BatchEncoding / dict in newer transformers
            ids = ids["input_ids"]
        if len(ids) and isinstance(ids[0], list):
            ids = ids[0]
        return [int(i) for i in ids]
    return [int(i) for i in tokenizer(prompt)["input_ids"]]


def eos_ids(tokenizer, generation_config=None) -> List[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    g = getattr(generation_config, "eos_token_id", None) if generation_config is not None else None
    if isinstance(g, int):
        ids.add(g)
    elif isinstance(g, (list, tuple)):
        ids.update(int(x) for x in g)
    for tok in ("<|im_end|>", "<|endoftext|>", "<|end|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(tok) if tok in tokenizer.get_vocab() else None
        if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
            ids.add(int(tid))
    return sorted(ids)


class HFGenerator:
    """Batched greedy generation with transformers; records top-K log-probs
    through a logits processor so full-vocabulary logits are never stored."""

    backend = "hf"

    def __init__(self, model_id: str, dtype: str = "bfloat16", device: Optional[str] = None,
                 top_k: int = 15, model=None, tokenizer=None, revision: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = int(top_k)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id, revision=revision)
        if model is None:
            torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, revision=revision)
        self.model = model.to(self.device).eval()
        self.eos = eos_ids(self.tokenizer, getattr(self.model, "generation_config", None))
        pad = self.tokenizer.pad_token_id
        self.pad_id = int(pad if pad is not None else self.eos[0])

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def _make_processor(self, store: list):
        torch = self.torch
        k = self.top_k

        class _Capture:
            def __call__(self, input_ids, scores):
                lp = torch.log_softmax(scores.float(), dim=-1)
                vals, _ = torch.topk(lp, k=min(k, lp.shape[-1]), dim=-1)
                store.append(vals.cpu())
                return scores

        return _Capture()

    def generate_ids(self, prompt_ids: Sequence[Sequence[int]], max_new_tokens: int) -> List[Dict]:
        torch = self.torch
        from transformers import LogitsProcessorList

        L = max(len(p) for p in prompt_ids)
        inp = torch.full((len(prompt_ids), L), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(prompt_ids), L), dtype=torch.long)
        for i, p in enumerate(prompt_ids):
            inp[i, L - len(p):] = torch.tensor(p, dtype=torch.long)
            att[i, L - len(p):] = 1
        store: list = []
        with torch.inference_mode():
            seq = self.model.generate(
                input_ids=inp.to(self.device), attention_mask=att.to(self.device),
                max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                logits_processor=LogitsProcessorList([self._make_processor(store)]),
                pad_token_id=self.pad_id, eos_token_id=self.eos,
            )
        gen = seq[:, L:].cpu()
        topk = torch.stack(store, dim=1) if store else torch.zeros((len(prompt_ids), 0, self.top_k))
        eos = set(self.eos)
        out = []
        for i in range(len(prompt_ids)):
            ids = gen[i].tolist()
            n = len(ids)
            finish = "length"
            for j, t in enumerate(ids):
                if t in eos:
                    n, finish = j, "stop"
                    break
            tk = topk[i, :n].tolist()
            out.append({
                "response": self.tokenizer.decode(ids[:n], skip_special_tokens=True),
                "n_in": len(prompt_ids[i]), "n_out": n, "finish": finish,
                "chosen_logprobs": [row[0] for row in tk],  # greedy => chosen = top-1
                "topk_logprobs": tk,
            })
        return out

    def generate(self, prompts: Sequence[str], max_new_tokens: int, batch_size: int = 16) -> List[Dict]:
        """Generate for many prompts (length-sorted batches); adds ``gpu_s``
        = batch wall time apportioned by (n_out + 1)."""
        ids = [chat_token_ids(self.tokenizer, p) for p in prompts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        results: List[Optional[Dict]] = [None] * len(ids)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            self._sync()
            t0 = time.perf_counter()
            res = self.generate_ids([ids[i] for i in idx], max_new_tokens)
            self._sync()
            dt = time.perf_counter() - t0
            w = sum(r["n_out"] + 1 for r in res)
            for i, r in zip(idx, res):
                r["gpu_s"] = dt * (r["n_out"] + 1) / w
                results[i] = r
        return results  # type: ignore[return-value]

    def generate_one_timed(self, prompt: str, max_new_tokens: int) -> Dict:
        """Single-request (batch size 1) generation with synchronised wall time."""
        ids = chat_token_ids(self.tokenizer, prompt)
        self._sync()
        t0 = time.perf_counter()
        r = self.generate_ids([ids], max_new_tokens)[0]
        self._sync()
        r["latency_s"] = time.perf_counter() - t0
        return r

    def _sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()


class VLLMGenerator:
    """Offline vLLM generation (continuous batching) with top-K log-probs."""

    backend = "vllm"

    def __init__(self, model_id: str, dtype: str = "bfloat16", top_k: int = 15,
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.85, seed: int = 0,
                 revision: Optional[str] = None):
        from vllm import LLM

        self.top_k = int(top_k)
        self.llm = LLM(model=model_id, dtype=dtype, max_model_len=max_model_len, seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization, max_logprobs=max(20, self.top_k),
                       revision=revision)
        self.tokenizer = self.llm.get_tokenizer()
        try:
            from transformers import GenerationConfig

            gc = GenerationConfig.from_pretrained(model_id, revision=revision)
        except Exception:
            gc = None
        self.eos = eos_ids(self.tokenizer, gc)

    def generate(self, prompts: Sequence[str], max_new_tokens: int, batch_size: int = 0) -> List[Dict]:
        from vllm import SamplingParams

        ids = [chat_token_ids(self.tokenizer, p) for p in prompts]
        sp = SamplingParams(temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_tokens=int(max_new_tokens),
                            logprobs=self.top_k, stop_token_ids=self.eos, seed=0)
        t0 = time.perf_counter()
        outs = self.llm.generate([{"prompt_token_ids": p} for p in ids], sp, use_tqdm=True)
        dt = time.perf_counter() - t0
        eos = set(self.eos)
        res = []
        for p, o in zip(ids, outs):
            c = o.outputs[0]
            tok = list(c.token_ids)
            lps = list(c.logprobs or [])
            n = len(tok)
            while n > 0 and tok[n - 1] in eos:
                n -= 1
            chosen, topk = [], []
            for j in range(n):
                d = lps[j]
                chosen.append(float(d[tok[j]].logprob))
                topk.append(sorted((float(v.logprob) for v in d.values()), reverse=True)[: self.top_k])
            res.append({
                "response": self.tokenizer.decode(tok[:n], skip_special_tokens=True),
                "n_in": len(p), "n_out": n,
                "finish": "stop" if c.finish_reason == "stop" else "length",
                "chosen_logprobs": chosen, "topk_logprobs": topk,
            })
        w = sum(r["n_out"] + 1 for r in res)
        for r in res:
            r["gpu_s"] = dt * (r["n_out"] + 1) / w
        return res
