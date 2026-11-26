from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
from torch import Tensor
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerBase
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

TINY_LLAMA_ID = "llama.tiny.synthetic"
LLAMA2_7B_ID = "meta-llama/Llama-2-7b-hf"
TINYLLAMA_1B_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass(frozen=True)
class LlamaMetadata:
    id: str
    revision: str
    tokenizer_revision: str | None
    license: str | None
    pretraining_regime: str
    objective: str | None


LLAMA_REGISTRY = {
    TINY_LLAMA_ID: LlamaMetadata(TINY_LLAMA_ID, "v1", None, None, "untrained", None),
    LLAMA2_7B_ID: LlamaMetadata(
        id=LLAMA2_7B_ID,
        revision="01c7f73d771dfac7d292323805ebc428287df4f9",
        tokenizer_revision="01c7f73d771dfac7d292323805ebc428287df4f9",
        license="llama2",
        pretraining_regime="self_supervised",
        objective="next-token prediction",
    ),
    TINYLLAMA_1B_ID: LlamaMetadata(
        id=TINYLLAMA_1B_ID,
        revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        tokenizer_revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        license="apache-2.0",
        pretraining_regime="self_supervised",
        objective="next-token pretraining, supervised chat fine-tuning, and preference optimization",
    ),
}

QKTransform = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


def _attention_with_post_rope(
    attn: LlamaAttention,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
    past_key_value: Cache | None = None,
    cache_position: Tensor | None = None,
    *,
    transform: QKTransform,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    batch, seq, _ = hidden_states.shape
    shape = (batch, seq, -1, attn.head_dim)
    q, k, v = (
        proj(hidden_states).reshape(shape).transpose(1, 2)
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
    )
    cos, sin = position_embeddings
    q, k = transform(*apply_rotary_pos_emb(q, k, cos, sin))
    if past_key_value is not None:
        k, v = past_key_value.update(
            k, v, attn.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position}
        )
    out, weights = eager_attention_forward(
        attn, q, k, v, attention_mask,
        scaling=attn.scaling,
        dropout=attn.attention_dropout if attn.training else 0.0,
        **kwargs,
    )
    return attn.o_proj(out.reshape(batch, seq, -1)), weights


@dataclass(frozen=True)
class LlamaAdapter:
    model: LlamaForCausalLM

    @property
    def embeddings(self):
        return self.model.model.embed_tokens

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def final_norm(self):
        return self.model.model.norm

    @property
    def lm_head(self):
        return self.model.lm_head

    def norms(self, layer_idx: int):
        layer = self.layers[layer_idx]
        return layer.input_layernorm, layer.post_attention_layernorm

    def projections(self, layer_idx: int) -> dict[str, torch.nn.Module]:
        layer = self.layers[layer_idx]
        return {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }

    @contextmanager
    def post_rope(self, layer_idx: int, transform: QKTransform) -> Iterator[None]:
        """Transform [B,H,T,D] queries and new keys before the cache update."""
        attn = self.layers[layer_idx].self_attn
        if attn.config._attn_implementation != "eager":
            raise ValueError("post_rope requires eager Llama attention")
        previous = vars(attn).get("forward")

        def forward(module, *args, **kwargs):
            return _attention_with_post_rope(module, *args, transform=transform, **kwargs)

        attn.forward = MethodType(forward, attn)
        try:
            yield
        finally:
            if previous is None:
                del attn.forward
            else:
                attn.forward = previous

    @contextmanager
    def gated_mlp_output(self, layer_idx: int, transform: Callable[[Tensor], Tensor]) -> Iterator[None]:
        def apply(_, args):
            return (transform(args[0]),)

        with self.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(apply):
            yield


@dataclass(frozen=True)
class LlamaBackbone:
    model: LlamaForCausalLM
    tokenizer: PreTrainedTokenizerBase | None
    metadata: LlamaMetadata

    @property
    def adapter(self) -> LlamaAdapter:
        return LlamaAdapter(self.model)


def _tiny_llama(seed: int) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        model = LlamaForCausalLM(cfg)
    norms = [model.model.norm]
    for layer in model.model.layers:
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
    with torch.no_grad():
        for i, norm in enumerate(norms):
            gain = torch.linspace(0.8, 1.2, cfg.hidden_size, device=norm.weight.device)
            norm.weight.copy_(gain.roll(i))
    return model


def load_llama(
    backbone_id: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    cache_dir: str | Path | None = None,
) -> LlamaBackbone:
    meta = LLAMA_REGISTRY[backbone_id]
    if backbone_id == TINY_LLAMA_ID:
        model = _tiny_llama(seed).to(dtype=dtype)
        tok = None
    else:
        tok = AutoTokenizer.from_pretrained(
            meta.id, revision=meta.tokenizer_revision, cache_dir=cache_dir, use_fast=True
        )
        model = LlamaForCausalLM.from_pretrained(
            meta.id, revision=meta.revision, cache_dir=cache_dir, torch_dtype=dtype,
            attn_implementation="eager", use_safetensors=True,
        )
    return LlamaBackbone(model.to(device=device).eval(), tok, meta)
