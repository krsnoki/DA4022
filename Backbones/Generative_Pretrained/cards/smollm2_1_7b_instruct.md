# Backbone model card: HuggingFaceTB/SmolLM2-1.7B-Instruct

- **Registry ID:** `text.smollm2_1_7b_instruct`
- **Modality:** text
- **Architecture:** LlamaForCausalLM (model_type llama)
- **Pretraining regime:** mixed (self-supervised next-token pretraining, then supervised/preference post-training)
- **Training objective/data:** base model trained on 11T tokens; instruct model via SFT then DPO on UltraFeedback (model card)
- **Weights URI and exact revision:** https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct @ `31b70e2e869a7173562077fd711b654946d38674`
- **License/usage terms:** Apache-2.0 (HF metadata), ungated
- **Tokenizer/processor revision:** bundled tokenizer and chat template at the same revision
- **Parent backbone (if fine-tuned):** unknown (not recorded in the fetched metadata)
- **Input/output contract:** chat messages in, generated text tokens out (OpenAI-compatible chat API through vLLM); context: unknown (not stated on the model card)
- **Parameter count / weight size:** 1,711,376,384 parameters (safetensors metadata); 3,422,777,952 bytes `model.safetensors` (BF16)

## Loading

Used in `Serving/staircase_streaming` as **proposer 2**, served by vLLM (command built by `launch_vllm.py`):

```bash
vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct --revision 31b70e2e869a7173562077fd711b654946d38674 --served-model-name <id> --port <port> --dtype auto
```

Tested runtime (staircase-streaming benchmark, 2026-11-24): vLLM 0.34.0, PyTorch 2.14.0 (CUDA 13.0 build), transformers 5.22.2, FlashAttention 2 backend, dtype bfloat16 (`--dtype auto`), on 1× NVIDIA GeForce RTX 4090 24 GB (driver 595.84, CUDA driver 13.2), Python 3.11.10, shared with two other vLLM servers. Run with `VLLM_USE_FLASHINFER_SAMPLER=0` (see `Serving/staircase_streaming/remote/RUNBOOK.md` §5).

## Preprocessing and outputs

The server applies the checkpoint's own chat template. Continuing a partial assistant answer uses `continue_final_message=True` with `add_generation_prompt=False`. Outputs are streamed text deltas with token usage counts.

## Smoke test

```bash
STAIRCASE_LIVE_CONFIG=Serving/staircase_streaming/configs/default.yaml python -m pytest Serving/staircase_streaming/tests -m slow -q
```

Expected: every server returns at least one token for a short prompt, and the aggregator continues a partial answer.

## Caveats

- This is a small model. Answer quality is far below the 7–9B models used in the paper.
- Per-model memory use and prefill/decode speed were not measured separately. End-to-end MoA latency, throughput and peak device memory (all three servers on one RTX 4090) are reported in `Serving/staircase_streaming/PAPER_CARD.md`.
- The checkpoint is BF16. On GPUs without BF16 (e.g. T4), it is served in FP16 (`dtype: half`).
