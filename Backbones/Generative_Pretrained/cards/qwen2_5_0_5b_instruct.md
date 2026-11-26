# Backbone model card: Qwen/Qwen2.5-0.5B-Instruct

- **Registry ID:** `text.qwen2_5_0_5b_instruct`
- **Modality:** text
- **Architecture:** Qwen2ForCausalLM (model_type qwen2)
- **Pretraining regime:** mixed (self-supervised next-token pretraining, then supervised/preference post-training)
- **Training objective/data:** 'Pretraining & Post-training' (model card; no further detail on the card)
- **Weights URI and exact revision:** https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct @ `7ae557604adf67be50417f59c2c2f167def9a775`
- **License/usage terms:** Apache-2.0 (HF metadata), ungated
- **Tokenizer/processor revision:** bundled tokenizer and chat template at the same revision
- **Parent backbone (if fine-tuned):** unknown (not recorded in the fetched metadata)
- **Input/output contract:** chat messages in, generated text tokens out (OpenAI-compatible chat API through vLLM); context: 32,768 tokens context, 8,192 tokens generation (model card)
- **Parameter count / weight size:** 494,032,768 parameters (safetensors metadata); 988,097,824 bytes `model.safetensors` (BF16)

## Loading

Used in `Serving/staircase_streaming` as **proposer 3**, served by vLLM (command built by `launch_vllm.py`):

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --served-model-name <id> --port <port> --dtype auto
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
