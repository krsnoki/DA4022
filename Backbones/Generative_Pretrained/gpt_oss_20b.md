# Backbone model card: gpt-oss-20b

**Status: not used.** We planned to use this model with vLLM. vLLM release builds from 0.11.1 onward need a CUDA 12.9 or newer toolchain, and the NVIDIA driver on the machine we used (570.211) only accepts CUDA 12.8. Loading failed with `cudaErrorUnsupportedPtxVersion`. Without vLLM's 4-bit kernels the weights do not fit in 32 GB. We use OLMoE-1B-7B-0125-Instruct instead (see `olmoe_1b_7b_0125_instruct.md`).

- **Registry ID:** text.gpt-oss-20b.mxfp4
- **Modality:** text
- **Architecture:** Decoder-only Mixture-of-Experts transformer (Hugging Face class `GptOssForCausalLM`). 24 layers, hidden size 2880, 32 experts per MoE layer, 4 experts active per token. The router takes the top-k experts and then applies softmax over the k selected logits.
- **Pretraining regime:** self_supervised
- **Training objective/data:** Next-token prediction. Further training details unknown.
- **Weights URI and exact revision:** https://huggingface.co/openai/gpt-oss-20b. Revision: `6cee5e81ee83917806bbde320786a8fb61efebee` (the snapshot we downloaded).
- **License/usage terms:** Apache-2.0
- **Tokenizer/processor revision:** Same repository and revision (`6cee5e81ee83917806bbde320786a8fb61efebee`).
- **Parent backbone (if fine-tuned):** none
- **Input/output contract:** Text in, text out. Prompts must use the harmony chat format. The chat template applies it.
- **Parameter count / weight size:** About 21B total parameters and about 3.6B active per token (model card figures). Exact integer count not verified. The MoE weights are MXFP4-quantized. The other tensors are BF16. The model card states it runs in about 16 GB of memory.

## Loading

```bash
vllm serve openai/gpt-oss-20b --host 127.0.0.1 --port 8765 --gpu-memory-utilization 0.60 --max-model-len 8192
```

Versions: vLLM 0.32.0+cu129, torch 2.13.0+cu129.

## Preprocessing and outputs

Tokenization and the harmony chat format are applied by the tokenizer's chat template. Tokenizer revision: same as the weights (`6cee5e81ee83917806bbde320786a8fb61efebee`). Expected output shapes, and the output fields returned by the serving stack: unknown.

## Smoke test

Failed at model load: `torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain` (cudaErrorUnsupportedPtxVersion), raised while repacking MXFP4 expert weights for the Marlin MoE backend. No generation was produced.

## Caveats

- Our GPU is an NVIDIA RTX PRO 4500 Blackwell (sm_120). This GPU is not on vLLM's official supported list for this model.
- Reports indicate that the model runs on this GPU through the Marlin MoE backend. Output quality on that path is not yet confirmed by us.
- Further training details (data, compute, and filtering) are unknown.

## Registry entry

```yaml
id: text.gpt-oss-20b.mxfp4
name: gpt-oss-20b
modality: text                   # text | vision | audio | multimodal | other
architecture: GptOssForCausalLM  # decoder-only Mixture-of-Experts transformer
pretraining_regime: self_supervised
objective: next-token prediction; further training details unknown
weights_source: huggingface
weights_uri: https://huggingface.co/openai/gpt-oss-20b
revision: 6cee5e81ee83917806bbde320786a8fb61efebee
license: Apache-2.0
num_parameters: unknown          # model card: about 21B total, about 3.6B active per token; integer not verified
input_spec:
  format: text tokens, harmony chat format applied by chat template
  weight_precision: MXFP4 (MoE weights), BF16 (other tensors)
preprocessing: tokenizer in the same repository at revision 6cee5e81ee83917806bbde320786a8fb61efebee
parent_backbone: null
fine_tuning_dataset: null
notes: Status not used. Our GPU (NVIDIA RTX PRO 4500 Blackwell, sm_120) is not on vLLM's official supported list for this model. Output quality on the Marlin MoE path is not yet confirmed. Smoke test failed at model load (cudaErrorUnsupportedPtxVersion while repacking MXFP4 weights for the Marlin MoE backend).
