#!/usr/bin/env bash
# Run the experiment matrix for one model: fine-tune with each noise mode and seed,
# then quantize + evaluate each checkpoint. Also evaluates the un-finetuned backbone once.
#
#   MODEL=siglip224 bash Quantization_and_Pruning/experiments/jacobian_noise/run_matrix.sh
#   MODEL=qwen05 MODES="none jacobian" SEEDS="0" WEIGHT_METHODS="rtn gptq" bash .../run_matrix.sh
#
# Env vars (defaults in brackets):
#   MODEL            siglip224 | siglip384 | qwen05 | smoke_lm | smoke_siglip  [siglip224]
#   MODES            noise modes to train                                  [none jacobian uniform_matched]
#   SEEDS            seeds                                                 [0 1 2]
#   WEIGHT_METHODS   rtn and/or gptq                                       [rtn]
#   OUT              output root                                           [outputs/jacobian_noise]
#   KEEP_CKPT        1 keeps checkpoints after evaluation (they are large) [0]
#   EVAL_PRETRAINED  1 also evaluates the original backbone                [1]
#   EXTRA_TRAIN / EXTRA_EVAL  extra "--set" overrides, e.g. EXTRA_TRAIN="train.max_steps=200"
set -euo pipefail
export USE_TF=${USE_TF:-0}   # transformers must not import TensorFlow (Kaggle images ship it)
cd "$(dirname "$0")/../../.."

MODEL=${MODEL:-siglip224}
MODES=${MODES:-"none jacobian uniform_matched"}
SEEDS=${SEEDS:-"0 1 2"}
WEIGHT_METHODS=${WEIGHT_METHODS:-rtn}
OUT=${OUT:-outputs/jacobian_noise}
KEEP_CKPT=${KEEP_CKPT:-0}
EVAL_PRETRAINED=${EVAL_PRETRAINED:-1}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}
PY=${PYTHON:-python}

CFG=Quantization_and_Pruning/configs/jacobian_noise
EXP=Quantization_and_Pruning/experiments/jacobian_noise
case "$MODEL" in
  siglip224) TRAIN=$EXP/train_siglip.py; TCFG=$CFG/siglip_base_224.yaml; ECFG=$CFG/eval_siglip_base_224.yaml ;;
  siglip384) TRAIN=$EXP/train_siglip.py; TCFG=$CFG/siglip_base_384.yaml; ECFG=$CFG/eval_siglip_base_384.yaml ;;
  qwen05)    TRAIN=$EXP/train_lm.py;     TCFG=$CFG/qwen2_5_0_5b.yaml;   ECFG=$CFG/eval_qwen2_5_0_5b.yaml ;;
  smoke_lm)     TRAIN=$EXP/train_lm.py;     TCFG=$CFG/smoke_lm.yaml;     ECFG=$CFG/smoke_eval_lm.yaml ;;      # CPU test
  smoke_siglip) TRAIN=$EXP/train_siglip.py; TCFG=$CFG/smoke_siglip.yaml; ECFG=$CFG/smoke_eval_siglip.yaml ;;  # CPU test
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac

if [[ "$EVAL_PRETRAINED" == "1" ]]; then
  for wm in $WEIGHT_METHODS; do
    $PY $EXP/eval_quant.py --config "$ECFG" --set experiment.output_dir="$OUT/eval" quant.weight_method="$wm" $EXTRA_EVAL
  done
fi

for seed in $SEEDS; do
  for mode in $MODES; do
    echo "=== $MODEL mode=$mode seed=$seed ==="
    run_dir=$($PY $TRAIN --config "$TCFG" --set experiment.output_dir="$OUT" experiment.seed="$seed" noise.mode="$mode" $EXTRA_TRAIN | tee /dev/stderr | sed -n 's/^\[train_[a-z]*\] done -> \([^ ]*\).*/\1/p')
    for wm in $WEIGHT_METHODS; do
      $PY $EXP/eval_quant.py --config "$ECFG" --set experiment.output_dir="$OUT/eval" model.path="$run_dir/checkpoint" quant.weight_method="$wm" $EXTRA_EVAL
    done
    if [[ "$KEEP_CKPT" != "1" ]]; then rm -rf "$run_dir/checkpoint"; fi
  done
done
$PY $EXP/summarize.py --results "$OUT/eval/results.csv" --out "$OUT/summary" || true
echo "done. results: $OUT/eval/results.csv  summary: $OUT/summary"
