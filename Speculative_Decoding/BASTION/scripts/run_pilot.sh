#!/usr/bin/env bash
# Tiny end-to-end pipeline check on one model: 2 samples per dataset (mt-bench 1),
# short generations, all three modes plus one fixed budget. Resumable (rerun = continue).
# Usage: scripts/run_pilot.sh [model-key]   (default qwen3-4b)
#   env: PILOT_SAMPLES (2), PILOT_TOKENS (256), PILOT_DATASETS (all 8), RUN_NAME (pilot_<key>)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
KEY="${1:-qwen3-4b}"
resolve_model "$KEY"
RUN_NAME="${RUN_NAME:-pilot_$KEY}"
cd "$ROOT"
"$PY" run_benchmark.py \
  --model "$MODEL" --draft-model "$DRAFT" \
  --datasets ${PILOT_DATASETS:-$DATASETS_ALL} \
  --max-samples "${PILOT_SAMPLES:-2}" --cap mt-bench=1 \
  --modes ar dflash bastion --budgets adaptive 64 \
  --max-new-tokens "${PILOT_TOKENS:-256}" --temperature 0.0 --seed 0 --warmup 1 \
  --out-dir "$RUNS" --run-name "$RUN_NAME" 2>&1 | tee -a "$RUNS/$RUN_NAME.log"
