#!/usr/bin/env bash
# E1/E3: AR vs DFlash vs adaptive BASTION over the 8 short benchmarks, T=0, batch 1.
# Resumable: rerunning with the same RUN_NAME skips (dataset, sample, turn, mode) already in the JSONL.
# Usage: scripts/run_e1.sh <qwen3-4b|qwen3-8b|llama3.1-8b>
#   env: MAX_SAMPLES (64), CAP_AIME25 (30), CAP_MTBENCH (40), EXTRA_CAPS ("name=N ..."),
#        DATASETS (all 8), MAX_NEW_TOKENS (2048), WARMUP (1; upstream does no warm-up), RUN_NAME (e1_<key>),
#        PROFILE_MODE (warm: controller constants from a second, warmed-up profiling call;
#                      upstream: upstream's cold 3-trial profile cache)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
KEY="${1:?usage: run_e1.sh <model-key>}"
resolve_model "$KEY"
RUN_NAME="${RUN_NAME:-e1_$KEY}"
CAPS=(--cap "aime25=${CAP_AIME25:-30}" --cap "mt-bench=${CAP_MTBENCH:-40}")
for c in ${EXTRA_CAPS:-}; do CAPS+=(--cap "$c"); done
cd "$ROOT"
echo "[$(date -Is)] start $RUN_NAME" | tee -a "$RUNS/$RUN_NAME.log"
"$PY" run_benchmark.py \
  --model "$MODEL" --draft-model "$DRAFT" \
  --datasets ${DATASETS:-$DATASETS_ALL} \
  --max-samples "${MAX_SAMPLES:-64}" "${CAPS[@]}" \
  --modes ar dflash bastion --budgets adaptive --profile-mode "${PROFILE_MODE:-warm}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --temperature 0.0 --seed 0 --warmup "${WARMUP:-1}" \
  --out-dir "$RUNS" --run-name "$RUN_NAME" 2>&1 | tee -a "$RUNS/$RUN_NAME.log"
echo "[$(date -Is)] end $RUN_NAME" | tee -a "$RUNS/$RUN_NAME.log"
