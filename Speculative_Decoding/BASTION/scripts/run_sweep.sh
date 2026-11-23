#!/usr/bin/env bash
# E4: BASTION fixed verification budgets N in {32..1024} vs adaptive (BASTION mode only).
# Two invocations write into the same JSONL (resume keys keep them apart):
#   1) fixed budgets + adaptive with --profile-mode warm (the configuration used for E1),
#   2) adaptive with --profile-mode upstream (upstream's cold 3-trial profile cache),
#      skipped when SWEEP_UPSTREAM_PROFILE=0.
# Fixed-budget trees do not depend on the profile constants, so they run only once.
# Speedups are computed later against the AR records of the E1 run (runs/e1_<key>.jsonl),
# matched on (dataset, orig_idx, turn) with a prompt-hash check. The first SWEEP_SAMPLES
# of upstream's selection order are a subset of E1's selection for every dataset
# (same seeded permutation; aime25 is fully covered by E1's cap of 30).
# MT-Bench turn-2 prompts use E1's AR outputs as history (--history-jsonl) when available.
# Usage: scripts/run_sweep.sh <model-key>      (plan: qwen3-8b)
#   env: SWEEP_SAMPLES (16), BUDGETS ("32 64 128 256 512 1024"), DATASETS (all 8),
#        MAX_NEW_TOKENS (2048), WARMUP (1), RUN_NAME (sweep_<key>), E1_RUN (e1_<key>),
#        SWEEP_UPSTREAM_PROFILE (1)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
KEY="${1:?usage: run_sweep.sh <model-key>}"
resolve_model "$KEY"
RUN_NAME="${RUN_NAME:-sweep_$KEY}"
E1_JSONL="$RUNS/${E1_RUN:-e1_$KEY}.jsonl"
HIST=()
if [ -f "$E1_JSONL" ]; then HIST=(--history-jsonl "$E1_JSONL"); else
  echo "[warn] $E1_JSONL not found: turn-2 history will come from the first budget's output and no AR reference exists yet" >&2; fi
cd "$ROOT"

sweep() {  # sweep <profile-mode> <budgets...>
  local pm="$1"; shift
  echo "[$(date -Is)] start $RUN_NAME profile=$pm budgets=$*" | tee -a "$RUNS/$RUN_NAME.log"
  "$PY" run_benchmark.py \
    --model "$MODEL" --draft-model "$DRAFT" \
    --datasets ${DATASETS:-$DATASETS_ALL} \
    --max-samples "${SWEEP_SAMPLES:-16}" \
    --modes bastion --budgets "$@" --profile-mode "$pm" "${HIST[@]}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --temperature 0.0 --seed 0 --warmup "${WARMUP:-1}" \
    --out-dir "$RUNS" --run-name "$RUN_NAME" 2>&1 | tee -a "$RUNS/$RUN_NAME.log"
  echo "[$(date -Is)] end $RUN_NAME profile=$pm" | tee -a "$RUNS/$RUN_NAME.log"
}

sweep warm ${BUDGETS:-32 64 128 256 512 1024} adaptive
if [ "${SWEEP_UPSTREAM_PROFILE:-1}" != "0" ]; then
  sweep upstream adaptive
fi
