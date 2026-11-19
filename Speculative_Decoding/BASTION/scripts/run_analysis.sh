#!/usr/bin/env bash
# Aggregate + lossless check + math scoring + latency model + plots into results/.
# CPU only; safe to rerun. Uses whatever runs/e1_*.jsonl and runs/sweep_*.jsonl exist.
#   env: STATUS (DA4022 status column, default "partial"), DEVICE_LABEL (for plot legends)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
LOG="$RUNS/analysis.log"
shopt -s nullglob
E1=(runs/e1_*.jsonl)  # relative paths: they are written into the outputs
SWEEPS=(runs/sweep_*.jsonl)
{
echo "[$(date -Is)] analysis: E1=${E1[*]:-none} SWEEPS=${SWEEPS[*]:-none}"
SUMMARIES=()
if [ ${#E1[@]} -gt 0 ]; then
  "$PY" analysis/score_math.py --runs "${E1[@]}" --out-dir "$RESULTS" --prefix e1_ || echo "[warn] score_math failed"
  QUAL=(); [ -f "$RESULTS/e1_math_scores.csv" ] && QUAL=(--quality "$RESULTS/e1_math_scores.csv")
  "$PY" analysis/aggregate.py --runs "${E1[@]}" --out-dir "$RESULTS" --prefix e1_ --status "${STATUS:-partial}" "${QUAL[@]}"
  "$PY" analysis/check_lossless.py --runs "${E1[@]}" --out-dir "$RESULTS" --prefix e1_
  "$PY" analysis/e1_extras.py --runs "${E1[@]}" --summary "$RESULTS/e1_summary.json" --out-dir "$RESULTS" --prefix e1_
  SUMMARIES+=("$RESULTS/e1_summary.json")
fi
SWEEP_SUMMARIES=()
for s in "${SWEEPS[@]}"; do
  name="$(basename "$s" .jsonl)"; key="${name#sweep_}"
  ref="runs/e1_$key.jsonl"; REF=(); [ -f "$ref" ] && REF=(--ar-jsonl "$ref")
  "$PY" analysis/aggregate.py --runs "$s" "${REF[@]}" --out-dir "$RESULTS" --prefix "${name}_" --status "${STATUS:-partial}"
  "$PY" analysis/check_lossless.py --runs "$s" "${REF[@]}" --out-dir "$RESULTS" --prefix "${name}_"
  [ -f "$ref" ] && "$PY" analysis/sweep_extras.py --sweep "$s" --e1 "$ref" --out-dir "$RESULTS" --prefix "${name}_"
  SWEEP_SUMMARIES+=("$RESULTS/${name}_summary.json")
done
"$PY" analysis/latency_model.py --out-dir "$RESULTS" --runs "${E1[@]}" "${SWEEPS[@]}"
"$PY" analysis/plots.py --out-dir "$RESULTS/figures" --latency-dir "$RESULTS" \
  ${SUMMARIES:+--summary "${SUMMARIES[@]}"} ${SWEEP_SUMMARIES:+--sweep-summary "${SWEEP_SUMMARIES[@]}"} \
  --device-label "${DEVICE_LABEL:-}"
echo "[$(date -Is)] analysis done -> $RESULTS"
} 2>&1 | tee -a "$LOG"
