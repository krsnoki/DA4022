#!/usr/bin/env bash
# One-command GPU run for the staircase-streaming reproduction (Linux + NVIDIA GPU).
#
#   bash Serving/staircase_streaming/remote/run_all.sh [CONFIG]
#
# Steps: check GPU -> venv + install (vLLM) -> env report -> CPU tests -> start vLLM
# servers -> live smoke tests -> 3-prompt sanity run -> full benchmark -> results tarball.
# The servers are stopped on exit (also on error / Ctrl-C).
#
# Environment knobs:
#   SKIP_VENV=1       install into the current Python instead of creating .venv-gpu
#   SKIP_INSTALL=1    reuse an existing environment (second run on the same machine)
#   SANITY_ONLY=1     stop after the sanity run (check the printed answers first)
#   LIMIT=N           at most N prompts per prompt set (quick end-to-end check)
#   HF_HOME           model cache (default: <repo>/.hf_cache; on a persistent volume, if available)
set -euo pipefail

CONFIG="${1:-Serving/staircase_streaming/configs/default.yaml}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

echo "=== [1/8] GPU check"
nvidia-smi || { echo "nvidia-smi failed: no NVIDIA GPU/driver visible"; exit 1; }

echo "=== [2/8] Python environment"
if [ "${SKIP_VENV:-0}" != "1" ]; then
  [ -d .venv-gpu ] || "$PY" -m venv .venv-gpu
  # shellcheck disable=SC1091
  source .venv-gpu/bin/activate
  PY=python
fi
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r Serving/staircase_streaming/requirements-gpu.txt
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export DA4022_DATA_CACHE="${DA4022_DATA_CACHE:-$REPO_ROOT/.cache/datasets}"
OUT="$("$PY" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['experiment']['output_dir'])" "$CONFIG")"
mkdir -p "$OUT"
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
echo "HF_HOME=$HF_HOME  output=$OUT"

echo "=== [3/8] Environment report"
"$PY" Serving/staircase_streaming/env_check.py
cp results/env_report.json "$OUT/env_report.json"

echo "=== [4/8] CPU tests (mock backend, fake server, loaders)"
"$PY" -m pytest Serving/staircase_streaming/tests Dataloaders/tests -q

echo "=== [5/8] Start vLLM servers (downloads models on first run)"
stop_servers() { "$PY" -m Serving.staircase_streaming.launch_vllm --config "$CONFIG" stop || true; }
trap stop_servers EXIT
"$PY" -m Serving.staircase_streaming.launch_vllm --config "$CONFIG" start
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | tee "$OUT/gpu_after_server_start.csv"

echo "=== [6/8] Live smoke tests"
STAIRCASE_LIVE_CONFIG="$CONFIG" "$PY" -m pytest Serving/staircase_streaming/tests -m slow -q

echo "=== [7/8] Sanity run (3 prompts, answers printed)"
"$PY" -m Serving.staircase_streaming.run_benchmark --config "$CONFIG" --sanity 2>&1 | tee "$OUT/sanity.log"
if [ "${SANITY_ONLY:-0}" = "1" ]; then
  echo "SANITY_ONLY=1 -> stopping here. Re-run without it for the full benchmark."
  exit 0
fi

echo "=== [8/8] Full benchmark (resumable: re-running continues where it stopped)"
LIMIT_ARG=()
[ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")
"$PY" -m Serving.staircase_streaming.run_benchmark --config "$CONFIG" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} 2>&1 | tee "$OUT/benchmark.log"

TARBALL="$(dirname "$OUT")/$(basename "$OUT")_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "=== DONE. Results: $OUT   Archive to bring back: $TARBALL"
