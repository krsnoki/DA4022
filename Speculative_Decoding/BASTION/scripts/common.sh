# Sourced by the run_*.sh scripts. Resolves paths and model keys.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUNS="$ROOT/runs"
RESULTS="$ROOT/results"
mkdir -p "$RUNS" "$RESULTS"
[ -x "$PY" ] || { echo "missing venv python: $PY" >&2; exit 1; }
export PYTHONUNBUFFERED=1
DATASETS_ALL="gsm8k math500 aime25 humaneval mbpp lcb mt-bench alpaca"

resolve_model() {  # sets MODEL and DRAFT from a model key
  case "$1" in
    qwen3-4b)    MODEL="Qwen/Qwen3-4B"; DRAFT="z-lab/Qwen3-4B-DFlash-b16" ;;
    qwen3-8b)    MODEL="Qwen/Qwen3-8B"; DRAFT="z-lab/Qwen3-8B-DFlash-b16" ;;
    llama3.1-8b) MODEL="meta-llama/Llama-3.1-8B-Instruct"; DRAFT="z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat" ;;
    *) echo "unknown model key '$1' (qwen3-4b | qwen3-8b | llama3.1-8b)" >&2; exit 2 ;;
  esac
}
