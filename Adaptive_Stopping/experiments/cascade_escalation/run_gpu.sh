#!/usr/bin/env bash
# One-command GPU run for the cascade-escalation reproduction (Bouchard 2026).
#
#   git clone -b cascade-escalation https://github.com/Jigsaw442/DA4022 && cd DA4022
#   bash Adaptive_Stopping/experiments/cascade_escalation/run_gpu.sh
#
# Produces results/cascade_results.tar.gz (send this file back). Safe to re-run:
# finished (model, dataset) outputs are skipped. Env overrides:
#   BACKEND=auto|vllm|hf   CONFIG=...   OUT=results/cascade   SKIP_VLLM_INSTALL=1   SKIP_LIVE=1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-$HERE/configs/qwen25_phi35_pool.yaml}"
OUT="${OUT:-results/cascade}"
BACKEND="${BACKEND:-auto}"
mkdir -p "$OUT"
LOG="$OUT/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== cascade run started $(date -u +%FT%TZ) on $(hostname) ==="
nvidia-smi || echo "WARNING: nvidia-smi not found"

# Keep model downloads on the persistent volume when there is one (RunPod: /workspace).
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/hf_cache; fi
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY="${PYTHON:-python3}"

echo "=== [1/6] installing dependencies ==="
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r "$HERE/requirements-gpu.txt"
if [ "$BACKEND" != "hf" ] && [ -z "${SKIP_VLLM_INSTALL:-}" ]; then
  if ! $PY -c "import vllm" 2>/dev/null; then
    $PY -m pip install -q vllm || { echo "vLLM install failed -> using transformers backend"; BACKEND=hf; }
  fi
fi
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

MODELS=$($PY -c "import yaml,sys; print(' '.join(m['id'] for m in yaml.safe_load(open('$CONFIG'))['models']))")
SMALLEST=$(echo "$MODELS" | awk '{print $1}')

echo "=== [2/6] CPU unit tests ==="
$PY -m pytest -q Adaptive_Stopping/tests || { echo "unit tests failed"; exit 1; }

echo "=== [3/6] GPU smoke test (8 queries/dataset, $SMALLEST) ==="
if [ "$BACKEND" != "hf" ]; then
  if ! $PY "$HERE/collect_generations.py" --config "$CONFIG" --model "$SMALLEST" --out "$OUT/smoke" \
        --backend vllm --num-examples 8 --tag _vllm; then
    echo "vLLM smoke failed -> using transformers backend"; BACKEND=hf
  else
    BACKEND=vllm
  fi
fi
# Always run the HF path too: it is used by the live run, and lets us compare backends.
$PY "$HERE/collect_generations.py" --config "$CONFIG" --model "$SMALLEST" --out "$OUT/smoke" \
    --backend hf --num-examples 8 --tag _hf
echo "backend for full run: $BACKEND"

echo "=== [4/6] full generation run (one process per model) ==="
for M in $MODELS; do
  $PY "$HERE/collect_generations.py" --config "$CONFIG" --model "$M" --out "$OUT" --backend "$BACKEND" \
    || { echo "WARNING: $M failed with $BACKEND, retrying with transformers"; \
         $PY "$HERE/collect_generations.py" --config "$CONFIG" --model "$M" --out "$OUT" --backend hf; }
done

echo "=== [5/6] live batch-size-1 cascade run (latency / peak memory) ==="
if [ -z "${SKIP_LIVE:-}" ]; then
  $PY "$HERE/live_cascade.py" --config "$CONFIG" --results "$OUT" || echo "WARNING: live run failed (generations are still saved)"
fi

echo "=== [6/6] packaging results ==="
cp "$CONFIG" "$OUT/config_used.yaml"
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
tar -czf results/cascade_results.tar.gz -C results cascade
ls -lh results/cascade_results.tar.gz
echo "=== done $(date -u +%FT%TZ). Send back: $ROOT/results/cascade_results.tar.gz ==="
