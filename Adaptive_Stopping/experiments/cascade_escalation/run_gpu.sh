#!/usr/bin/env bash
# One-command GPU run for the cascade-escalation reproduction (Bouchard 2026).
#
#   git clone -b cascade-escalation https://github.com/Jigsaw442/DA4022 && cd DA4022
#   bash Adaptive_Stopping/experiments/cascade_escalation/run_gpu.sh
#
# Works on RunPod and vast.ai (PyTorch image; run inside tmux or with nohup).
# Produces results/cascade_results.tar.gz (send this file back). Safe to re-run after a
# crash: finished (model, dataset) outputs are skipped, unfinished ones continue from their
# last 500-query checkpoint, and finished live measurements are reused. Env overrides:
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
if [ -z "${HF_HOME:-}" ]; then
  if [ -d /workspace ]; then export HF_HOME=/workspace/hf_cache; else export HF_HOME="$ROOT/.hf_cache"; fi
fi
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Use the python that already has CUDA torch (vast.ai images: /venv/main -> `python`; RunPod: python3).
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif python -c "import torch" 2>/dev/null; then PY=python
else PY=python3; fi
echo "python: $($PY --version) at $(command -v $PY); HF_HOME=$HF_HOME"
mkdir -p "$HF_HOME"
echo "free disk at HF_HOME: $(df -h "$HF_HOME" | tail -1)  (need ~60 GB free)"

# Fail fast on a machine without a working CUDA torch (before spending time on installs).
if [ -z "${ALLOW_CPU:-}" ]; then $PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"; fi

echo "=== [1/6] installing dependencies ==="
$PY -m pip install -q --upgrade pip || echo "pip upgrade skipped"
$PY -m pip install -q -r "$HERE/requirements-gpu.txt"
# vLLM goes in its own clean venv so it cannot replace the image's torch, which the
# transformers/live path needs. (On vast.ai `python` is itself a venv, so a nested
# --system-site-packages venv would not see its packages: install everything explicitly.)
# On any failure we fall back to the transformers backend.
PYV="$PY"
if [ "$BACKEND" != "hf" ] && [ -z "${SKIP_VLLM_INSTALL:-}" ]; then
  if ! $PY -c "import vllm" 2>/dev/null; then
    VENV="${VLLM_VENV:-$ROOT/.vllm_venv}"
    if [ ! -x "$VENV/bin/python" ]; then
      $PY -m venv "$VENV" 2>/dev/null || { $PY -m pip install -q virtualenv && $PY -m virtualenv -q "$VENV"; } || true
    fi
    echo "installing vLLM into $VENV (a few minutes)"
    if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip install -q --upgrade pip \
        && "$VENV/bin/python" -m pip install -q vllm -r "$HERE/requirements-gpu.txt" \
        && "$VENV/bin/python" -c "import vllm, torch; assert torch.cuda.is_available() or '${ALLOW_CPU:-}'; print('vllm', vllm.__version__, 'torch', torch.__version__)"; then
      PYV="$VENV/bin/python"
    else echo "vLLM install failed -> using transformers backend"; BACKEND=hf; fi
  fi
fi

MODELS=$($PY -c "import yaml,sys; print(' '.join(m['id'] for m in yaml.safe_load(open('$CONFIG'))['models']))")
SMALLEST=$(echo "$MODELS" | awk '{print $1}')

echo "=== [2/6] CPU unit tests ==="
$PY -m pytest -q Adaptive_Stopping/tests || { echo "unit tests failed"; exit 1; }

echo "=== [3/6] GPU smoke test (8 queries/dataset, $SMALLEST) ==="
if [ "$BACKEND" != "hf" ]; then
  if ! $PYV "$HERE/collect_generations.py" --config "$CONFIG" --model "$SMALLEST" --out "$OUT/smoke" \
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
  if [ "$BACKEND" = "vllm" ]; then RUNPY="$PYV"; else RUNPY="$PY"; fi
  $RUNPY "$HERE/collect_generations.py" --config "$CONFIG" --model "$M" --out "$OUT" --backend "$BACKEND" \
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
