#!/usr/bin/env bash
# Usage: smoke.sh <model> <draft>   (refuses to run if any other compute process is on the GPU)
set -u
ROOT=/home/prateek/Documents/gpu-test/DA4022/Speculative_Decoding/BASTION
MODEL=$1; DRAFT=$2; TAG=$(basename $MODEL)
LOG=$ROOT/runs/setup_smoke_${TAG}.log; MEM=$ROOT/runs/setup_smoke_${TAG}.gpumem.csv
mkdir -p $ROOT/runs
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$others" -ne 0 ]; then echo "GPU BUSY ($others compute procs); refusing"; exit 3; fi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 500 > $MEM &
SMI=$!
cd $ROOT/upstream
{ echo "# cmd: python tools/bastion_local.py --model $MODEL --draft-model $DRAFT --dataset gsm8k --max-samples 2 --max-new-tokens 256 --temperature 0.0"; date; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv; } > $LOG
start=$(date +%s.%N)
HF_HUB_OFFLINE=1 BASTION_LOG_LEVEL=INFO /usr/bin/time -v $ROOT/.venv/bin/python $ROOT/tools/bastion_local.py \
  --model $MODEL --draft-model $DRAFT --dataset gsm8k --max-samples 2 --max-new-tokens 256 --temperature 0.0 >> $LOG 2>&1
rc=$?
end=$(date +%s.%N)
kill $SMI
peak=$(tail -n +2 $MEM | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}')
echo "# exit=$rc wall_s=$(echo "$end - $start" | bc) peak_gpu_mem_used_MiB=$peak" | tee -a $LOG
