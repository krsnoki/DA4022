# Stage A driver: trace generation on a Kaggle T4 machine (one process per GPU).
# Prepended by build_kernel.py with the embedded repo files.
import glob
import json
import subprocess
import time

T0 = time.time()
OUT = "/kaggle/working"
GEN = "Adaptive_Stopping/experiments/terminator/generate_traces.py"


def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("pip install -q vllm math-verify 2>&1 | tail -n 3")
sh('python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.device_count())"')

import torch  # noqa: E402

n_gpu = max(1, torch.cuda.device_count())
smoke = os.environ.get("SMOKE") == "1"
backend = os.environ.get("BACKEND", "vllm")
n_math500 = int(os.environ.get("N_MATH500", "16" if smoke else "500"))
n_train = int(os.environ.get("N_TRAIN", "8" if smoke else "2000"))
max_think = int(os.environ.get("MAX_THINK", "6000"))
deadline = float(os.environ.get("DEADLINE_S", str(11 * 3600)))
if smoke:
    n_gpu = 1
else:
    # quick check of the whole generation path before the long run
    rc = sh(f"python {GEN} --split math500 --n 4 --max-think 256 --max-answer 256 "
            f"--backend {backend} --gpu-mem 0.5 --out {OUT}/check.jsonl > {OUT}/log_check.txt 2>&1")
    ok = rc == 0 and os.path.exists(f"{OUT}/check.jsonl")
    sh(f"tail -n 20 {OUT}/log_check.txt")
    if not ok and backend == "vllm":
        print("vLLM check failed, falling back to HF generate", flush=True)
        backend = "hf"
    sh(f"rm -f {OUT}/check.jsonl")

procs = []
for g in range(n_gpu):
    left = deadline - (time.time() - T0)
    common = (f"--num-shards {n_gpu} --shard {g} --backend {backend} --max-think {max_think} "
              f"--deadline-s {left:.0f}")
    cmd = (f"CUDA_VISIBLE_DEVICES={g} python {GEN} --split math500 --n {n_math500} {common} "
           f"--out {OUT}/traces_math500.shard{g}.jsonl > {OUT}/log_gpu{g}.txt 2>&1; "
           f"CUDA_VISIBLE_DEVICES={g} python {GEN} --split train --n {n_train} {common} "
           f"--out {OUT}/traces_train.shard{g}.jsonl >> {OUT}/log_gpu{g}.txt 2>&1")
    print("+", cmd, flush=True)
    procs.append(subprocess.Popen(cmd, shell=True))
    time.sleep(60)

while any(p.poll() is None for p in procs):
    time.sleep(120)
    for g in range(n_gpu):
        sh(f"tail -n 2 {OUT}/log_gpu{g}.txt")

for g in range(n_gpu):
    sh(f"tail -n 40 {OUT}/log_gpu{g}.txt")

for split in ["math500", "train"]:
    rows = []
    for fn in sorted(glob.glob(f"{OUT}/traces_{split}.shard*.jsonl")):
        with open(fn) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        continue
    with open(f"{OUT}/traces_{split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    summary = {
        "split": split,
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "mean_thinking_tokens": sum(r["thinking_tokens"] for r in rows) / n,
        "cap_rate": sum(r["think_end"] == "cap" for r in rows) / n,
        "eos_in_thinking_rate": sum(r["think_end"] == "eos" for r in rows) / n,
        "answer_truncated_rate": sum(r["answer_truncated"] for r in rows) / n,
        "no_boxed_rate": sum(r["pred"] is None for r in rows) / n,
    }
    print(json.dumps(summary), flush=True)
    with open(f"{OUT}/summary_{split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    sh(f"rm {OUT}/traces_{split}.shard*.jsonl")

print(f"total time {time.time() - T0:.0f}s")
