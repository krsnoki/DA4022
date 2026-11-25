# Stages B-E driver on a Kaggle 2x T4 machine. Needs the traces (a dataset or
# the output of the trace kernel) and, when "train" is not in STAGES, the
# output of a training run mounted as input. Prepended by build_kernel.py.
# STAGES is a comma list:
#   train:   labels and probe training on GPU 0
#   eval:    offline evaluation with vLLM, one shard per GPU, then a merge;
#            answers are cached per chunk, and cache files found in the inputs
#            (from an interrupted eval kernel) are reused
#   latency: online runs at the same time on both GPUs, no stopping on GPU 1
#            and TERMINATOR on GPU 0, so both see the same CPU load
import glob
import subprocess
import time

T0 = time.time()
OUT = "/kaggle/working"
EXP = "Adaptive_Stopping/experiments/terminator"
N_LAT = int(os.environ.get("N_LATENCY", "50"))
STAGES = os.environ.get("STAGES", "train,eval,latency").split(",")


def sh(cmd):
    print("+", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"rc={rc} t={time.time() - T0:.0f}s", flush=True)
    return rc


def find_all(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    print(name, hits, flush=True)
    return hits


def find(name):
    hits = find_all(name)
    return hits[0] if hits else None


sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
if "eval" in STAGES:
    sh("pip install -q vllm math-verify 2>&1 | tail -n 3")
else:
    # HF-only stages: keep the image's torch, pin the transformers version the
    # CPU tests use
    sh("pip install -q transformers==4.57.0 math-verify 2>&1 | tail -n 3")
sh('python -c "import torch, transformers; print(torch.__version__, transformers.__version__, '
   'torch.cuda.device_count())"')

tr_train, tr_test = find("traces_train.jsonl"), find("traces_math500.jsonl")
lat_deadline = float(os.environ.get("LAT_DEADLINE_S", str(11 * 3600)))
eval_deadline = float(os.environ.get("EVAL_DEADLINE_S", str(11 * 3600)))
G0 = "CUDA_VISIBLE_DEVICES=0"
probe = f"{OUT}/probe/probe.pt"

if "train" in STAGES:
    sh(f"python {EXP}/build_labels.py --traces {tr_train} {tr_test} --out {OUT}/labels.jsonl "
       f"--summary {OUT}/labels_summary.json")
    sh(f"{G0} python {EXP}/train_probe.py --traces {tr_train} {tr_test} "
       f"--labels {OUT}/labels.jsonl --out-dir {OUT}/probe 2>&1 | tee {OUT}/log_train.txt "
       f"| grep -v '^epoch' ")
else:
    probe = find("probe.pt")

if "eval" in STAGES:
    labels = f"{OUT}/labels.jsonl" if "train" in STAGES else find("labels.jsonl")
    probs = (f"{OUT}/probe/probs_test.jsonl {OUT}/probe/probs_val.jsonl" if "train" in STAGES
             else f"{find('probs_test.jsonl')} {find('probs_val.jsonl')}")
    resume = " ".join(find_all("cache.shard*.jsonl"))
    left = eval_deadline - (time.time() - T0)
    common = (f"python {EXP}/evaluate.py --out-dir {OUT}/eval --num-shards 2 --traces {tr_test} "
              f"{tr_train} --labels {labels} --probs {probs}")
    procs = [
        subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={g} {common} --shard {g} --deadline-s {left:.0f} "
            f"--resume-cache {resume} > {OUT}/log_eval_shard{g}.txt 2>&1",
            shell=True)
        for g in (0, 1)
    ]
    for p in procs:
        p.wait()
    print("eval shard return codes", [p.returncode for p in procs], flush=True)
    sh(f"grep -v 'INFO\\|WARNING' {OUT}/log_eval_shard0.txt | tail -n 20; "
       f"grep -v 'INFO\\|WARNING' {OUT}/log_eval_shard1.txt | tail -n 20")
    sh(f"python {EXP}/evaluate.py --out-dir {OUT}/eval --num-shards 2 --merge "
       f"> {OUT}/log_eval_merge.txt 2>&1; cat {OUT}/log_eval_merge.txt")

if "latency" in STAGES:
    if "eval" in STAGES:
        # the vLLM install above may have changed transformers
        sh("pip install -q transformers==4.57.0 2>&1 | tail -n 3")
    left = lat_deadline - (time.time() - T0)
    lat = [
        subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={gpu} python {EXP}/latency.py --method {m} {extra} --n {N_LAT} "
            f"--deadline-s {left:.0f} --out {OUT}/latency_{m}.jsonl "
            f"> {OUT}/log_latency_{m}.txt 2>&1",
            shell=True)
        for gpu, m, extra in [(1, "no_stop", ""), (0, "terminator", f"--probe {probe}")]
    ]
    for p in lat:
        p.wait()
    sh(f"tail -n 25 {OUT}/log_latency_no_stop.txt {OUT}/log_latency_terminator.txt")
print(f"total time {time.time() - T0:.0f}s")
