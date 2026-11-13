"""Create a ready-to-push Kaggle job folder from a template.

    python Quantization_and_Pruning/experiments/jacobian_noise/kaggle/make_job.py \
        --user <kaggle-username> --name jn-siglip224-s0 \
        --repo https://github.com/<github-user>/DA4022.git --branch jacobian-noise \
        --command "MODEL=siglip224 SEEDS=0 bash Quantization_and_Pruning/experiments/jacobian_noise/run_matrix.sh"
    kaggle kernels push -p kaggle_jobs/jn-siglip224-s0
    kaggle kernels status <kaggle-username>/jn-siglip224-s0
    kaggle kernels output <kaggle-username>/jn-siglip224-s0 -p results/jn-siglip224-s0

The folder (default ``kaggle_jobs/<name>``, listed in .gitignore) contains
``run_on_kaggle.py`` and ``kernel-metadata.json`` (private, GPU on, internet on).
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username")
    ap.add_argument("--name", required=True, help="kernel slug, e.g. jn-siglip224-s0")
    ap.add_argument("--repo", required=True, help="https URL of your fork")
    ap.add_argument("--branch", default="jacobian-noise")
    ap.add_argument("--command", required=True)
    ap.add_argument("--extra-pip", default="", help='e.g. "lm-eval peft"')
    ap.add_argument("--dest", default=None)
    args = ap.parse_args(argv)

    dest = args.dest or os.path.join("kaggle_jobs", args.name)
    os.makedirs(dest, exist_ok=True)
    src = open(os.path.join(HERE, "run_on_kaggle.template.py")).read()
    for k, v in {"__REPO_URL__": args.repo, "__BRANCH__": args.branch, "__COMMAND__": args.command,
                 "__EXTRA_PIP__": args.extra_pip}.items():
        src = src.replace(k, v)
    with open(os.path.join(dest, "run_on_kaggle.py"), "w") as f:
        f.write(src)
    meta = {
        "id": f"{args.user}/{args.name}",
        "title": args.name,
        "code_file": "run_on_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(dest, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {dest}/run_on_kaggle.py and kernel-metadata.json\nnext: kaggle kernels push -p {dest}")


if __name__ == "__main__":
    main()
