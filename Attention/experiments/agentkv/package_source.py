"""Create a small cloud-upload bundle without Git metadata, weights or caches."""

import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from .run import source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/agentkv-source.zip")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if not (root / "Attention/configs/agentkv/pilot.json").is_file():
        raise SystemExit("Run this command from the repository root.")
    selected = []
    for folder in ("Attention", "Backbones", "Dataloaders", "Performance_Metrics"):
        for path in sorted((root / folder).rglob("*")):
            relative = path.relative_to(root)
            if any(part.startswith(".") or part in ("__pycache__", "artifacts") for part in relative.parts):
                continue
            if path.is_file() and path.suffix in (".py", ".md", ".json", ".txt", ".ipynb"):
                selected.append(path)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("Source bundle already exists; choose a new --out path to preserve provenance.")
    out.parent.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], capture_output=True, text=True)
    snapshot = {"base_commit": git.stdout.strip() or "unknown", "source_sha256": source_hash(),
                "note": "Includes working-tree changes; source hash identifies implementation, not base commit alone."}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("SOURCE_SNAPSHOT.json", json.dumps(snapshot, indent=2))
    print(f"{out.resolve()} ({out.stat().st_size:,} bytes, {len(selected)} source files)")


if __name__ == "__main__":
    main()
