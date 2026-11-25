"""Build a self-contained Colab notebook from a prepared, versioned source ZIP."""

import argparse
import base64
import hashlib
import json
from pathlib import Path


def build(source_path, output):
    source = Path(source_path).read_bytes()
    cells = []

    def cell(name, text, code=False):
        value = {"cell_type": "code" if code else "markdown", "id": name,
                 "metadata": {}, "source": text.splitlines(keepends=True)}
        if code:
            compile(text, name, "exec")
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell("intro", """# AgentKV: reviewed scoring and pretrained integration check

Select **Runtime → Change runtime type → T4 GPU**, then **Run all**.
Source code is embedded; no ZIP upload is needed. All model execution is in Colab.

This is a new experiment, not another identical rerun. It tests the released
AgentKV preset's mean-GQA/candidate-only scoring and sink16/recent8 protection,
while retaining our documented streaming parser and plain tensor cache adapter.
The output cap stays at 512. Both compressed policies use the same revised settings.

Before 35 measured pilot episodes, the notebook runs CPU tests and a pretrained
numerical integration gate. An error stops the execution cell before the pilot.
The longer 210-episode evaluation is prepared in the source but intentionally
not launched until this new pilot is reviewed. Original pilot files are untouched.
""")
    cell("setup", f"""import base64, hashlib, io, os, zipfile
from pathlib import Path
import subprocess, sys

subprocess.run([sys.executable, '-c', "import torch; assert torch.cuda.is_available(), 'Select a GPU runtime first'; print(torch.cuda.get_device_name(0), torch.__version__)"], check=True)
source_bytes = base64.b64decode({base64.b64encode(source).decode()!r})
assert hashlib.sha256(source_bytes).hexdigest() == {hashlib.sha256(source).hexdigest()!r}
WORKDIR = Path('/content/agentkv-reviewed-work')
WORKDIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
    for name in archive.namelist():
        assert (WORKDIR / name).resolve().is_relative_to(WORKDIR.resolve())
    archive.extractall(WORKDIR)
os.chdir(WORKDIR)
print(WORKDIR)
""", True)
    cell("install", """subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'Attention/requirements-agentkv.txt'], check=True)
# All model/test work runs in fresh subprocesses; no notebook kernel restart is needed.
""", True)
    cell("execute", """import json, shutil

OUT = Path('artifacts/agentkv-reviewed-pilot')
VERIFY = Path('artifacts/reviewed-verification.json')
subprocess.run([sys.executable, '-m', 'pytest', 'Attention/tests', '-q'], check=True)
subprocess.run([sys.executable, '-u', '-m', 'Attention.experiments.agentkv.verify_model', '--out', str(VERIFY)], check=True)
verification = json.loads(VERIFY.read_text())
assert verification['passed'] and not verification['smoke']
command = [sys.executable, '-u', '-m', 'Attention.experiments.agentkv.run',
           '--config', 'Attention/configs/agentkv/reviewed_pilot.json', '--out', str(OUT)]
if (OUT / 'manifest.json').exists():
    command.append('--resume')
subprocess.run(command, check=True)
shutil.copyfile(VERIFY, OUT / 'verification.json')
print('Checks and reviewed pilot completed. No larger evaluation was started.')
""", True)
    cell("results", """for row in json.loads((OUT / 'summary.json').read_text()):
    print(row['method'], (row['policy'] or {}).get('budget'),
          'accuracy=', row.get('quality_value'), 'categories=', row.get('answer_categories'),
          'tokens/s=', row.get('throughput'), 'runtime failures=', row.get('failures'))
""", True)
    cell("download", """from google.colab import files

result_zip = shutil.make_archive('/content/agentkv-reviewed-results', 'zip', str(OUT))
print('Download:', result_zip)
files.download(result_zip)
""", True)
    cell("finish", """Save the downloaded file to `C:\\Yash\\DA4022\\DA4022\\artifacts\\agentkv-reviewed-results.zip`.
If any check fails, send the error before attempting the benchmark. Keep earlier
results ZIPs; this profile is a separate experiment. A successful integration
gate verifies specified numerical properties, not the paper's quality claims.
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Created {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/agentkv-reviewed-source.zip")
    parser.add_argument("--out", default="artifacts/agentkv_reviewed_pilot.ipynb")
    args = parser.parse_args()
    build(args.source, args.out)
