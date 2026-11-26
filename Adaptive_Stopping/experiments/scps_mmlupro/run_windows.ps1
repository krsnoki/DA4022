# SCPS on MMLU-Pro, native Windows + NVIDIA GPU (Transformers backend).
# Run from the repository root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File Adaptive_Stopping\experiments\scps_mmlupro\run_windows.ps1 -Config scps_mmlupro_qwen35_9b_bnb4
# Pick the torch wheel index for your driver with https://pytorch.org/get-started/locally/ (default below: CUDA 12.8).
param(
    [string]$Config = "scps_mmlupro_qwen35_9b_bnb4",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [int]$Pilot = 8
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force results | Out-Null

if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url $TorchIndex
.\.venv\Scripts\python -m pip install "transformers>=5.3" accelerate bitsandbytes "datasets>=2.18" pyyaml pytest

# 1. environment record
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | Tee-Object results\env_nvidia_smi.txt
.\.venv\Scripts\python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)" | Tee-Object results\env_python.txt
.\.venv\Scripts\python -m pip freeze | Out-File -Encoding utf8 results\pip_freeze.txt

# 2. tests
.\.venv\Scripts\python -m pytest -q Adaptive_Stopping\tests | Tee-Object results\pytest.txt

# 3. pilot on a few questions (catches loading and prompt problems fast)
.\.venv\Scripts\python -m Adaptive_Stopping.experiments.scps_mmlupro.run --config "Adaptive_Stopping\configs\$Config.yaml" --out-dir "results\${Config}_pilot" --limit $Pilot

# 4. full run (256 questions, 4 methods)
.\.venv\Scripts\python -m Adaptive_Stopping.experiments.scps_mmlupro.run --config "Adaptive_Stopping\configs\$Config.yaml" --out-dir "results\$Config"

Compress-Archive -Force -Path results\* -DestinationPath "results_$Config.zip"
Write-Host "Done. Send results_$Config.zip back."
