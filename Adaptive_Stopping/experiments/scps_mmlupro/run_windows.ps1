# SCPS on MMLU-Pro, native Windows + NVIDIA GPU (Transformers backend).
# Run from the repository root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File Adaptive_Stopping\experiments\scps_mmlupro\run_windows.ps1 -Config scps_mmlupro_qwen35_4b_bnb4 -Python C:\path\to\python.exe
# -Python must be Python 3.10 to 3.12 (it only creates .venv; everything then runs inside .venv).
# Pick the torch wheel index for your driver with https://pytorch.org/get-started/locally/ (default below: CUDA 12.8).
# Re-running is safe: finished questions are kept (--resume). Use -SkipSetup to go straight to the run.
param(
    [string]$Config = "scps_mmlupro_qwen35_4b_bnb4",
    [string]$Python = "python",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [int]$Pilot = 8,
    [string]$Methods = "",            # e.g. "self_consistency,tree_prefix"; empty = all methods in the config
    [switch]$SkipSetup                # skip venv, pip, environment record, tests and pilot
)
$ErrorActionPreference = "Stop"
# PowerShell 5.1 does not stop on a failing native command, so check exit codes explicitly.
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed with exit code $LASTEXITCODE" } }
New-Item -ItemType Directory -Force results | Out-Null
$Py = ".\.venv\Scripts\python"
$CfgPath = "Adaptive_Stopping\configs\$Config.yaml"

if (-not $SkipSetup) {
    $ver = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ([version]$ver -lt [version]"3.10" -or [version]$ver -ge [version]"3.13") { throw "Python $ver found at '$Python'; need 3.10 to 3.12 (pass -Python)" }
    if (-not (Test-Path .venv)) { & $Python -m venv .venv; Check "venv" }
    & $Py -m pip install --upgrade pip; Check "pip upgrade"
    & $Py -m pip install torch --index-url $TorchIndex; Check "torch install"
    & $Py -m pip install "transformers>=5.3" accelerate bitsandbytes "datasets>=2.18" pyyaml pytest; Check "package install"

    # 1. environment record
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | Tee-Object results\env_nvidia_smi.txt
    & $Py -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)" | Tee-Object results\env_python.txt
    & $Py -m pip freeze | Out-File -Encoding utf8 results\pip_freeze.txt

    # 2. tests, then a CUDA check inside .venv
    & $Py -m pytest -q Adaptive_Stopping\tests | Tee-Object results\pytest.txt; Check "tests"
    & $Py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available in .venv: reinstall torch with the right -TorchIndex')"; Check "CUDA check"

    # 3. pilot on a few questions (catches loading and prompt problems fast)
    & $Py -m Adaptive_Stopping.experiments.scps_mmlupro.run --config $CfgPath --out-dir "results\${Config}_pilot" --limit $Pilot; Check "pilot run"
}

# 4. full run (256 questions). Each method writes per_question_<method>.jsonl as it goes.
$RunArgs = @("-m", "Adaptive_Stopping.experiments.scps_mmlupro.run", "--config", $CfgPath, "--out-dir", "results\$Config", "--resume")
if ($Methods) { $RunArgs += @("--methods", $Methods) }
& $Py @RunArgs; Check "full run"

Compress-Archive -Force -Path results\* -DestinationPath "results_$Config.zip"
Write-Host "Done. All methods finished; results are in results\$Config and results_$Config.zip."
