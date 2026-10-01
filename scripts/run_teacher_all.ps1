# Teacher run on Windows (PowerShell). Run from the repository folder:
#
#     powershell -ExecutionPolicy Bypass -File scripts\run_teacher_all.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\run_teacher_all.ps1 -Splits test      # one split only
#     powershell -ExecutionPolicy Bypass -File scripts\run_teacher_all.ps1 -Limit 20         # quick trial run
#
# It checks the GPU, picks the config that fits it, checks the input files and Hugging Face access,
# then runs each split and saves the full console output to experiments\teacher\<split>.console.log.
# Rerunning resumes where it stopped.

param(
    [string[]]$Splits = @("train", "dev", "test"),
    [string]$Config = "",
    [int]$Limit = 0
)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path "experiments\teacher" | Out-Null

function Stop-With($msg) {
    Write-Host ""
    Write-Host "STOP: $msg" -ForegroundColor Red
    exit 1
}

Write-Host "== 1. GPU" -ForegroundColor Cyan
$gpu = python -c "import torch; ok=torch.cuda.is_available(); print(ok, round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if ok else 0, torch.cuda.is_bf16_supported() if ok else False, torch.cuda.get_device_name(0) if ok else '-')"
if ($LASTEXITCODE -ne 0) { Stop-With "Python or torch is not installed here. Run: pip install -r requirements.txt" }
$parts = $gpu -split " ", 4
if ($parts[0] -ne "True") {
    Stop-With "PyTorch sees no CUDA GPU. If nvidia-smi shows one, reinstall torch with CUDA: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"
}
$memGB = [double]$parts[1]
Write-Host "GPU: $($parts[3]), $memGB GB, bfloat16: $($parts[2])"

if (-not $Config) {
    if ($memGB -lt 20 -or $parts[2] -ne "True") { $Config = "configs/teacher_full_small_gpu.yaml" }
    else { $Config = "configs/teacher_full.yaml" }
}
Write-Host "config: $Config"
if ($memGB -lt 20 -and $Config -notmatch "small_gpu") {
    Stop-With "$Config needs about 16 GB for the 7B teacher; this GPU has $memGB GB. Use configs/teacher_full_small_gpu.yaml"
}

Write-Host "== 2. Input files" -ForegroundColor Cyan
foreach ($s in $Splits) {
    $f = "data/processed/teacher_inputs_$s.jsonl"
    if (-not (Test-Path $f)) {
        Stop-With "$f not found (current folder: $(Get-Location)). Prepare it first: python scripts/prepare_teacher_data.py --dataset mlpaper/teacher-inputs --split $s --output $f"
    }
    $n = (Get-Content $f | Where-Object { $_.Trim() -ne "" }).Count
    if ($n -eq 0) { Stop-With "$f is empty" }
    Write-Host "$f : $n prompts"
}

Write-Host "== 3. Hugging Face access" -ForegroundColor Cyan
$models = python -c "import yaml; c=yaml.safe_load(open('$Config')); print(' '.join([c['same_tokenizer']['model']] + [h['model'] for h in c.get('heterogeneous') or []]))"
foreach ($m in ($models -split " ")) {
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('$m', 'config.json')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Stop-With "cannot download $m. Gated model: log in on huggingface.co, accept its licence, then run 'hf auth login' (or set HF_TOKEN) and try again."
    }
    Write-Host "OK: $m"
}

Write-Host "== 4. Run" -ForegroundColor Cyan
foreach ($s in $Splits) {
    $out = "experiments/teacher/$s"
    $log = "experiments/teacher/$s.console.log"
    $pyArgs = @("scripts/run_teacher.py", "--config", $Config, "--input", "data/processed/teacher_inputs_$s.jsonl", "--output-dir", $out)
    if ($Limit -gt 0) { $pyArgs += @("--limit", "$Limit") }
    Write-Host ""
    Write-Host ">>> split $s -> $out   (console copy: $log)" -ForegroundColor Green
    python @pyArgs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Stop-With "split $s failed. Send $log to the team." }
    Get-ChildItem $out | Select-Object Name, Length | Format-Table -AutoSize
}

Write-Host ""
Write-Host "DONE. Zip experiments\teacher (all split folders and the .console.log files) and send it privately." -ForegroundColor Green
