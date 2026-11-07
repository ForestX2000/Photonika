# Фоновое обучение на локальной NVIDIA GPU (окно терминала можно закрыть).
#   .\scripts\train-local.ps1 -Architecture physics-unrolled -Base 16 -StageBase 8 -Minutes 480
#   .\scripts\train-local.ps1 -Architecture unet -Unroll 0 -Minutes 240 -InitWeights data\weights\doe_multiplane.pt -Lr 3e-5 -WSsim 2
#   .\scripts\train-local.ps1 -Resume -Minutes 120                  # продолжить после stop-training.ps1
param(
    [ValidateRange(1, 1440)]
    [int]$Minutes = 120,
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [int]$Base = 32,
    [ValidateSet('physics-unrolled', 'unet', 'pdr')]
    [string]$Architecture = 'physics-unrolled',
    [ValidateRange(0, 32)]
    [int]$Unroll = 4,
    [int]$StageBase = 16,
    [ValidateSet('imgphase', 'backprop', 'raw')]
    [string]$InputMode = 'imgphase',
    [string]$Batch = 'auto',
    [double]$Lr = 1e-4,
    [double]$WSsim = 1.0,
    [double]$Worst = 0.5,
    [int]$PoolSize = 20000,
    [string]$InitWeights = '',
    [string]$RunName = 'training',
    [string]$WeightsName = 'doe_multiplane.pt',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$RepoDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$PythonPath = Join-Path $RepoDirectory '.venv\Scripts\python.exe'
$RunDirectory = Join-Path $RepoDirectory "data\$RunName"
$WeightsPath = Join-Path $RepoDirectory "data\weights\$WeightsName"
$ProcessPath = Join-Path $RunDirectory 'process.json'
$StopPath = Join-Path $RunDirectory 'STOP'

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Create .venv and install the dependencies first; see README.md.'
}
New-Item -ItemType Directory -Path $RunDirectory -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $WeightsPath) -Force | Out-Null

# Проверяем командную строку, а не только PID: Windows переиспользует PID.
$ExistingRuns = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine -match '(?i)-m\s+ml\.training\b' -and $_.CommandLine.Contains($RunDirectory) })
if ($ExistingRuns.Count -gt 0) {
    throw "Training is already running for this run directory (PID $($ExistingRuns[0].ProcessId)). Use scripts/training-status.ps1."
}
if ($Resume -and -not (Test-Path -LiteralPath (Join-Path $RunDirectory 'last.pt') -PathType Leaf)) {
    throw "No $RunDirectory\last.pt checkpoint exists to resume."
}
if (Test-Path -LiteralPath $StopPath) { Remove-Item -LiteralPath $StopPath }

$ArchiveStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
foreach ($LogName in @('training.log', 'training.stderr.log')) {
    $LogPath = Join-Path $RunDirectory $LogName
    if (Test-Path -LiteralPath $LogPath -PathType Leaf) {
        Copy-Item -LiteralPath $LogPath -Destination (Join-Path $RunDirectory "$LogName.$ArchiveStamp")
    }
}

$Inv = [System.Globalization.CultureInfo]::InvariantCulture
$TrainingArguments = @(
    '-u', '-m', 'ml.training',
    '--device', $Device,
    '--minutes', $Minutes.ToString($Inv),
    '--base', $Base.ToString($Inv), '--input-mode', $InputMode, '--batch', $Batch,
    '--architecture', $Architecture, '--unroll', $Unroll.ToString($Inv), '--stage-base', $StageBase.ToString($Inv),
    '--lr', $Lr.ToString($Inv), '--w-ssim', $WSsim.ToString($Inv), '--worst', $Worst.ToString($Inv),
    '--pool-size', $PoolSize.ToString($Inv),
    '--output', ('"{0}"' -f $RunDirectory),
    '--weights', ('"{0}"' -f $WeightsPath)
)
if ($InitWeights) { $TrainingArguments += @('--init-weights', ('"{0}"' -f (Resolve-Path $InitWeights))) }
if ($Resume) { $TrainingArguments += '--resume' }

$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$TrainingProcess = Start-Process -FilePath $PythonPath -ArgumentList $TrainingArguments `
    -WorkingDirectory $RepoDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $RunDirectory 'training.log') `
    -RedirectStandardError (Join-Path $RunDirectory 'training.stderr.log')

[ordered]@{
    pid = $TrainingProcess.Id; started_at = (Get-Date).ToString('o'); minutes = $Minutes; device = $Device
    resume = [bool]$Resume; repository = $RepoDirectory; output = $RunDirectory; weights = $WeightsPath
} | ConvertTo-Json | Set-Content -LiteralPath $ProcessPath -Encoding UTF8

Write-Output "Training started: PID $($TrainingProcess.Id), device $Device, limit $Minutes minutes, run '$RunName'."
Write-Output "Status: .\scripts\training-status.ps1 -RunName $RunName"
Write-Output "Save and stop: .\scripts\stop-training.ps1 -RunName $RunName"
