# Фаза 1 плана: короткие A/B-прогоны (по умолчанию 4 × 20 минут, последовательно на одной GPU).
#   .\scripts\ab-test.ps1
#   .\scripts\ab-test.ps1 -Minutes 15 -Only ab_imgphase_b32,ab_backprop_unroll4
# Сравнить: python -m ml.compare_runs "data/runs/ab_*"
param([int]$Minutes = 20, [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
      [ValidateSet('unet', 'pdr', 'physics-unrolled')][string]$Architecture = 'unet', [string[]]$Only = @())
$ErrorActionPreference = 'Stop'
$RepoDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$PythonPath = Join-Path $RepoDirectory '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$Runs = @(
    @{ Name = 'ab_backprop_b32';     Args = @('--input-mode', 'backprop', '--base', '32', '--unroll', '0') },
    @{ Name = 'ab_imgphase_b32';     Args = @('--input-mode', 'imgphase', '--base', '32', '--unroll', '0') },
    @{ Name = 'ab_backprop_unroll4'; Args = @('--input-mode', 'backprop', '--base', '32', '--unroll', '4') },
    @{ Name = 'ab_imgphase_unroll4'; Args = @('--input-mode', 'imgphase', '--base', '32', '--unroll', '4') }
)
if ($Only.Count -gt 0) { $Runs = @($Runs | Where-Object { $Only -contains $_.Name }) }
Push-Location $RepoDirectory
try {
    foreach ($Run in $Runs) {
        $Out = Join-Path $RepoDirectory "data\runs\$($Run.Name)"
        Write-Output "=== $($Run.Name): $Minutes min ==="
        & $PythonPath -u -m ml.training --device $Device --architecture $Architecture --minutes $Minutes @($Run.Args) `
            --output $Out --weights (Join-Path $Out 'weights.pt') --eval-every 60
        if ($LASTEXITCODE -ne 0) { throw "Run $($Run.Name) failed" }
    }
    & $PythonPath -m ml.compare_runs (Join-Path $RepoDirectory 'data\runs\ab_*')
} finally {
    Pop-Location
}
