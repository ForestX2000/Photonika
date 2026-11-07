param([string]$RunName = 'training')
$ErrorActionPreference = 'Stop'
$RepoDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RunDirectory = Join-Path $RepoDirectory "data\$RunName"
$ProcessPath = Join-Path $RunDirectory 'process.json'
$StatusPath = Join-Path $RunDirectory 'status.json'
$LogPath = Join-Path $RunDirectory 'training.log'
$ErrorLogPath = Join-Path $RunDirectory 'training.stderr.log'

if (Test-Path -LiteralPath $ProcessPath -PathType Leaf) {
    $Recorded = Get-Content -LiteralPath $ProcessPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $RecordedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$Recorded.pid)"
    $IsTraining = $RecordedProcess -and $RecordedProcess.CommandLine -and
        $RecordedProcess.CommandLine -match '(?i)-m\s+app\.training\b' -and
        $RecordedProcess.CommandLine.Contains($RunDirectory)
    if ($IsTraining) {
        Write-Output "Running: PID $($Recorded.pid), started $($Recorded.started_at), limit $($Recorded.minutes) min."
    } else {
        Write-Output "Stopped: recorded training PID $($Recorded.pid) is no longer running."
    }
} else {
    Write-Output 'No launcher process record exists.'
}

if (Test-Path -LiteralPath $StatusPath -PathType Leaf) {
    $S = Get-Content -LiteralPath $StatusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Output ("status={0} step={1} batch={2} elapsed={3:N1} min best(PCC+SSIM)={4}" -f `
        $S.status, $S.step, $S.batch, ($S.elapsed_seconds / 60), $S.best_score)
    if ($S.validation) {
        Write-Output ("val PCC={0:N4} SSIM={1:N4} | letters {2:N3} | photos {3:N3} | mixed {4:N3}" -f `
            $S.validation.pcc, $S.validation.ssim, $S.validation.letters.pcc, $S.validation.photos.pcc, $S.validation.mixed.pcc)
    }
    if ($S.error) { Write-Output "error: $($S.error)" }
}
if (Test-Path -LiteralPath (Join-Path $RunDirectory 'STOP')) { Write-Output 'A graceful stop was requested.' }
if (Test-Path -LiteralPath $LogPath -PathType Leaf) {
    Write-Output 'Recent training log:'
    Get-Content -LiteralPath $LogPath -Tail 8 -Encoding UTF8
}
if ((Test-Path -LiteralPath $ErrorLogPath -PathType Leaf) -and (Get-Item -LiteralPath $ErrorLogPath).Length -gt 0) {
    Write-Output 'Recent stderr:'
    Get-Content -LiteralPath $ErrorLogPath -Tail 6 -Encoding UTF8
}
