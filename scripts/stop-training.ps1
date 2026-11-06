param([string]$RunName = 'training')
$ErrorActionPreference = 'Stop'
$RepoDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RunDirectory = Join-Path $RepoDirectory "data\$RunName"
if (-not (Test-Path -LiteralPath $RunDirectory -PathType Container)) {
    throw "No data\$RunName directory exists; training has not been launched."
}
Set-Content -LiteralPath (Join-Path $RunDirectory 'STOP') -Value (Get-Date).ToString('o') -Encoding UTF8
Write-Output 'Stop requested. Training will validate, save its checkpoint, write the final report and exit.'
