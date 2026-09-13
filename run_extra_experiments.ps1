[CmdletBinding()]
param(
    [string]$Python = 'python',
    [int]$Seeds = 3
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    & $Python code/experiments/calibration.py --seeds $Seeds
    if ($LASTEXITCODE -ne 0) { throw 'Calibration experiment failed.' }
    & $Python code/experiments/rank_mi.py
    if ($LASTEXITCODE -ne 0) { throw 'Effective-rank experiment failed.' }
    & $Python code/plots/calibration.py
    if ($LASTEXITCODE -ne 0) { throw 'Calibration plot generation failed.' }
    & $Python code/plots/rank_mi.py
    if ($LASTEXITCODE -ne 0) { throw 'Effective-rank plot generation failed.' }
} finally {
    Pop-Location
}
