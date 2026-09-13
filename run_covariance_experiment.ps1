[CmdletBinding()]
param(
    [string]$Python = 'python',
    [int]$Seeds = 3,
    [int]$WsdEpochs = 600
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    & $Python code/tests/test_covariance.py
    if ($LASTEXITCODE -ne 0) { throw 'Covariance tests failed.' }
    & $Python code/experiments/train_wsd_anchor.py --epochs $WsdEpochs
    if ($LASTEXITCODE -ne 0) { throw 'WSD anchor training failed.' }
    & $Python code/experiments/covariance_anchor.py --seeds $Seeds
    if ($LASTEXITCODE -ne 0) { throw 'Five-score-call covariance experiment failed.' }
    & $Python code/experiments/covariance_anchor.py --nfe 8 --t-starts 123 `
        --seeds $Seeds --output results/covariance_nfe8.json
    if ($LASTEXITCODE -ne 0) { throw 'Nine-score-call covariance experiment failed.' }
    & $Python code/plots/covariance_anchor.py
    if ($LASTEXITCODE -ne 0) { throw 'Covariance plot generation failed.' }
    & $Python code/plots/anchor_overview.py
    if ($LASTEXITCODE -ne 0) { throw 'Anchor-overview plot generation failed.' }
} finally {
    Pop-Location
}
