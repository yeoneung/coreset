[CmdletBinding()]
param(
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    & $Python code/theory_bounds.py --output figures/theory_bounds.pdf
    if ($LASTEXITCODE -ne 0) { throw 'Theory-bound plot generation failed.' }
    & $Python code/plots/anchor_overview.py
    if ($LASTEXITCODE -ne 0) { throw 'Anchor-overview plot generation failed.' }
} finally {
    Pop-Location
}
