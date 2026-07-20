param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv'

if (-not (Test-Path -LiteralPath (Join-Path $venvRoot 'Scripts\python.exe'))) {
    & $Python -m venv $venvRoot
}

$runtimePython = Join-Path $venvRoot 'Scripts\python.exe'
& $runtimePython -m pip install --upgrade pip
& $runtimePython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
& $runtimePython -m pip install -e $projectRoot

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) {
    Copy-Item -LiteralPath (Join-Path $projectRoot '.env.example') -Destination (Join-Path $projectRoot '.env')
}

foreach ($relative in @('storage\datasets\raw','storage\datasets\processed','storage\datasets\temporary','storage\models','storage\jobs','storage\exports','storage\temporary','instance')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot $relative) | Out-Null
}

Write-Host 'Vedock local node is ready.'
Write-Host "Start it with: $runtimePython serve.py"
