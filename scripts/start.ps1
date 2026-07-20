$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Run scripts\setup-portable.ps1 first.'
}
Set-Location -LiteralPath $projectRoot
Write-Host 'Starting Vedock at http://127.0.0.1:5464'
Write-Host 'Press Ctrl+C to stop the server.'
& $python serve.py
