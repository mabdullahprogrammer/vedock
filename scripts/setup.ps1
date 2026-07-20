param(
    [string]$LegacyEnvironment = 'D:\LLM\cuda',
    [string]$StoryMakerRoot = 'D:\LLM\new-llm\LLM-2025\StoryMaker'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv'
$legacyPython = Join-Path $LegacyEnvironment 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $legacyPython)) {
    throw "Legacy Python was not found at $legacyPython"
}
if (-not (Test-Path -LiteralPath $StoryMakerRoot)) {
    throw "StoryMaker was not found at $StoryMakerRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $venvRoot 'Scripts\python.exe'))) {
    & $legacyPython -m venv $venvRoot
}

$sitePackages = Join-Path $venvRoot 'Lib\site-packages'
$legacySitePackages = Join-Path $LegacyEnvironment 'Lib\site-packages'
Set-Content -LiteralPath (Join-Path $sitePackages 'vedock_legacy_runtime.pth') -Value $legacySitePackages -Encoding ascii

$siteCustom = @"
from __future__ import annotations
import os
from pathlib import Path
LEGACY_ENV = Path(r"$LegacyEnvironment")
DLL_DIRECTORIES = [LEGACY_ENV / "Library" / "bin", LEGACY_ENV / "Lib" / "site-packages" / "torch" / "lib"]
_DLL_HANDLES = []
for directory in DLL_DIRECTORIES:
    if directory.is_dir():
        os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
"@
Set-Content -LiteralPath (Join-Path $sitePackages 'sitecustomize.py') -Value $siteCustom -Encoding utf8

$python = Join-Path $venvRoot 'Scripts\python.exe'
& $python -m pip install --disable-pip-version-check -r (Join-Path $projectRoot 'requirements.txt')
& $python -m pip install --disable-pip-version-check -e $projectRoot

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) {
    Copy-Item -LiteralPath (Join-Path $projectRoot '.env.example') -Destination (Join-Path $projectRoot '.env')
}

foreach ($relative in @('storage\datasets\raw','storage\datasets\processed','storage\datasets\temporary','storage\models','storage\jobs','storage\exports','storage\temporary','instance')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot $relative) | Out-Null
}

& $python -c "import torch, flask, sqlalchemy, accelerate, peft; print('Vedock environment ready'); print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
Write-Host "Run: $python run.py"
