from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from flask import current_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_cli_archive() -> BytesIO:
    package_root = PROJECT_ROOT / "vedock_cli"
    pyproject = """[build-system]
requires = [\"setuptools>=68\", \"wheel\"]
build-backend = \"setuptools.build_meta\"

[project]
name = \"vedock-cli\"
version = \"0.1.0\"
description = \"Local API client for Vedock\"
requires-python = \">=3.11\"
dependencies = [\"click>=8,<9\", \"requests>=2.31,<3\"]

[project.scripts]
vedock = \"vedock_cli.main:cli\"

[tool.setuptools.packages.find]
include = [\"vedock_cli*\"]
"""
    api_url = f"http://127.0.0.1:{current_app.config['APP_PORT']}/api/v1"
    readme = f"""# {current_app.config['APP_NAME']} CLI

Install with `python -m pip install .`, start the local Vedock node, then run:

```text
vedock login
vedock doctor
vedock models list
vedock chat MODEL
```

Default local API: `{api_url}`
"""
    installer = """$ErrorActionPreference = 'Stop'
$archiveRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
python -m pip install $archiveRoot
Write-Host 'Vedock CLI installed. Run: vedock login'
"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("vedock-cli/pyproject.toml", pyproject)
        archive.writestr("vedock-cli/README.md", readme)
        archive.writestr("vedock-cli/install.ps1", installer)
        for path in sorted(package_root.glob("*.py")):
            archive.writestr(f"vedock-cli/vedock_cli/{path.name}", path.read_bytes())
    output.seek(0)
    return output


def build_node_archive() -> BytesIO:
    output = BytesIO()
    included_roots = ["vedock", "vedock_cli", "scripts", "demo", "docs", "tests"]
    root_files = [
        "run.py", "serve.py", "worker.py", "pyproject.toml", "requirements.txt",
        "requirements-client.txt", "requirements-local-core.txt", "requirements-core.txt", "requirements-text.txt", "requirements-fast-ml.txt", "requirements-dev.txt",
        ".env.example", "start-vedock.cmd", "README.md",
    ]
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for filename in root_files:
            path = PROJECT_ROOT / filename
            if path.is_file():
                archive.write(path, f"vedock-node/{filename}")
        for root_name in included_roots:
            root = PROJECT_ROOT / root_name
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                    continue
                archive.write(path, f"vedock-node/{path.relative_to(PROJECT_ROOT).as_posix()}")
        archive.writestr(
            "vedock-node/INSTALL.txt",
            "Install Python 3.11 or newer, open PowerShell in this folder, run scripts\\setup-portable.ps1, then start-vedock.cmd. All compute and storage stay on this computer by default.\n",
        )
    output.seek(0)
    return output


def build_client_archive() -> BytesIO:
    """Small connected-client source payload; heavy ML wheels are on-demand."""
    output = BytesIO()
    root_files = [
        "pyproject.toml",
        "requirements-client.txt",
        "requirements-local-core.txt",
        "requirements-text.txt",
        "requirements-fast-ml.txt",
    ]
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for filename in root_files:
            path = PROJECT_ROOT / filename
            archive.write(path, f"vedock-client/{filename}")
        for root_name in ["vedock", "vedock_cli"]:
            root = PROJECT_ROOT / root_name
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                    continue
                archive.write(path, f"vedock-client/{path.relative_to(PROJECT_ROOT).as_posix()}")
        archive.writestr(
            "vedock-client/CLIENT.txt",
            "Connected to https://vedock.ecorims.com. The client claims hosted training tasks and executes them only after explicit owner approval. ML runtimes are installed on demand.\n",
        )
    output.seek(0)
    return output
