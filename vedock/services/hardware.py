from __future__ import annotations

import importlib.metadata
import shutil
import sys
from pathlib import Path
from typing import Any

from flask import current_app


PACKAGE_NAMES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "bitsandbytes",
    "flask",
    "flask-sqlalchemy",
    "sqlalchemy",
    "safetensors",
    "pandas",
    "pyarrow",
    "psutil",
    "scikit-learn",
    "Pillow",
    "joblib",
]


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def system_report(*, include_private: bool | None = None) -> dict[str, Any]:
    """Return node capabilities without leaking hosted-machine details.

    A hosted Vedock instance is a control plane and inference service. Its
    filesystem, Python executable, packages and hardware are implementation
    details, not user-facing device information. Full diagnostics are only
    returned by a local-compute node (or an explicit trusted caller).
    """
    if include_private is None:
        include_private = current_app.config["NODE_MODE"] == "local_compute"
    if not include_private:
        return {
            "private_details_hidden": True,
            "node": {
                "mode": current_app.config["NODE_MODE"],
                "name": current_app.config["APP_NAME"],
                "data_ownership": "private_host_storage",
                "inference_location": "hosted_service",
                "training_location": "connected_user_device",
            },
            "capabilities": {
                "hosted_inference": True,
                "connected_device_training": True,
                "local_device_diagnostics": False,
            },
        }
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": dependency_versions(),
        "cuda_available": False,
        "cuda_compiled": None,
        "devices": [],
        "ram_total_bytes": None,
        "ram_available_bytes": None,
        "storage_total_bytes": None,
        "storage_free_bytes": None,
        "storymaker_root": str(current_app.config["STORYMAKER_ROOT"]),
        "storymaker_root_exists": Path(current_app.config["STORYMAKER_ROOT"]).exists(),
        "protected_roots": [str(path) for path in current_app.config["PROTECTED_ROOTS"]],
        "node": {
            "mode": current_app.config["NODE_MODE"],
            "name": current_app.config["NODE_NAME"],
            "control_plane_url": current_app.config["CONTROL_PLANE_URL"] or None,
            "storage_root": str(current_app.config["STORAGE_ROOT"]),
            "data_ownership": "local_node",
            "inference_location": "local_node",
            "training_location": "local_worker",
        },
    }
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_compiled"] = torch.version.cuda
        if report["cuda_available"]:
            report["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": torch.cuda.get_device_properties(index).total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        report["torch_error"] = str(exc)
    try:
        import psutil

        memory = psutil.virtual_memory()
        report["ram_total_bytes"] = memory.total
        report["ram_available_bytes"] = memory.available
    except Exception as exc:
        report["memory_error"] = str(exc)
    try:
        usage = shutil.disk_usage(Path(current_app.config["STORAGE_ROOT"]))
        report["storage_total_bytes"] = usage.total
        report["storage_free_bytes"] = usage.free
    except Exception as exc:
        report["storage_error"] = str(exc)
    packages = report["packages"]
    report["capabilities"] = {
        "inference": bool(packages.get("torch") and packages.get("transformers")),
        "full_training": bool(packages.get("torch") and packages.get("transformers") and packages.get("accelerate")),
        "lora": bool(packages.get("peft") and packages.get("accelerate")),
        "qlora": bool(report["cuda_available"] and packages.get("bitsandbytes") and packages.get("peft")),
        "parquet": bool(packages.get("pyarrow")),
        "fast_pattern_models": True,
        "fast_image_classification": bool(packages.get("scikit-learn") and packages.get("Pillow") and packages.get("joblib")),
    }
    return report
