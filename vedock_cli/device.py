from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


RUNTIME_MODULES = {
    "text_training": ["torch", "transformers", "datasets", "accelerate", "peft", "safetensors"],
    "image_training": ["torch", "diffusers", "transformers", "accelerate", "PIL"],
    "fast_ml": ["numpy", "pandas", "sklearn", "joblib", "PIL"],
    "desktop": ["webview"],
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def runtime_report() -> list[dict[str, Any]]:
    labels = {
        "text_training": "LLM training",
        "image_training": "Image model tools",
        "fast_ml": "Fast pattern and classical ML",
        "desktop": "Vedock Desktop",
    }
    records = []
    for key, modules in RUNTIME_MODULES.items():
        missing = [name for name in modules if not _module_available(name)]
        records.append(
            {
                "key": key,
                "name": labels[key],
                "installed": not missing,
                "missing": missing,
                "modules": modules,
            }
        )
    return records


def local_device_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "os": platform.system(),
        "hostname": platform.node() or "Vedock device",
        "processor": platform.processor() or platform.machine() or "Unknown CPU",
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "ram_total_bytes": None,
        "ram_available_bytes": None,
        "disk_total_bytes": None,
        "disk_free_bytes": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpus": [],
        "runtimes": runtime_report(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        report["ram_total_bytes"] = memory.total
        report["ram_available_bytes"] = memory.available
    except Exception as exc:
        report["memory_error"] = str(exc)
    try:
        usage = shutil.disk_usage(Path.home())
        report["disk_total_bytes"] = usage.total
        report["disk_free_bytes"] = usage.free
    except Exception as exc:
        report["disk_error"] = str(exc)
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = torch.version.cuda
        if report["cuda_available"]:
            report["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        report["cuda_error"] = str(exc)
    return report
