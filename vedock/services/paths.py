from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from flask import current_app


class UnsafePathError(ValueError):
    pass


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_writable_path(path: Path) -> Path:
    resolved = path.resolve()
    storage_root = Path(current_app.config["STORAGE_ROOT"]).resolve()
    if not is_within(resolved, storage_root):
        raise UnsafePathError(f"Output must remain under the Vedock storage root: {storage_root}")
    for protected in current_app.config["PROTECTED_ROOTS"]:
        if is_within(resolved, Path(protected)):
            raise UnsafePathError(f"The protected StoryMaker path is read-only: {protected}")
    return resolved


def allocate_directory(*parts: str) -> Path:
    target = assert_writable_path(Path(current_app.config["STORAGE_ROOT"]).joinpath(*parts))
    target.mkdir(parents=True, exist_ok=False)
    return target


def ensure_storage_layout() -> None:
    root = Path(current_app.config["STORAGE_ROOT"]).resolve()
    for relative in [
        "datasets/raw",
        "datasets/processed",
        "datasets/temporary",
        "models",
        "jobs",
        "exports",
        "temporary",
    ]:
        target = assert_writable_path(root / relative)
        target.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, data: Any) -> None:
    target = assert_writable_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".vedock-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
