from __future__ import annotations

import json
import logging
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, current_app, g, request
from flask_login import current_user


_error_lock = threading.Lock()
_log_lock = threading.Lock()
_LOG_MAX_BYTES = 2 * 1024 * 1024
_LOG_BACKUPS = 3


class ResilientOperationsHandler(logging.Handler):
    """Append one record at a time so Windows never holds the log file open."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._vedock_operations_handler = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            encoded = (self.format(record) + "\n").encode("utf-8", errors="replace")
            with _log_lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed(len(encoded))
                with self.path.open("ab") as stream:
                    stream.write(encoded)
        except (OSError, ValueError):
            # Operational logging must never break or flood a live request when a
            # removable/network-backed storage volume is briefly unavailable.
            return

    def _rotate_if_needed(self, incoming_size: int) -> None:
        try:
            current_size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if current_size + incoming_size <= _LOG_MAX_BYTES:
            return
        try:
            oldest = self.path.with_name(f"{self.path.name}.{_LOG_BACKUPS}")
            oldest.unlink(missing_ok=True)
            for index in range(_LOG_BACKUPS - 1, 0, -1):
                source = self.path.with_name(f"{self.path.name}.{index}")
                if source.exists():
                    source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
            self.path.replace(self.path.with_name(f"{self.path.name}.1"))
        except OSError:
            # Another process or a virus scanner may briefly own the file. Keep
            # serving requests; the next record will retry the bounded rollover.
            return


def _logs_root(app: Flask | None = None) -> Path:
    application = app or current_app
    root = Path(application.config["STORAGE_ROOT"]) / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_operations_logging(app: Flask) -> None:
    """Write bounded server logs without exposing them outside the admin UI."""
    path = _logs_root(app) / "vedock.log"
    shared_handler = ResilientOperationsHandler(path)
    shared_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    for logger in (app.logger, logging.getLogger("waitress")):
        for existing_handler in list(logger.handlers):
            if getattr(existing_handler, "_vedock_operations_handler", False):
                logger.removeHandler(existing_handler)
                existing_handler.close()
        logger.addHandler(shared_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def log_request(response: Any) -> Any:
    elapsed_ms = 0
    started = getattr(g, "request_started_at", None)
    if started is not None:
        import time

        elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
    current_app.logger.info(
        "request id=%s method=%s path=%s status=%s duration_ms=%s",
        g.get("request_id", "-"),
        request.method,
        request.path[:500],
        response.status_code,
        elapsed_ms,
    )
    response.headers.setdefault("X-Request-ID", g.get("request_id", ""))
    return response


def record_error_event(error: Any, status: int) -> None:
    """Append a privacy-bounded diagnostic event; never store request bodies."""
    original = getattr(error, "original_exception", None) or error
    trace = traceback.format_exc()
    if trace.strip() == "NoneType: None":
        trace = ""
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "request_id": g.get("request_id", ""),
        "status": int(status),
        "method": request.method,
        "path": request.path[:500],
        "endpoint": request.endpoint or "",
        "user_id": int(getattr(getattr(g, "api_user", None), "id", 0) or 0)
        or int(getattr(current_user, "id", 0) or 0),
        "error_type": type(original).__name__,
        "message": str(getattr(error, "description", None) or original)[:1500],
        "traceback": trace[-10_000:] if status >= 500 else "",
    }
    path = _logs_root() / "errors.jsonl"
    encoded = json.dumps(event, ensure_ascii=False, default=str) + "\n"
    with _error_lock:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        if path.stat().st_size > 4 * 1024 * 1024:
            tail = _read_tail_bytes(path, 2 * 1024 * 1024)
            path.write_bytes(tail)
    current_app.logger.error(
        "error id=%s status=%s method=%s path=%s type=%s message=%s",
        event["request_id"],
        status,
        event["method"],
        event["path"],
        event["error_type"],
        event["message"],
    )


def _read_tail_bytes(path: Path, maximum: int) -> bytes:
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - maximum))
        data = stream.read()
    if size > maximum:
        first_newline = data.find(b"\n")
        if first_newline >= 0:
            data = data[first_newline + 1 :]
    return data


def read_error_events(limit: int = 200) -> list[dict[str, Any]]:
    path = _logs_root() / "errors.jsonl"
    if not path.is_file():
        return []
    text = _read_tail_bytes(path, 2 * 1024 * 1024).decode("utf-8", errors="replace")
    output: list[dict[str, Any]] = []
    for line in text.splitlines()[-max(1, min(limit, 1000)) :]:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                output.append(value)
        except json.JSONDecodeError:
            continue
    return list(reversed(output))


def read_application_log(lines: int = 300) -> str:
    path = _logs_root() / "vedock.log"
    if not path.is_file():
        return "No application log entries have been written yet."
    text = _read_tail_bytes(path, 512 * 1024).decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-max(1, min(lines, 2000)) :])
