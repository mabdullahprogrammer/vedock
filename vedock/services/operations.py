from __future__ import annotations

import json
import logging
import threading
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from flask import Flask, current_app, g, request
from flask_login import current_user


_error_lock = threading.Lock()


def _logs_root(app: Flask | None = None) -> Path:
    application = app or current_app
    root = Path(application.config["STORAGE_ROOT"]) / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_operations_logging(app: Flask) -> None:
    """Write bounded server logs without exposing them outside the admin UI."""
    path = _logs_root(app) / "vedock.log"
    for logger in (app.logger, logging.getLogger("waitress")):
        for handler in list(logger.handlers):
            if getattr(handler, "_vedock_operations_handler", False):
                logger.removeHandler(handler)
                handler.close()
        handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler._vedock_operations_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


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
