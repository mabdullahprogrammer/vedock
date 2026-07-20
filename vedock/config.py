from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: str | Path) -> Path:
    raw = os.getenv(name, "").strip()
    path = Path(raw).expanduser() if raw else Path(default)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _default_storymaker_root() -> Path:
    requested = Path(r"D:\LLM\StoryMaker")
    discovered = Path(r"D:\LLM\new-llm\LLM-2025\StoryMaker")
    return requested if requested.exists() else discovered


class Config:
    APP_NAME = os.getenv("APP_NAME", "Vedock")
    APP_SHORT_NAME = os.getenv("APP_SHORT_NAME", APP_NAME)
    CLI_NAME = os.getenv("CLI_NAME", "vedock")
    APP_TAGLINE = os.getenv("APP_TAGLINE", "Build any AI. No code. Full control.")
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = _env_int("APP_PORT", 5464)
    # Safe default: a publicly reachable Vedock host may serve inference, but it
    # must never execute another user's training workload. The installer writes
    # local_compute explicitly on the user's own machine.
    NODE_MODE = os.getenv("NODE_MODE", "hosted_inference").strip().lower()
    NODE_NAME = os.getenv("NODE_NAME", os.environ.get("COMPUTERNAME", "Vedock node"))
    CONTROL_PLANE_URL = os.getenv("CONTROL_PLANE_URL", "").strip()

    SECRET_KEY = os.getenv("SECRET_KEY", "vedock-local-dev-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'instance' / 'vedock.db').as_posix()}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "connect_args": {"timeout": 30},
        "pool_pre_ping": True,
    }

    STORAGE_ROOT = _env_path("VEDOCK_STORAGE_ROOT", "storage")
    DISTRIBUTION_ROOT = _env_path("VEDOCK_DISTRIBUTION_ROOT", "distribution")
    STORYMAKER_ROOT = _env_path("STORYMAKER_ROOT", _default_storymaker_root())
    STORYMAKER_FINAL_PATH = _env_path("STORYMAKER_FINAL_PATH", STORYMAKER_ROOT / "gpt-storygen-final")
    STORYMAKER_FINETUNED_PATH = _env_path("STORYMAKER_FINETUNED_PATH", STORYMAKER_ROOT / "gpt2fintuned_storymaker")
    RUNTIME_PYTHON = _env_path(
        "VEDOCK_RUNTIME_PYTHON",
        BASE_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    )
    PROTECTED_ROOTS = tuple(
        dict.fromkeys(
            [
                Path(r"D:\LLM\StoryMaker").resolve(),
                Path(r"D:\LLM\new-llm\LLM-2025\StoryMaker").resolve(),
                STORYMAKER_ROOT,
            ]
        )
    )

    MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 100)
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    URL_DOWNLOAD_MAX_BYTES = _env_int("URL_DOWNLOAD_MAX_MB", 100) * 1024 * 1024
    URL_CONNECT_TIMEOUT = 6
    URL_READ_TIMEOUT = 30
    URL_MAX_REDIRECTS = 3
    DATASET_SYNC_MAX_BYTES = 5 * 1024 * 1024
    DATASET_INSPECT_MAX_ROWS = 100_000
    DATASET_PREVIEW_ROWS = 20
    MAX_PROMPT_CHARS = 20_000
    MAX_NEW_TOKENS = 512
    OFFLINE_MODE = _env_bool("OFFLINE_MODE", False)
    MODEL_TRAINING_ENABLED = _env_bool("MODEL_TRAINING_ENABLED", True)

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", False)
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    JSON_SORT_KEYS = False
    DEBUG = _env_bool("FLASK_DEBUG", False)


class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False
