from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


@dataclass(frozen=True)
class ModelReference:
    kind: str
    source: str
    revision: str | None = None


def make_huggingface_reference(repository: str, revision: str = "main") -> str:
    normalized_repository = repository.strip().strip("/")
    normalized_revision = revision.strip() or "main"
    return f"hf://{normalized_repository}?revision={quote(normalized_revision, safe='')}"


def make_scratch_reference(identifier: str) -> str:
    return f"scratch://{identifier}"


def parse_model_reference(value: str) -> ModelReference:
    value = str(value).strip()
    if value.startswith("hf://"):
        parsed = urlparse(value)
        repository = f"{parsed.netloc}{parsed.path}".strip("/")
        revision = unquote((parse_qs(parsed.query).get("revision") or ["main"])[0])
        return ModelReference("huggingface", repository, revision)
    if value.startswith("scratch://"):
        parsed = urlparse(value)
        identifier = f"{parsed.netloc}{parsed.path}".strip("/")
        return ModelReference("scratch", identifier)
    return ModelReference("local", str(Path(value).expanduser().resolve()))
