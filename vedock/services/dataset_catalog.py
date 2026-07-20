from __future__ import annotations

from pathlib import Path

from werkzeug.datastructures import FileStorage

from vedock.models import RawDataset, User

from .datasets import DatasetError, import_upload


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMMUNITY_DATASETS = [
    {
        "id": "prompt-response-starter",
        "name": "Prompt / response starter",
        "creator": "Vedock community",
        "description": "Small paired-text dataset for exploring mapping, cleaning, formatting patterns, and causal-LM recipes.",
        "path": PROJECT_ROOT / "demo" / "story_prompts.csv",
        "recommended_schema": "prompt_response",
    },
    {
        "id": "pattern-sequence-starter",
        "name": "Pattern sequence starter",
        "creator": "Vedock community",
        "description": "Tiny token sequences for configuring the fast n-gram pattern runtime.",
        "path": PROJECT_ROOT / "demo" / "pattern_sequences.txt",
        "recommended_schema": "text_completion",
    },
]


def import_community_dataset(identifier: str, owner: User) -> RawDataset:
    entry = next((item for item in COMMUNITY_DATASETS if item["id"] == identifier), None)
    if not entry or not Path(entry["path"]).is_file():
        raise DatasetError("Community dataset is unavailable.")
    path = Path(entry["path"])
    with path.open("rb") as stream:
        upload = FileStorage(stream=stream, filename=path.name, content_type="text/csv" if path.suffix == ".csv" else "text/plain")
        return import_upload(upload, owner, entry["name"], f"{entry['description']} Creator: {entry['creator']}.")
