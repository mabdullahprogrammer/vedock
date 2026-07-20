from __future__ import annotations

import hashlib
from html import escape
from io import BytesIO
from pathlib import Path

from flask import current_app
from werkzeug.datastructures import FileStorage

from vedock.models import ModelRecord, new_id

from .paths import assert_writable_path


MAX_COVER_BYTES = 5 * 1024 * 1024
ALLOWED_COVER_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


def save_model_cover(model: ModelRecord, upload: FileStorage) -> str:
    payload = upload.stream.read(MAX_COVER_BYTES + 1)
    if not payload:
        raise ValueError("Choose a model image.")
    if len(payload) > MAX_COVER_BYTES:
        raise ValueError("Model images are limited to 5 MB.")
    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as image:
            image.verify()
            suffix = ALLOWED_COVER_FORMATS.get(str(image.format or "").upper())
    except Exception as exc:
        raise ValueError("Use a valid PNG, JPEG, or WebP model image.") from exc
    if not suffix:
        raise ValueError("Use a PNG, JPEG, or WebP model image.")
    directory = assert_writable_path(Path(current_app.config["STORAGE_ROOT"]) / "media" / "model-covers" / str(model.owner_id or "shared") / model.id)
    directory.mkdir(parents=True, exist_ok=True)
    destination = assert_writable_path(directory / f"{new_id()}{suffix}")
    with destination.open("xb") as stream:
        stream.write(payload)
    return str(destination)


def generated_cover_svg(model: ModelRecord) -> str:
    digest = hashlib.sha256(model.id.encode("utf-8")).digest()
    hue_a = int.from_bytes(digest[:2], "big") % 360
    hue_b = (hue_a + 45 + digest[2] % 130) % 360
    hue_c = (hue_b + 80 + digest[3] % 100) % 360
    initials = escape("".join(word[0] for word in model.name.split()[:2] if word).upper() or "AI")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 720" role="img" aria-label="{initials} model cover">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="hsl({hue_a} 74% 58%)"/><stop offset=".52" stop-color="hsl({hue_b} 72% 42%)"/><stop offset="1" stop-color="hsl({hue_c} 78% 25%)"/></linearGradient><filter id="blur"><feGaussianBlur stdDeviation="55"/></filter></defs>
<rect width="1200" height="720" fill="url(#g)"/><circle cx="190" cy="110" r="260" fill="white" opacity=".13" filter="url(#blur)"/><circle cx="1030" cy="620" r="330" fill="black" opacity=".18" filter="url(#blur)"/><path d="M0 620 C250 470 420 760 720 560 S1040 350 1200 470 V720 H0Z" fill="white" opacity=".1"/><text x="72" y="640" fill="white" opacity=".92" font-family="Segoe UI,Arial,sans-serif" font-size="154" font-weight="800" letter-spacing="-8">{initials}</text></svg>'''
