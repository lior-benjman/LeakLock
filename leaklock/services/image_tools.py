from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..models import Detection


@contextmanager
def crop_detection_to_temp_file(image_path: Path, detection: Detection):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError(
            "Pillow is required for cropping detection regions."
        ) from exc

    with Image.open(image_path) as image:
        box = (
            int(detection.box.x1),
            int(detection.box.y1),
            int(detection.box.x2),
            int(detection.box.y2),
        )
        cropped = image.crop(box)
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            cropped.save(temp_path)
            yield temp_path
        finally:
            temp_path.unlink(missing_ok=True)
