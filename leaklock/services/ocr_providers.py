from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import OcrExtraction


class OcrProvider(Protocol):
    def extract_text(self, image_path: Path) -> OcrExtraction:
        """Extract text from an image path."""


class TesseractOcrProvider:
    def __init__(self) -> None:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "pytesseract and Pillow are required for OCR extraction."
            ) from exc

    def extract_text(self, image_path: Path) -> OcrExtraction:
        import pytesseract
        from PIL import Image

        text = pytesseract.image_to_string(Image.open(image_path)).strip()
        return OcrExtraction(
            text=text,
            provider="tesseract",
            details="OCR extracted from cropped detection",
        )


class UnavailableOcrProvider:
    def extract_text(self, image_path: Path) -> OcrExtraction:
        return OcrExtraction(
            text="",
            provider="unavailable",
            details="OCR provider is not configured yet",
        )
