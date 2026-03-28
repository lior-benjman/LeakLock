from __future__ import annotations

from pathlib import Path

from ..models import Detection, OcrExtraction
from ..services.image_tools import crop_detection_to_temp_file
from ..services.ocr_providers import (
    EasyOcrProvider,
    FallbackOcrProvider,
    OcrProvider,
    RapidOcrProvider,
    TrOcrProvider,
    TesseractOcrProvider,
    UnavailableOcrProvider,
)


class OcrExtractionLayer:
    """Extracts OCR text for document-like detections."""

    def __init__(self, provider: OcrProvider | None = None) -> None:
        self._provider = provider or self._default_provider()

    def _default_provider(self) -> OcrProvider:
        providers: list[OcrProvider] = []
        errors: list[str] = []

        try:
            providers.append(TrOcrProvider())
        except RuntimeError as exc:
            errors.append(f"TrOCR unavailable: {exc}")

        try:
            providers.append(RapidOcrProvider())
        except RuntimeError as exc:
            errors.append(f"RapidOCR unavailable: {exc}")

        try:
            providers.append(EasyOcrProvider())
        except RuntimeError as exc:
            errors.append(f"EasyOCR unavailable: {exc}")

        try:
            providers.append(TesseractOcrProvider())
        except RuntimeError as exc:
            errors.append(f"Tesseract unavailable: {exc}")

        if providers:
            return FallbackOcrProvider(providers)

        return UnavailableOcrProvider(" | ".join(errors) if errors else "OCR provider is not configured yet")

    def extract(self, image_path: Path, detection: Detection) -> OcrExtraction:
        with crop_detection_to_temp_file(image_path=image_path, detection=detection) as crop_path:
            return self._provider.extract_text(crop_path)
