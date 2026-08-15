from __future__ import annotations

from pathlib import Path

from ..models import Detection, OcrExtraction
from ..services.image_tools import crop_detection_to_temp_file
from ..services.ocr_providers import (
    EasyOcrProvider,
    FallbackOcrProvider,
    LazyOcrProvider,
    OcrProvider,
    RapidOcrProvider,
    TrOcrProvider,
    TesseractOcrProvider,
)


class OcrExtractionLayer:
    """Extracts OCR text for document-like detections."""

    def __init__(self, provider: OcrProvider | None = None) -> None:
        self._provider = provider

    def _default_provider(self) -> OcrProvider:
        providers: list[OcrProvider] = []

        # Ordered cheapest/fastest -> most expensive. FallbackOcrProvider stops
        # at the first provider whose result is "good enough", so this order
        # determines how often the expensive TrOCR pass actually gets paid for.
        # RapidOCR: 1 inference pass. EasyOCR: up to 5, exits on first hit.
        # Tesseract: up to 10 (5 variants x 2 PSM), exits on first hit.
        # TrOCR: up to 39 (3 variants x ~13 slices) — last resort only.
        #
        # RapidOCR is used on nearly every document, so it's loaded eagerly
        # below (and pre-warmed at startup, see warm_up()). The other three
        # are only needed when the cheap tier can't read an image, so they're
        # wrapped in LazyOcrProvider: their models don't load into memory at
        # all until a document actually escalates to that tier. This avoids
        # permanently paying the memory cost of EasyOCR/Tesseract/TrOCR on
        # every running instance for documents that never need them.
        try:
            providers.append(RapidOcrProvider())
        except RuntimeError:
            pass  # genuinely unavailable; the lazy tiers below can still cover it

        providers.append(LazyOcrProvider("easyocr", EasyOcrProvider))
        providers.append(LazyOcrProvider("tesseract", TesseractOcrProvider))
        providers.append(LazyOcrProvider("trocr", TrOcrProvider, min_input_score=20))

        return FallbackOcrProvider(providers)

    def _get_provider(self) -> OcrProvider:
        if self._provider is None:
            self._provider = self._default_provider()
        return self._provider

    def warm_up(self) -> None:
        """Constructs the default provider chain now, loading the eager
        (RapidOCR) tier before the first real request. The lazy tiers are
        unaffected — they still only load when a document actually needs
        them, not at warm-up time."""
        self._get_provider()

    def extract(self, image_path: Path, detection: Detection) -> OcrExtraction:
        with crop_detection_to_temp_file(image_path=image_path, detection=detection) as crop_path:
            return self._get_provider().extract_text(crop_path)


class FastTextRegionOcrExtractionLayer(OcrExtractionLayer):
    """OCR layer for text-region gate crops.

    Text-region crops are already likely to contain text, so this keeps the
    path fast by avoiding the heavyweight TrOCR/EasyOCR fallback chain.
    """

    def _default_provider(self) -> OcrProvider:
        providers: list[OcrProvider] = []
        errors: list[str] = []

        try:
            providers.append(RapidOcrProvider())
        except RuntimeError as exc:
            errors.append(f"RapidOCR unavailable: {exc}")

        try:
            providers.append(TesseractOcrProvider())
        except RuntimeError as exc:
            errors.append(f"Tesseract unavailable: {exc}")

        if providers:
            return FallbackOcrProvider(providers)

        return UnavailableOcrProvider(
            " | ".join(errors) if errors else "Fast text-region OCR provider is not configured yet"
        )
