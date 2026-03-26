from __future__ import annotations

from typing import Protocol

from ..models import OcrExtraction


class ReasonGenerator(Protocol):
    def generate_ocr_reason(
        self,
        routed_from_class: str,
        extraction: OcrExtraction,
        classification: dict[str, object],
    ) -> str:
        """Generate a human-readable OCR risk explanation."""


class TemplateReasonGenerator:
    def generate_ocr_reason(
        self,
        routed_from_class: str,
        extraction: OcrExtraction,
        classification: dict[str, object],
    ) -> str:
        features = classification.get("matched_features", [])
        if not extraction.text.strip():
            return (
                f"{routed_from_class.title()} was detected, but no OCR text was extracted, "
                "so the text-sensitive risk stayed low"
            )

        if features:
            features_text = ", ".join(str(feature) for feature in features)
            return (
                f"{routed_from_class.title()} text contains sensitive indicators: {features_text}"
            )

        return f"{routed_from_class.title()} text was extracted and reviewed for sensitive personal data"
