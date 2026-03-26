from __future__ import annotations

import re
from typing import Protocol

from ..config import PipelineConfig


class SensitiveTextClassifier(Protocol):
    def classify(self, text: str) -> dict[str, object]:
        """Classify OCR text for privacy risk."""


class RuleBasedSensitiveTextClassifier:
    """
    Baseline OCR risk evaluator.

    This is a stopgap until a trained sensitive-text classifier is available.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def classify(self, text: str) -> dict[str, object]:
        normalized = " ".join(text.lower().split())
        matched_features: list[str] = []
        risk = 0

        for keyword in self._config.ocr_high_risk_patterns:
            if keyword in normalized:
                matched_features.append(keyword)
                risk += self._config.ocr_high_risk_keywords_weight

        digit_sequences = re.findall(r"\d{4,}", text)
        if digit_sequences:
            matched_features.append("long_digit_sequence")
            risk += min(len(digit_sequences) * self._config.ocr_digit_sequence_weight, 30)

        if re.search(r"\bname\b", normalized):
            matched_features.append("name")
            risk += self._config.ocr_name_weight

        if "@" in text:
            matched_features.append("email")
            risk += 20

        risk = min(risk, self._config.max_ocr_risk_percent)
        return {
            "risk_percent": risk,
            "matched_features": matched_features,
            "model": "rule_based_sensitive_text_classifier",
        }
