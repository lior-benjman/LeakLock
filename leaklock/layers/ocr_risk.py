from __future__ import annotations

import re

from ..config import PipelineConfig
from ..models import OcrExtraction, RiskResult
from ..services.reason_generators import ReasonGenerator, TemplateReasonGenerator
from ..services.text_classifiers import SensitiveTextClassifier, RuleBasedSensitiveTextClassifier


class OcrRiskEvaluationLayer:
    """Scores extracted OCR text for privacy risk."""

    def __init__(
        self,
        config: PipelineConfig,
        classifier: SensitiveTextClassifier | None = None,
        reason_generator: ReasonGenerator | None = None,
    ) -> None:
        self._config = config
        self._classifier = classifier or RuleBasedSensitiveTextClassifier(config)
        self._reason_generator = reason_generator or TemplateReasonGenerator()

    def evaluate(self, routed_from_class: str, extraction: OcrExtraction) -> RiskResult:
        classification = self._classifier.classify(extraction.text)
        reason = self._reason_generator.generate_ocr_reason(
            routed_from_class=routed_from_class,
            extraction=extraction,
            classification=classification,
        )
        return RiskResult(
            layer_name="ocr_risk_evaluation_layer",
            routed_from_class=routed_from_class,
            risk_percent=classification["risk_percent"],
            reason=reason,
            evidence={
                "ocr": extraction.to_dict(),
                "classification": classification,
                "digit_sequences": re.findall(r"\d{4,}", extraction.text),
            },
        )
