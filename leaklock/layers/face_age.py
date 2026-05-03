from __future__ import annotations

from pathlib import Path

from ..config import AgeRiskBand, PipelineConfig
from ..models import Detection, RiskResult
from ..services.age_estimators import (
    AgeEstimator,
    DeepFaceAgeEstimator,
    FallbackAgeEstimator,
    HuggingFaceOnnxAgeEstimator,
    HuggingFaceTransformersAgeEstimator,
    UnavailableAgeEstimator,
)


class FaceAgeRiskLayer:
    """Estimates age for face detections and converts it into risk."""

    def __init__(
        self,
        config: PipelineConfig,
        estimator: AgeEstimator | None = None,
    ) -> None:
        self._config = config
        self._estimator = estimator or self._default_estimator()

    def _default_estimator(self) -> AgeEstimator:
        return FallbackAgeEstimator(
            [
                HuggingFaceOnnxAgeEstimator(
                    repo_id=self._config.hf_age_model_repo_id,
                ),
                HuggingFaceTransformersAgeEstimator(
                    model_id=self._config.hf_age_classifier_model_id,
                ),
                DeepFaceAgeEstimator(
                    detector_backend=self._config.deepface_detector_backend,
                ),
                UnavailableAgeEstimator(),
            ]
        )

    def evaluate(self, image_path: Path, detection: Detection) -> RiskResult:
        try:
            estimate = self._estimator.estimate(image_path=image_path, detection=detection)
        except RuntimeError as exc:
            return RiskResult(
                layer_name="face_age_layer",
                routed_from_class=detection.class_name,
                risk_percent=0,
                reason="Face detected, but age estimation could not run in the current environment",
                evidence={"error": str(exc)},
            )
        if estimate.age_years is None:
            return RiskResult(
                layer_name="face_age_layer",
                routed_from_class=detection.class_name,
                risk_percent=0,
                reason="Face detected, but no age estimate was available in the current environment",
                evidence={"age_estimate": estimate.to_dict()},
            )

        band = self._find_band(estimate.age_years)
        return RiskResult(
            layer_name="face_age_layer",
            routed_from_class=detection.class_name,
            risk_percent=band.risk_percent,
            reason=band.reason_template,
            evidence={"age_estimate": estimate.to_dict()},
        )

    def _find_band(self, age_years: int) -> AgeRiskBand:
        for band in self._config.age_risk_bands:
            if band.min_age <= age_years <= band.max_age:
                return band
        return self._config.age_risk_bands[-1]
