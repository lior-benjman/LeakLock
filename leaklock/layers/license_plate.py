from __future__ import annotations

from ..config import PipelineConfig
from ..models import Detection, RiskResult


class LicensePlateRiskLayer:
    """Assigns a fixed risk to license plate detections."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def evaluate(self, detection: Detection) -> RiskResult:
        return RiskResult(
            layer_name="license_plate_risk_layer",
            routed_from_class=detection.class_name,
            risk_percent=self._config.license_plate_risk_percent,
            reason=self._config.license_plate_reason,
            evidence={"confidence": detection.confidence},
        )
