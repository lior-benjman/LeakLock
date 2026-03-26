from __future__ import annotations

from ..models import Detection, RiskResult


class UnsupportedDetectionLayer:
    """Fallback layer for unsupported detections."""

    def evaluate(self, detection: Detection) -> RiskResult:
        return RiskResult(
            layer_name="unsupported_layer",
            routed_from_class=detection.class_name,
            risk_percent=0,
            reason=f"No specialized risk layer is configured for '{detection.class_name}'",
            evidence={"confidence": detection.confidence},
        )
