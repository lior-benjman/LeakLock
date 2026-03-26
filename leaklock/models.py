from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["box"] = self.box.to_dict()
        return data


@dataclass(slots=True)
class AgeEstimate:
    age_years: int | None
    confidence: float | None = None
    provider: str = "unavailable"
    details: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OcrExtraction:
    text: str
    provider: str
    details: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RiskResult:
    layer_name: str
    routed_from_class: str
    risk_percent: int
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "routed_from_class": self.routed_from_class,
            "risk_percent": self.risk_percent,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class DetectionAnalysis:
    detection: Detection
    route: str
    risk: RiskResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection": self.detection.to_dict(),
            "route": self.route,
            "risk": self.risk.to_dict(),
        }


@dataclass(slots=True)
class ImageAnalysisResult:
    image_path: Path
    overall_risk_percent: int
    analyses: list[DetectionAnalysis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "overall_risk_percent": self.overall_risk_percent,
            "analyses": [item.to_dict() for item in self.analyses],
        }
