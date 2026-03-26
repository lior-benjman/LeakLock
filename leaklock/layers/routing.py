from __future__ import annotations

from ..config import PipelineConfig
from ..models import Detection


class DetectionRouter:
    """Maps detections into specialized processing layers."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def route(self, detection: Detection) -> str:
        class_name = detection.class_name
        if class_name in self._config.supported_face_classes:
            return "face_age_layer"
        if class_name in self._config.supported_document_classes:
            return "ocr_extraction_layer"
        if class_name in self._config.supported_license_plate_classes:
            return "license_plate_risk_layer"
        return "unsupported_layer"
