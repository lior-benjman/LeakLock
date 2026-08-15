from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AgeRiskBand:
    min_age: int
    max_age: int
    risk_percent: int
    reason_template: str


@dataclass(slots=True)
class PipelineConfig:
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    yolo_weights_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "runs_sensitive"
        / "yolov86"
        / "weights"
        / "best.pt"
    )
    detection_confidence_threshold: float = 0.25
    hf_age_model_repo_id: str = "onnx-community/age-gender-prediction-ONNX"
    hf_age_classifier_model_id: str = "nateraw/vit-age-classifier"
    deepface_detector_backend: str = "opencv"
    license_plate_risk_percent: int = 75
    license_plate_reason: str = "License-plate was detected"
    age_risk_bands: tuple[AgeRiskBand, ...] = (
        AgeRiskBand(
            min_age=0,
            max_age=11,
            risk_percent=60,
            reason_template="Face estimated at 11 years old or younger was detected",
        ),
        AgeRiskBand(
            min_age=12,
            max_age=15,
            risk_percent=30,
            reason_template="12-15 years old face detected",
        ),
        AgeRiskBand(
            min_age=16,
            max_age=200,
            risk_percent=20,
            reason_template="Face estimated at 16 years old or older was detected",
        ),
    )
    supported_document_classes: frozenset[str] = frozenset(
        {"card", "document", "id", "id-card", "id_card", "passport"}
    )
    supported_face_classes: frozenset[str] = frozenset({"face"})
    supported_license_plate_classes: frozenset[str] = frozenset({"license-plates", "license_plate"})
    ocr_high_risk_patterns: tuple[str, ...] = (
        "passport",
        "identity",
        "identification",
        "credit card",
        "card number",
        "personal number",
        "date of birth",
        "dob",
        "address",
        "passport no",
        "id number",
        "license no",
        "document no",
    )
    ocr_high_risk_keywords_weight: int = 25
    ocr_digit_sequence_weight: int = 10
    ocr_name_weight: int = 15
    ocr_credential_keyword_weight: int = 75
    ocr_wifi_context_weight: int = 15
    max_ocr_risk_percent: int = 90
    enable_document_ml_risk: bool = True
    enable_text_region_gate: bool = True
    text_region_class_name: str = "text_region"
    text_region_max_image_side: int = 960
    text_region_max_regions: int = 4
    text_region_min_area_ratio: float = 0.0008
    text_region_max_area_ratio: float = 0.18
    text_region_min_aspect_ratio: float = 1.2
    text_region_min_width_ratio: float = 0.12
    text_region_padding_ratio: float = 0.025
