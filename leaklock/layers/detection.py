from __future__ import annotations

from pathlib import Path

from ..config import PipelineConfig
from ..models import BoundingBox, Detection


class YoloDetectionLayer:
    """Runs YOLOv8 inference and normalizes the detections."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "ultralytics is required for the detection layer. "
                "Install it in the runtime environment before running the pipeline."
            ) from exc

        weights_path = self._config.yolo_weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights were not found at: {weights_path}")

        self._model = YOLO(str(weights_path))
        return self._model

    def detect(self, image_path: Path) -> list[Detection]:
        model = self._load_model()
        results = model.predict(
            source=str(image_path),
            conf=self._config.detection_confidence_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: list[Detection] = []
        xyxy_list = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()
        cls_list = boxes.cls.tolist()

        for xyxy, confidence, class_id_float in zip(xyxy_list, conf_list, cls_list):
            class_id = int(class_id_float)
            class_name = str(names[class_id]).strip().lower()
            box = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidence),
                    box=box,
                )
            )

        return detections
