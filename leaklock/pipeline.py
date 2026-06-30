from __future__ import annotations

import time
from pathlib import Path

from .config import PipelineConfig
from .layers.detection import YoloDetectionLayer
from .layers.face_age import FaceAgeRiskLayer
from .layers.license_plate import LicensePlateRiskLayer
from .layers.ocr import OcrExtractionLayer
from .layers.ocr_risk import OcrRiskEvaluationLayer
from .layers.routing import DetectionRouter
from .layers.unsupported import UnsupportedDetectionLayer
from .models import BoundingBox, Detection, DetectionAnalysis, ImageAnalysisResult, OcrExtraction


class LeakLockPipeline:
    """Coordinates the layered risk-analysis flow."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._detection_layer = YoloDetectionLayer(self._config)
        self._router = DetectionRouter(self._config)
        self._face_layer = FaceAgeRiskLayer(self._config)
        self._ocr_layer = OcrExtractionLayer()
        self._ocr_risk_layer = OcrRiskEvaluationLayer(self._config)
        self._license_plate_layer = LicensePlateRiskLayer(self._config)
        self._unsupported_layer = UnsupportedDetectionLayer()

    def analyze_image(self, image_path: str | Path) -> ImageAnalysisResult:
        image = Path(image_path).resolve()
        if not image.exists():
            raise FileNotFoundError(f"Input image was not found at: {image}")

        total_start = time.perf_counter()
        detect_start = time.perf_counter()
        detections = self._detection_layer.detect(image)
        print(f"[LeakLock] {image.name}: YOLO detect found {len(detections)} object(s) in {time.perf_counter() - detect_start:.2f}s")

        analyses: list[DetectionAnalysis] = []

        for detection in detections:
            route = self._router.route(detection)
            step_start = time.perf_counter()
            extra_info = ""

            if route == "face_age_layer":
                risk = self._face_layer.evaluate(image_path=image, detection=detection)
                provider = (risk.evidence or {}).get("age_estimate", {}).get("provider")
                if provider:
                    extra_info = f", age_provider={provider}"
            elif route == "ocr_extraction_layer":
                extraction = self._ocr_layer.extract(image_path=image, detection=detection)
                risk = self._ocr_risk_layer.evaluate(
                    routed_from_class=detection.class_name,
                    extraction=extraction,
                )
                extra_info = f", ocr_provider={extraction.provider}"
            elif route == "license_plate_risk_layer":
                risk = self._license_plate_layer.evaluate(detection)
            else:
                risk = self._unsupported_layer.evaluate(detection)

            elapsed = time.perf_counter() - step_start
            print(
                f"[LeakLock] {image.name}: {detection.class_name} -> {route} "
                f"took {elapsed:.2f}s{extra_info}"
            )

            analyses.append(DetectionAnalysis(detection=detection, route=route, risk=risk))

        overall_risk = max((item.risk.risk_percent for item in analyses), default=0)
        print(f"[LeakLock] {image.name}: total analysis time {time.perf_counter() - total_start:.2f}s")
        return ImageAnalysisResult(
            image_path=image,
            overall_risk_percent=overall_risk,
            analyses=analyses,
        )

    def warm_up(self) -> None:
        """Force each layer's eager model to load against a throwaway image,
        so the first real request doesn't pay model-loading latency. OCR's
        lazy fallback tiers (EasyOCR/Tesseract/TrOCR) are intentionally
        excluded — they load on first actual need instead, see
        OcrExtractionLayer."""
        import tempfile

        try:
            from PIL import Image
        except ImportError:
            return

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.new("RGB", (64, 64), color=(128, 128, 128)).save(tmp.name)
            dummy_path = Path(tmp.name)

        try:
            self._detection_layer.detect(dummy_path)  # loads YOLO weights

            dummy_box = BoundingBox(x1=0, y1=0, x2=64, y2=64)
            # Loads only the eager (RapidOCR) tier. Deliberately not calling
            # .extract() here: a textless dummy image would never cross the
            # "good enough" score, so FallbackOcrProvider would escalate
            # through every lazy tier (EasyOCR/Tesseract/TrOCR) and force-load
            # all of them anyway, defeating the point of making them lazy.
            self._ocr_layer.warm_up()
            self._face_layer.evaluate(
                image_path=dummy_path,
                detection=Detection(class_id=2, class_name="face", confidence=1.0, box=dummy_box),
            )  # loads the age-estimator chain
            self._ocr_risk_layer.evaluate(
                routed_from_class="document",
                extraction=OcrExtraction(text="warm up", provider="warmup"),
            )  # loads the zero-shot document classifier
        except Exception:
            # Warm-up is best-effort: a failure here just defers the cost
            # to whichever real request first exercises that code path.
            pass
        finally:
            dummy_path.unlink(missing_ok=True)
