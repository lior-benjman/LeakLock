from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from leaklock.config import PipelineConfig
from leaklock.layers.ocr_risk import OcrRiskEvaluationLayer
from leaklock.models import BoundingBox, Detection, OcrExtraction
from leaklock.pipeline import LeakLockPipeline
from leaklock.services.text_classifiers import RuleBasedSensitiveTextClassifier
from leaklock.services.text_region_gate import TextRegionGate


class _NoDetectionsLayer:
    def detect(self, image_path: Path) -> list[Detection]:
        return []


class _SingleTextRegionGate:
    def detect(self, image_path: Path) -> list[Detection]:
        return [
            Detection(
                class_id=-2,
                class_name="text_region",
                confidence=0.8,
                box=BoundingBox(x1=0, y1=0, x2=120, y2=60),
            )
        ]


class _TextOcrLayer:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract(self, image_path: Path, detection: Detection) -> OcrExtraction:
        return OcrExtraction(text=self._text, provider="test")


class TextRegionPipelineTests(unittest.TestCase):
    def _make_pipeline(self, text: str) -> LeakLockPipeline:
        config = PipelineConfig()
        config.enable_document_ml_risk = False
        pipeline = LeakLockPipeline.__new__(LeakLockPipeline)
        pipeline._config = config  # type: ignore[attr-defined]
        pipeline._detection_layer = _NoDetectionsLayer()  # type: ignore[attr-defined]
        pipeline._text_region_ocr_layer = _TextOcrLayer(text)  # type: ignore[attr-defined]
        pipeline._ocr_risk_layer = OcrRiskEvaluationLayer(config)  # type: ignore[attr-defined]
        pipeline._text_region_gate = _SingleTextRegionGate()  # type: ignore[attr-defined]
        return pipeline

    def _make_temp_image(self) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        image_path = Path(tmp.name)
        Image.new("RGB", (240, 120), color=(255, 255, 255)).save(image_path)
        self.addCleanup(lambda: image_path.unlink(missing_ok=True))
        return image_path

    def test_text_region_gate_routes_sensitive_text_to_ocr_risk(self) -> None:
        pipeline = self._make_pipeline("WiFi password: BlueRiver")

        result = pipeline.analyze_image(self._make_temp_image())

        self.assertEqual(result.overall_risk_percent, 90)
        self.assertEqual(len(result.analyses), 1)
        self.assertEqual(result.analyses[0].route, "text_region_ocr_layer")
        self.assertEqual(result.analyses[0].detection.class_name, "text_region")

    def test_text_region_gate_ignores_harmless_text(self) -> None:
        pipeline = self._make_pipeline("Welcome to the lobby")

        result = pipeline.analyze_image(self._make_temp_image())

        self.assertEqual(result.overall_risk_percent, 0)
        self.assertEqual(result.analyses, [])


class RuleBasedSensitiveTextClassifierTests(unittest.TestCase):
    def test_wifi_password_text_is_sensitive(self) -> None:
        result = RuleBasedSensitiveTextClassifier(PipelineConfig()).classify(
            "SSID: Home WiFi\nPassword: BlueRiver"
        )

        self.assertEqual(result["risk_percent"], 90)
        self.assertIn("credential", result["matched_features"])
        self.assertIn("wifi_network", result["matched_features"])


class TextRegionGateDetectionTests(unittest.TestCase):
    def test_detects_synthetic_text_region_without_full_image_box(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and numpy are required for text region detection")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        image_path = Path(tmp.name)
        self.addCleanup(lambda: image_path.unlink(missing_ok=True))

        image = np.full((320, 800, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "WiFi Password",
            (40, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(image_path), image)

        detections = TextRegionGate(PipelineConfig()).detect(image_path)

        self.assertGreaterEqual(len(detections), 1)
        largest = max(detections, key=lambda item: (item.box.x2 - item.box.x1) * (item.box.y2 - item.box.y1))
        area = (largest.box.x2 - largest.box.x1) * (largest.box.y2 - largest.box.y1)
        self.assertLess(area, 800 * 320 * 0.4)

    def test_detects_bright_text_on_dark_background(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and numpy are required for text region detection")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        image_path = Path(tmp.name)
        self.addCleanup(lambda: image_path.unlink(missing_ok=True))

        image = np.zeros((320, 800, 3), dtype=np.uint8)
        image[:, :, 0] = 50
        image[:, :, 1] = 80
        image[:, :, 2] = 110
        cv2.putText(
            image,
            "WiFi Password",
            (40, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(image_path), image)

        detections = TextRegionGate(PipelineConfig()).detect(image_path)

        self.assertGreaterEqual(len(detections), 1)


if __name__ == "__main__":
    unittest.main()
