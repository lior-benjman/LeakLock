from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from ..models import AgeEstimate, Detection
from .image_tools import crop_detection_to_temp_file


class AgeEstimator(Protocol):
    def estimate(self, image_path: Path, detection: Detection) -> AgeEstimate:
        """Return an age estimate for the given face detection."""


class UnavailableAgeEstimator:
    def estimate(self, image_path: Path, detection: Detection) -> AgeEstimate:
        return AgeEstimate(
            age_years=None,
            confidence=None,
            provider="unavailable",
            details="No face-age model has been configured yet",
        )


class FixedAgeEstimator:
    """Useful for local testing until a real age model is plugged in."""

    def __init__(self, age_years: int, confidence: float = 1.0) -> None:
        self._age_years = age_years
        self._confidence = confidence

    def estimate(self, image_path: Path, detection: Detection) -> AgeEstimate:
        return AgeEstimate(
            age_years=self._age_years,
            confidence=self._confidence,
            provider="fixed",
            details="Fixed age estimator for testing",
        )


class DeepFaceAgeEstimator:
    """
    Age estimator backed by DeepFace.

    DeepFace documents age analysis support in its official repository, including
    a reported age-model MAE of about 4.65 years.
    """

    def __init__(self, detector_backend: str = "opencv") -> None:
        self._detector_backend = detector_backend
        # DeepFace and related detector packages may require legacy tf.keras behavior
        # when TensorFlow 2.16+ pulls in Keras 3 by default.
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    def _load_deepface(self):
        try:
            from deepface import DeepFace
            return DeepFace
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "deepface is required for real face-age estimation."
            ) from exc
        except ValueError as exc:  # pragma: no cover - depends on runtime
            message = str(exc)
            if "requires tf-keras package" in message.lower():
                raise RuntimeError(
                    "DeepFace requires the tf-keras compatibility package with your "
                    "current TensorFlow/Keras setup. Install it with `pip install tf-keras`, "
                    "restart the kernel, and try again."
                ) from exc
            raise RuntimeError(
                f"DeepFace failed to initialize: {message}"
            ) from exc

    def estimate(self, image_path: Path, detection: Detection) -> AgeEstimate:
        DeepFace = self._load_deepface()

        with crop_detection_to_temp_file(image_path=image_path, detection=detection) as crop_path:
            result = DeepFace.analyze(
                img_path=str(crop_path),
                actions=["age"],
                detector_backend=self._detector_backend,
                enforce_detection=False,
            )

        if isinstance(result, list):
            result = result[0] if result else {}

        age_value = result.get("age")
        if age_value is None:
            return AgeEstimate(
                age_years=None,
                confidence=None,
                provider="deepface",
                details="DeepFace did not return an age estimate",
            )

        try:
            age_years = int(round(float(age_value)))
        except (TypeError, ValueError):
            age_years = None

        return AgeEstimate(
            age_years=age_years,
            confidence=None,
            provider="deepface",
            details=f"detector_backend={self._detector_backend}",
        )


class HuggingFaceOnnxAgeEstimator:
    """
    Age estimator backed by the Hugging Face ONNX age-gender model files.

    The public model card describes a helper module, but the repo files available
    to snapshot downloads can vary. To avoid depending on that helper, this
    estimator runs the exported ONNX model directly with onnxruntime.
    """

    def __init__(self, repo_id: str = "onnx-community/age-gender-prediction-ONNX") -> None:
        self._repo_id = repo_id
        self._preprocessor_config = None
        self._session = None
        self._model_path: Path | None = None

    def _load_runtime(self):
        if self._preprocessor_config is not None and self._session is not None:
            return self._preprocessor_config, self._session, self._model_path

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "huggingface_hub is required for the ONNX face-age estimator."
            ) from exc

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "onnxruntime is required for the ONNX face-age estimator."
            ) from exc

        try:
            snapshot_path = snapshot_download(
                repo_id=self._repo_id,
                allow_patterns=["*.json", "*.txt", "onnx/*.onnx"],
            )
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"Could not download the ONNX age model from Hugging Face: {exc}"
            ) from exc

        snapshot_dir = Path(snapshot_path)
        candidate_paths = (
            snapshot_dir / "onnx" / "model.onnx",
            snapshot_dir / "onnx" / "model_int8.onnx",
            snapshot_dir / "onnx" / "model_quantized.onnx",
            snapshot_dir / "onnx" / "model_uint8.onnx",
            snapshot_dir / "onnx" / "model_q4f16.onnx",
            snapshot_dir / "onnx" / "model_q4.onnx",
        )
        model_path = next((path for path in candidate_paths if path.exists()), None)
        if model_path is None:
            raise RuntimeError(
                f"No ONNX model file was found in the snapshot for {self._repo_id}."
            )

        preprocessor_path = snapshot_dir / "preprocessor_config.json"
        if not preprocessor_path.exists():
            raise RuntimeError(
                f"No preprocessor_config.json file was found in the snapshot for {self._repo_id}."
            )

        try:
            preprocessor_config = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"Could not load the preprocessor config from {preprocessor_path}: {exc}"
            ) from exc

        try:
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"Could not load the ONNX runtime session from {model_path}: {exc}"
            ) from exc

        self._preprocessor_config = preprocessor_config
        self._session = session
        self._model_path = model_path
        return self._preprocessor_config, self._session, self._model_path

    def _preprocess_image(self, image: Image.Image, preprocessor_config: dict) -> np.ndarray:
        size_config = preprocessor_config.get("size", {})
        width = int(size_config.get("width", 224))
        height = int(size_config.get("height", 224))

        resized = image.convert("RGB").resize((width, height), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)

        if preprocessor_config.get("do_rescale", True):
            array *= float(preprocessor_config.get("rescale_factor", 1.0 / 255.0))

        if preprocessor_config.get("do_normalize", True):
            mean = np.asarray(preprocessor_config.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std = np.asarray(preprocessor_config.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
            array = (array - mean) / std

        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array.astype(np.float32), axis=0)
        return array

    def estimate(self, image_path: Path, detection: Detection) -> AgeEstimate:
        preprocessor_config, session, model_path = self._load_runtime()

        with crop_detection_to_temp_file(image_path=image_path, detection=detection) as crop_path:
            try:
                cropped_image = Image.open(crop_path).convert("RGB")
                pixel_values = self._preprocess_image(cropped_image, preprocessor_config)
                input_name = session.get_inputs()[0].name
                result = session.run(None, {input_name: pixel_values})
            except Exception as exc:  # pragma: no cover - depends on runtime
                raise RuntimeError(
                    f"ONNX age estimation failed: {exc}"
                ) from exc

        if not result:
            return AgeEstimate(
                age_years=None,
                confidence=None,
                provider="huggingface_onnx_age_gender",
                details=f"No outputs were returned by {model_path}",
            )

        logits = result[0]
        try:
            age_logit = float(logits[0][0])
            gender_score = float(logits[0][1])
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"Unexpected ONNX output format from {model_path}: {exc}"
            ) from exc

        age_value = max(0.0, min(100.0, age_logit))
        confidence = max(0.0, min(1.0, max(gender_score, 1.0 - gender_score)))
        gender_label = "Female" if gender_score >= 0.5 else "Male"

        try:
            age_years = int(round(float(age_value))) if age_value is not None else None
        except (TypeError, ValueError):
            age_years = None

        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None

        return AgeEstimate(
            age_years=age_years,
            confidence=confidence_value,
            provider="huggingface_onnx_age_gender",
            details=f"repo_id={self._repo_id}; model={model_path.name}; gender={gender_label}",
        )
