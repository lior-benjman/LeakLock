from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

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
