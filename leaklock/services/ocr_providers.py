from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..models import OcrExtraction


class OcrProvider(Protocol):
    def extract_text(self, image_path: Path) -> OcrExtraction:
        """Extract text from an image path."""


class TrOcrProvider:
    def __init__(self, model_name: str = "microsoft/trocr-base-printed") -> None:
        self._model_name = model_name
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "transformers and torch are required for TrOCR extraction."
            ) from exc

        try:
            self._torch = torch
            self._processor = TrOCRProcessor.from_pretrained(model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"TrOCR could not initialize from {model_name}: {exc}"
            ) from exc

    def _predict_text(self, image: Image.Image) -> str:
        pixel_values = self._processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        with self._torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text

    def _line_slices(self, image: Image.Image) -> list[tuple[str, Image.Image]]:
        width, height = image.size
        slices: list[tuple[str, Image.Image]] = [("full", image)]
        if height < 120:
            return slices

        bands = 4
        band_height = max(1, height // bands)
        overlap = max(8, band_height // 8)

        for index in range(bands):
            top = max(0, index * band_height - overlap)
            bottom = min(height, (index + 1) * band_height + overlap)
            if bottom - top < 24:
                continue
            slices.append((f"band_{index + 1}", image.crop((0, top, width, bottom))))

        return slices

    def extract_text(self, image_path: Path) -> OcrExtraction:
        try:
            variants = _ocr_image_variants(image_path)
        except Exception as exc:  # pragma: no cover - depends on runtime
            return OcrExtraction(
                text="",
                provider="trocr",
                details=f"TrOCR could not prepare image variants: {exc}",
            )

        collected: list[str] = []
        seen: set[str] = set()
        last_error: str | None = None

        for variant_name, variant_image in variants:
            for slice_name, candidate in self._line_slices(variant_image):
                try:
                    text = self._predict_text(candidate)
                except Exception as exc:  # pragma: no cover - depends on runtime
                    last_error = str(exc)
                    continue

                normalized = text.strip()
                if len(normalized) < 2:
                    continue
                if normalized in seen:
                    continue

                seen.add(normalized)
                collected.append(normalized)

            if collected:
                break

        if collected:
            return OcrExtraction(
                text="\n".join(collected),
                provider="trocr",
                details=f"OCR extracted from cropped detection using model={self._model_name}",
            )

        return OcrExtraction(
            text="",
            provider="trocr",
            details=(
                f"TrOCR ran but found no text. Last error: {last_error}"
                if last_error
                else f"TrOCR ran but found no text using model={self._model_name}"
            ),
        )


class RapidOcrProvider:
    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "rapidocr is required for RapidOCR extraction."
            ) from exc

        try:
            self._engine = RapidOCR()
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"RapidOCR could not initialize: {exc}"
            ) from exc

    def extract_text(self, image_path: Path) -> OcrExtraction:
        try:
            result = self._engine(str(image_path))
        except Exception as exc:  # pragma: no cover - depends on runtime
            return OcrExtraction(
                text="",
                provider="rapidocr",
                details=f"RapidOCR could not run: {exc}",
            )

        if isinstance(result, tuple):
            detections = result[0]
        else:
            detections = result

        text_parts: list[str] = []
        if detections:
            for item in detections:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text_value = item[1]
                    if text_value:
                        text_parts.append(str(text_value).strip())

        return OcrExtraction(
            text="\n".join(part for part in text_parts if part),
            provider="rapidocr",
            details="OCR extracted from cropped detection",
        )


def _pil_resampling_lanczos():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _collect_text_lines(results: list[object]) -> str:
    text_parts: list[str] = []
    for item in results:
        if len(item) >= 2 and item[1]:
            text_parts.append(str(item[1]).strip())
    return "\n".join(part for part in text_parts if part)


def _ocr_image_variants(image_path: Path) -> list[tuple[str, Image.Image]]:
    base = Image.open(image_path).convert("RGB")
    resample = _pil_resampling_lanczos()

    enlarged = base.resize((max(1, base.width * 2), max(1, base.height * 2)), resample)
    gray = ImageOps.grayscale(enlarged)
    autocontrast = ImageOps.autocontrast(gray)
    sharpened = autocontrast.filter(ImageFilter.SHARPEN)

    threshold_source = np.array(sharpened)
    _, threshold = cv2.threshold(
        threshold_source,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold_image = Image.fromarray(threshold)

    return [
        ("original", base),
        ("enlarged", enlarged),
        ("grayscale_autocontrast", autocontrast),
        ("grayscale_sharpened", sharpened),
        ("threshold_otsu", threshold_image),
    ]


class EasyOcrProvider:
    def __init__(self, languages: tuple[str, ...] = ("en",)) -> None:
        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "easyocr is required for EasyOCR extraction."
            ) from exc

        try:
            self._reader = easyocr.Reader(list(languages), gpu=False)
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"EasyOCR could not initialize: {exc}"
            ) from exc

    def extract_text(self, image_path: Path) -> OcrExtraction:
        last_error: str | None = None

        for variant_name, variant_image in _ocr_image_variants(image_path):
            try:
                results = self._reader.readtext(np.array(variant_image), detail=1, paragraph=False)
                text = _collect_text_lines(results).strip()
            except Exception as exc:  # pragma: no cover - depends on runtime
                last_error = str(exc)
                continue

            if text:
                return OcrExtraction(
                    text=text,
                    provider="easyocr",
                    details=f"OCR extracted from cropped detection using variant={variant_name}",
                )

        return OcrExtraction(
            text="",
            provider="easyocr",
            details=(
                f"EasyOCR ran but found no text. Last error: {last_error}"
                if last_error
                else "EasyOCR ran but found no text in any tested image variant"
            ),
        )


class TesseractOcrProvider:
    def __init__(self) -> None:
        try:
            import pytesseract
            from PIL import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "pytesseract and Pillow are required for OCR extraction."
            ) from exc

        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "tesseract is not installed or it's not in your PATH."
            ) from exc

    def extract_text(self, image_path: Path) -> OcrExtraction:
        import pytesseract

        last_error: str | None = None

        for variant_name, variant_image in _ocr_image_variants(image_path):
            for psm in ("6", "11"):
                try:
                    text = pytesseract.image_to_string(
                        variant_image,
                        config=f"--psm {psm}",
                    ).strip()
                except Exception as exc:  # pragma: no cover - depends on runtime
                    last_error = str(exc)
                    continue

                if text:
                    return OcrExtraction(
                        text=text,
                        provider="tesseract",
                        details=f"OCR extracted from cropped detection using variant={variant_name}, psm={psm}",
                    )

        return OcrExtraction(
            text="",
            provider="tesseract",
            details=(
                f"Tesseract ran but found no text. Last error: {last_error}"
                if last_error
                else "Tesseract ran but found no text in any tested image variant"
            ),
        )


class FallbackOcrProvider:
    def __init__(self, providers: list[OcrProvider]) -> None:
        self._providers = providers

    def extract_text(self, image_path: Path) -> OcrExtraction:
        attempts: list[str] = []
        for provider in self._providers:
            extraction = provider.extract_text(image_path)
            if extraction.text.strip():
                return extraction
            attempts.append(f"{extraction.provider}: {extraction.details}")

        return OcrExtraction(
            text="",
            provider="unavailable",
            details=" | ".join(attempts) if attempts else "No OCR providers were available",
        )


class UnavailableOcrProvider:
    def __init__(self, details: str = "OCR provider is not configured yet") -> None:
        self._details = details

    def extract_text(self, image_path: Path) -> OcrExtraction:
        return OcrExtraction(
            text="",
            provider="unavailable",
            details=self._details,
        )
