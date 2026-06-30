from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

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
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "transformers and torch are required for TrOCR extraction."
            ) from exc

        transformers_logging.set_verbosity_error()

        def _load_from_pretrained(
            processor_cls,
            model_cls,
            *,
            local_files_only: bool,
        ):
            processor = processor_cls.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            model = model_cls.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            return processor, model

        try:
            self._torch = torch
            try:
                # Prefer local cache first so OCR works in restricted/no-network runtimes.
                self._processor, self._model = _load_from_pretrained(
                    TrOCRProcessor,
                    VisionEncoderDecoderModel,
                    local_files_only=True,
                )
            except Exception:
                # Fallback to default HF behavior when internet access is available.
                self._processor, self._model = _load_from_pretrained(
                    TrOCRProcessor,
                    VisionEncoderDecoderModel,
                    local_files_only=False,
                )
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"TrOCR could not initialize from {model_name}: {exc}"
            ) from exc

    def _predict_text(self, image: Any) -> str:
        pixel_values = self._processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        with self._torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=2,
            )
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text

    def _line_slices(self, image: Any) -> list[tuple[str, Any]]:
        width, height = image.size
        slices: list[tuple[str, Any]] = [("full", image)]
        if height < 120:
            return slices

        # Coarse bands preserve broad context.
        bands = 4
        band_height = max(1, height // bands)
        overlap = max(8, band_height // 8)

        for index in range(bands):
            top = max(0, index * band_height - overlap)
            bottom = min(height, (index + 1) * band_height + overlap)
            if bottom - top < 24:
                continue
            slices.append((f"band_{index + 1}", image.crop((0, top, width, bottom))))

        # Fine-grained horizontal windows improve dense-document OCR with TrOCR.
        fine_window = max(48, min(120, height // 10))
        fine_step = max(24, fine_window // 2)
        slice_index = 1
        max_fine_slices = 8
        for top in range(0, max(1, height - fine_window + 1), fine_step):
            if slice_index > max_fine_slices:
                break
            bottom = min(height, top + fine_window)
            if bottom - top < 24:
                continue
            slices.append((f"fine_{slice_index}", image.crop((0, top, width, bottom))))
            slice_index += 1

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

        best_text = ""
        best_variant_name: str | None = None
        best_score = 0
        last_error: str | None = None

        preferred_variants = {"original", "grayscale_autocontrast", "threshold_otsu"}
        variants_for_trocr = [
            (name, image) for name, image in variants if name in preferred_variants
        ]
        if not variants_for_trocr:
            variants_for_trocr = variants

        for variant_name, variant_image in variants_for_trocr:
            collected: list[str] = []
            seen: set[str] = set()
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

            if not collected:
                continue

            variant_text = "\n".join(collected)
            variant_score = _extraction_quality_score(variant_text)
            if variant_score > best_score:
                best_score = variant_score
                best_text = variant_text
                best_variant_name = variant_name

            # High-quality extraction reached; no need to try additional variants.
            if best_score >= GOOD_ENOUGH_SCORE:
                break

        if best_text:
            return OcrExtraction(
                text=best_text,
                provider="trocr",
                details=(
                    "OCR extracted from cropped detection using "
                    f"model={self._model_name}, best_variant={best_variant_name}, "
                    f"quality_score={best_score}"
                ),
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


def _pil_resampling_lanczos(image_module: Any):
    if hasattr(image_module, "Resampling"):
        return image_module.Resampling.LANCZOS
    return image_module.LANCZOS


def _collect_text_lines(results: list[object]) -> str:
    text_parts: list[str] = []
    for item in results:
        if len(item) >= 2 and item[1]:
            text_parts.append(str(item[1]).strip())
    return "\n".join(part for part in text_parts if part)


# A score at or above this bar means the extracted text is substantial
# enough that trying further (more expensive) providers/variants isn't
# worth the cost. Matches the threshold TrOcrProvider already used
# internally to stop trying additional image variants.
GOOD_ENOUGH_SCORE = 220


def _extraction_quality_score(text: str) -> int:
    normalized = " ".join(text.split())
    if not normalized:
        return 0

    alpha_chars = sum(character.isalpha() for character in normalized)
    digit_chars = sum(character.isdigit() for character in normalized)
    tokens = re.findall(r"[A-Za-z0-9]{2,}", normalized)
    unique_token_count = len(set(tokens))
    line_count = len([line for line in text.splitlines() if line.strip()])

    return (alpha_chars * 2) + digit_chars + (unique_token_count * 8) + (line_count * 4)


def _ocr_image_variants(image_path: Path) -> list[tuple[str, Any]]:
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageFilter, ImageOps
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError(
            "Pillow, opencv-python, and numpy are required to prepare OCR image variants."
        ) from exc

    base = Image.open(image_path).convert("RGB")
    resample = _pil_resampling_lanczos(Image)

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
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on runtime
            return OcrExtraction(
                text="",
                provider="easyocr",
                details=f"EasyOCR could not run because numpy is unavailable: {exc}",
            )

        last_error: str | None = None

        try:
            variants = _ocr_image_variants(image_path)
        except Exception as exc:  # pragma: no cover - depends on runtime
            return OcrExtraction(
                text="",
                provider="easyocr",
                details=f"EasyOCR could not prepare image variants: {exc}",
            )

        for variant_name, variant_image in variants:
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

        try:
            variants = _ocr_image_variants(image_path)
        except Exception as exc:  # pragma: no cover - depends on runtime
            return OcrExtraction(
                text="",
                provider="tesseract",
                details=f"Tesseract could not prepare image variants: {exc}",
            )

        for variant_name, variant_image in variants:
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
    """Tries providers in order, stopping as soon as one is good enough.

    Providers should be ordered cheapest/fastest first so the common case
    (a clean image) resolves on the first or second provider instead of
    always paying for every engine in the chain.
    """

    def __init__(self, providers: list[OcrProvider]) -> None:
        self._providers = providers

    def extract_text(self, image_path: Path) -> OcrExtraction:
        best_extraction: OcrExtraction | None = None
        best_score = 0
        attempts: list[str] = []

        for provider in self._providers:
            extraction = provider.extract_text(image_path)
            score = _extraction_quality_score(extraction.text)

            if score > best_score:
                best_score = score
                best_extraction = extraction

            attempts.append(
                f"{extraction.provider}: score={score}; details={extraction.details}"
            )

            if best_score >= GOOD_ENOUGH_SCORE:
                # Good enough — skip the remaining, more expensive providers.
                break

        if best_extraction and best_score > 0:
            return best_extraction

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
