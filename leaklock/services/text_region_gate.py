from __future__ import annotations

from pathlib import Path

from ..config import PipelineConfig
from ..models import BoundingBox, Detection


class TextRegionGate:
    """Finds likely text regions without running OCR.

    This is a cheap gate for text-only secrets. It should produce small crops
    for the existing OCR layer, not read text itself and not return whole-image
    boxes.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def detect(self, image_path: Path) -> list[Detection]:
        if not self._config.enable_text_region_gate:
            return []

        try:
            import cv2
        except ImportError:
            return []

        image = cv2.imread(str(image_path))
        if image is None:
            return []

        original_height, original_width = image.shape[:2]
        if original_width <= 0 or original_height <= 0:
            return []

        max_side = max(1, self._config.text_region_max_image_side)
        scale = min(max_side / float(max(original_width, original_height)), 1.0)
        if scale < 1.0:
            small = cv2.resize(
                image,
                (int(original_width * scale), int(original_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image

        small_height, small_width = small.shape[:2]
        if small_width <= 0 or small_height <= 0:
            return []

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        dark_text_mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )
        light_text_mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            -10,
        )
        image_area = float(small_width * small_height)
        min_area = image_area * self._config.text_region_min_area_ratio
        max_area = image_area * self._config.text_region_max_area_ratio
        min_width = max(20, int(small_width * self._config.text_region_min_width_ratio))
        candidates = _candidate_boxes_from_mask(
            cv2=cv2,
            mask=dark_text_mask,
            small_width=small_width,
            small_height=small_height,
            image_area=image_area,
            min_area=min_area,
            max_area=max_area,
            min_width=min_width,
            min_aspect_ratio=self._config.text_region_min_aspect_ratio,
        )
        candidates.extend(
            _candidate_boxes_from_mask(
                cv2=cv2,
                mask=light_text_mask,
                small_width=small_width,
                small_height=small_height,
                image_area=image_area,
                min_area=min_area,
                max_area=max_area,
                min_width=min_width,
                min_aspect_ratio=self._config.text_region_min_aspect_ratio,
            )
        )

        if not candidates:
            return []

        padded = [
            _pad_box(
                box,
                small_width,
                small_height,
                max(4, int(small_width * self._config.text_region_padding_ratio)),
                max(4, int(small_height * self._config.text_region_padding_ratio)),
            )
            for box in candidates
        ]
        merged = _merge_boxes(padded, small_width, small_height)
        merged = [
            box
            for box in merged
            if _merged_box_is_usable(
                box,
                image_area,
                min_area,
                max_area,
                min_width,
                self._config.text_region_min_aspect_ratio,
            )
        ]
        merged.sort(key=lambda item: _box_area(item), reverse=True)

        detections: list[Detection] = []
        inverse_scale = 1.0 / scale
        for index, box in enumerate(merged[: self._config.text_region_max_regions]):
            x1, y1, x2, y2, density = box
            detections.append(
                Detection(
                    class_id=-(index + 2),
                    class_name=self._config.text_region_class_name,
                    confidence=max(0.35, min(0.95, 0.45 + density)),
                    box=BoundingBox(
                        x1=max(0.0, x1 * inverse_scale),
                        y1=max(0.0, y1 * inverse_scale),
                        x2=min(float(original_width), x2 * inverse_scale),
                        y2=min(float(original_height), y2 * inverse_scale),
                    ),
                )
            )

        return detections


def _pad_box(
    box: tuple[int, int, int, int, float],
    image_width: int,
    image_height: int,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int, float]:
    x1, y1, x2, y2, density = box
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
        density,
    )


def _merge_boxes(
    boxes: list[tuple[int, int, int, int, float]],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int, float]]:
    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        next_boxes: list[tuple[int, int, int, int, float]] = []

        while merged:
            current = merged.pop()
            match_index = None
            for index, other in enumerate(merged):
                if _boxes_should_merge(current, other):
                    match_index = index
                    break

            if match_index is None:
                next_boxes.append(current)
                continue

            other = merged.pop(match_index)
            current_area = _box_area(current)
            other_area = _box_area(other)
            density = (
                (current[4] * current_area) + (other[4] * other_area)
            ) / max(current_area + other_area, 1.0)
            merged.append(
                (
                    max(0, min(current[0], other[0])),
                    max(0, min(current[1], other[1])),
                    min(image_width, max(current[2], other[2])),
                    min(image_height, max(current[3], other[3])),
                    density,
                )
            )
            changed = True

        merged = next_boxes

    return merged


def _candidate_boxes_from_mask(
    *,
    cv2,
    mask,
    small_width: int,
    small_height: int,
    image_area: float,
    min_area: float,
    max_area: float,
    min_width: int,
    min_aspect_ratio: float,
) -> list[tuple[int, int, int, int, float]]:
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(12, small_width // 45),
            max(3, small_height // 180),
        ),
    )
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    connected = cv2.morphologyEx(
        connected,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(width * height)
        if not _region_shape_is_usable(
            width=width,
            height=height,
            area=area,
            image_area=image_area,
            min_area=min_area,
            max_area=max_area,
            min_width=min_width,
            min_aspect_ratio=min_aspect_ratio,
        ):
            continue

        crop_mask = mask[y : y + height, x : x + width]
        density = cv2.countNonZero(crop_mask) / max(area, 1.0)
        if density < 0.035 or density > 0.65:
            continue

        candidates.append((x, y, x + width, y + height, density))

    return candidates


def _merged_box_is_usable(
    box: tuple[int, int, int, int, float],
    image_area: float,
    min_area: float,
    max_area: float,
    min_width: int,
    min_aspect_ratio: float,
) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    area = _box_area(box)
    return _region_shape_is_usable(
        width=width,
        height=height,
        area=area,
        image_area=image_area,
        min_area=min_area,
        max_area=max_area,
        min_width=min_width,
        min_aspect_ratio=min_aspect_ratio,
    )


def _region_shape_is_usable(
    *,
    width: int,
    height: int,
    area: float,
    image_area: float,
    min_area: float,
    max_area: float,
    min_width: int,
    min_aspect_ratio: float,
) -> bool:
    if area < min_area or area > max_area:
        return False
    if width < min_width or height < 8:
        return False

    aspect_ratio = width / max(float(height), 1.0)
    if aspect_ratio < 0.5:
        return False
    if aspect_ratio < min_aspect_ratio and area < image_area * 0.02:
        return False

    return True


def _boxes_should_merge(
    first: tuple[int, int, int, int, float],
    second: tuple[int, int, int, int, float],
) -> bool:
    x_gap = max(0, max(first[0], second[0]) - min(first[2], second[2]))
    y_gap = max(0, max(first[1], second[1]) - min(first[3], second[3]))
    min_height = max(1, min(first[3] - first[1], second[3] - second[1]))
    return x_gap <= min_height * 2 and y_gap <= max(4, min_height * 0.3)


def _box_area(box: tuple[int, int, int, int, float]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))
