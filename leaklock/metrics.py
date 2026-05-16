from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .config import PipelineConfig
from .models import DetectionAnalysis, ImageAnalysisResult
from .pipeline import LeakLockPipeline


DEFAULT_RISK_TARGETS = {
    "barcode": 40,
    "card": 90,
    "code_screen": 40,
    "document": 90,
    "face": 60,
    "id-card": 90,
    "id_card": 90,
    "id": 90,
    "license-plates": 75,
    "license_plate": 75,
    "passport": 90,
}

DEFAULT_SYNTHETIC_PATTERNS = (
    "synthetic",
    "generated",
    "gemini",
    "random",
)

DOCUMENT_CLASSES = {"card", "document", "id", "id-card", "id_card", "passport"}


@dataclass(slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(slots=True)
class GroundTruthObject:
    class_name: str
    box: Box


@dataclass(slots=True)
class PredictionObject:
    class_name: str
    confidence: float
    box: Box
    risk_percent: int
    route: str
    evidence: dict[str, Any]


@dataclass(slots=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    dataset_split: str
    source_slice: str


@dataclass(slots=True)
class MatchResult:
    prediction: PredictionObject
    ground_truth: GroundTruthObject | None
    iou: float


class SliceAccumulator:
    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.image_count = 0
        self.synthetic_count = 0
        self.real_count = 0
        self.object_count = 0
        self.prediction_count = 0
        self.tp_by_class: dict[str, int] = defaultdict(int)
        self.fp_by_class: dict[str, int] = defaultdict(int)
        self.fn_by_class: dict[str, int] = defaultdict(int)
        self.support_by_class: dict[str, int] = defaultdict(int)
        self.image_risk_errors: list[float] = []
        self.image_risk_squared_errors: list[float] = []
        self.image_risk_band_correct = 0
        self.object_risk_errors: list[float] = []
        self.object_risk_squared_errors: list[float] = []
        self.object_risk_band_correct = 0
        self.object_risk_match_count = 0
        self.gt_high_count = 0
        self.pred_high_count = 0
        self.high_tp_count = 0
        self.unsupported_prediction_count = 0
        self.ocr_route_count = 0
        self.ocr_success_count = 0
        self.face_prediction_count = 0
        self.age_success_count = 0
        self.processing_seconds: list[float] = []

    def add_image(
        self,
        *,
        source_slice: str,
        ground_truth: list[GroundTruthObject],
        predictions: list[PredictionObject],
        matches: list[MatchResult],
        false_negatives: list[GroundTruthObject],
        gt_image_risk: int,
        pred_image_risk: int,
        processing_seconds: float | None,
        risk_targets: dict[str, int],
    ) -> None:
        self.image_count += 1
        if source_slice == "synthetic_only":
            self.synthetic_count += 1
        elif source_slice == "real_only":
            self.real_count += 1

        self.object_count += len(ground_truth)
        self.prediction_count += len(predictions)
        if processing_seconds is not None:
            self.processing_seconds.append(processing_seconds)

        for item in ground_truth:
            self.support_by_class[item.class_name] += 1

        for match in matches:
            prediction = match.prediction
            if match.ground_truth is None:
                self.fp_by_class[prediction.class_name] += 1
                continue

            gt = match.ground_truth
            self.tp_by_class[gt.class_name] += 1
            expected_risk = risk_targets.get(gt.class_name, 0)
            risk_error = prediction.risk_percent - expected_risk
            self.object_risk_errors.append(abs(risk_error))
            self.object_risk_squared_errors.append(risk_error * risk_error)
            self.object_risk_match_count += 1
            if risk_band(prediction.risk_percent) == risk_band(expected_risk):
                self.object_risk_band_correct += 1

        for item in false_negatives:
            self.fn_by_class[item.class_name] += 1

        image_error = pred_image_risk - gt_image_risk
        self.image_risk_errors.append(abs(image_error))
        self.image_risk_squared_errors.append(image_error * image_error)
        if risk_band(pred_image_risk) == risk_band(gt_image_risk):
            self.image_risk_band_correct += 1

        gt_high = gt_image_risk >= 60
        pred_high = pred_image_risk >= 60
        if gt_high:
            self.gt_high_count += 1
        if pred_high:
            self.pred_high_count += 1
        if gt_high and pred_high:
            self.high_tp_count += 1

        for prediction in predictions:
            if prediction.route == "unsupported_layer":
                self.unsupported_prediction_count += 1
            if prediction.route == "ocr_extraction_layer":
                self.ocr_route_count += 1
                ocr_info = prediction.evidence.get("ocr", {}) if isinstance(prediction.evidence, dict) else {}
                if isinstance(ocr_info, dict) and str(ocr_info.get("text", "")).strip():
                    self.ocr_success_count += 1
            if prediction.class_name == "face":
                self.face_prediction_count += 1
                age_info = prediction.evidence.get("age_estimate", {}) if isinstance(prediction.evidence, dict) else {}
                if isinstance(age_info, dict) and age_info.get("age_years") is not None:
                    self.age_success_count += 1

    def summary(self, slice_name: str) -> dict[str, Any]:
        return {
            "slice": slice_name,
            "images": self.image_count,
            "ground_truth_objects": self.object_count,
            "predicted_objects": self.prediction_count,
            "mean_processing_seconds": mean(self.processing_seconds),
            "image_risk_mae": mean(self.image_risk_errors),
            "image_risk_rmse": rmse(self.image_risk_squared_errors),
            "image_risk_band_accuracy": safe_divide(self.image_risk_band_correct, self.image_count),
            "image_high_risk_precision": safe_divide(self.high_tp_count, self.pred_high_count),
            "image_high_risk_recall": safe_divide(self.high_tp_count, self.gt_high_count),
            "object_risk_mae": mean(self.object_risk_errors),
            "object_risk_rmse": rmse(self.object_risk_squared_errors),
            "object_risk_band_accuracy": safe_divide(
                self.object_risk_band_correct,
                self.object_risk_match_count,
            ),
            "unsupported_prediction_rate": safe_divide(
                self.unsupported_prediction_count,
                self.prediction_count,
            ),
            "ocr_success_rate": safe_divide(self.ocr_success_count, self.ocr_route_count),
            "age_success_rate": safe_divide(self.age_success_count, self.face_prediction_count),
        }

    def per_class_rows(self, slice_name: str) -> list[dict[str, Any]]:
        classes = sorted(
            set(self.class_names)
            | set(self.support_by_class)
            | set(self.tp_by_class)
            | set(self.fp_by_class)
            | set(self.fn_by_class)
        )
        rows: list[dict[str, Any]] = []
        for class_name in classes:
            tp = self.tp_by_class[class_name]
            fp = self.fp_by_class[class_name]
            fn = self.fn_by_class[class_name]
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            rows.append(
                {
                    "slice": slice_name,
                    "class": class_name,
                    "support": self.support_by_class[class_name],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": safe_divide(2 * precision * recall, precision + recall),
                }
            )
        return rows


def normalize_class_name(class_name: str) -> str:
    normalized = class_name.strip().lower()
    aliases = {
        "license_plate": "license-plates",
        "license-plate": "license-plates",
        "license plates": "license-plates",
        "id_card": "id-card",
        "id card": "id-card",
    }
    return aliases.get(normalized, normalized)


def risk_band(percent: int | float) -> str:
    if percent >= 60:
        return "high"
    if percent >= 30:
        return "medium"
    return "low"


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def rmse(squared_errors: list[float]) -> float:
    if not squared_errors:
        return 0.0
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def box_iou(a: Box, b: Box) -> float:
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    union = area_a + area_b - intersection
    return safe_divide(intersection, union)


def parse_class_names(data_yaml: Path) -> list[str]:
    text = data_yaml.read_text(encoding="utf-8")
    match = re.search(r"names:\s*(\[.*?\])", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a `names: [...]` list in {data_yaml}")
    names = ast.literal_eval(match.group(1))
    if not isinstance(names, list):
        raise ValueError(f"`names` in {data_yaml} must be a list")
    return [normalize_class_name(str(name)) for name in names]


def parse_yolo_label_file(label_path: Path, class_names: list[str], image_size: tuple[int, int]) -> list[GroundTruthObject]:
    width, height = image_size
    objects: list[GroundTruthObject] = []
    if not label_path.exists():
        return objects

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id >= len(class_names):
            continue
        values = [float(value) for value in parts[1:]]
        box = yolo_values_to_box(values, width, height)
        objects.append(GroundTruthObject(class_name=class_names[class_id], box=box))
    return objects


def yolo_values_to_box(values: list[float], width: int, height: int) -> Box:
    if len(values) == 4:
        x_center, y_center, box_width, box_height = values
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
    else:
        xs = values[0::2]
        ys = values[1::2]
        x1 = min(xs) * width
        y1 = min(ys) * height
        x2 = max(xs) * width
        y2 = max(ys) * height
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def prediction_objects(result: ImageAnalysisResult) -> list[PredictionObject]:
    predictions: list[PredictionObject] = []
    for analysis in result.analyses:
        detection = analysis.detection
        predictions.append(
            PredictionObject(
                class_name=normalize_class_name(detection.class_name),
                confidence=detection.confidence,
                box=Box(
                    x1=detection.box.x1,
                    y1=detection.box.y1,
                    x2=detection.box.x2,
                    y2=detection.box.y2,
                ),
                risk_percent=analysis.risk.risk_percent,
                route=analysis.route,
                evidence=analysis.risk.evidence,
            )
        )
    return predictions


def prediction_objects_from_dict(result: dict[str, Any]) -> list[PredictionObject]:
    predictions: list[PredictionObject] = []
    for analysis in result.get("analyses", []):
        detection = analysis.get("detection", {})
        risk = analysis.get("risk", {})
        box = detection.get("box", {})
        predictions.append(
            PredictionObject(
                class_name=normalize_class_name(str(detection.get("class_name", ""))),
                confidence=float(detection.get("confidence", 0.0)),
                box=Box(
                    x1=float(box.get("x1", 0.0)),
                    y1=float(box.get("y1", 0.0)),
                    x2=float(box.get("x2", 0.0)),
                    y2=float(box.get("y2", 0.0)),
                ),
                risk_percent=int(risk.get("risk_percent", 0)),
                route=str(analysis.get("route", "")),
                evidence=risk.get("evidence", {}) if isinstance(risk.get("evidence", {}), dict) else {},
            )
        )
    return predictions


def match_predictions(
    ground_truth: list[GroundTruthObject],
    predictions: list[PredictionObject],
    iou_threshold: float,
) -> tuple[list[MatchResult], list[GroundTruthObject]]:
    unmatched_gt = set(range(len(ground_truth)))
    matches: list[MatchResult] = []

    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        best_index: int | None = None
        best_iou = 0.0
        for gt_index in unmatched_gt:
            gt = ground_truth[gt_index]
            if gt.class_name != prediction.class_name:
                continue
            current_iou = box_iou(gt.box, prediction.box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = gt_index

        if best_index is not None and best_iou >= iou_threshold:
            unmatched_gt.remove(best_index)
            matches.append(MatchResult(prediction=prediction, ground_truth=ground_truth[best_index], iou=best_iou))
        else:
            matches.append(MatchResult(prediction=prediction, ground_truth=None, iou=0.0))

    false_negatives = [ground_truth[index] for index in sorted(unmatched_gt)]
    return matches, false_negatives


def infer_source_slice(image_path: Path, synthetic_patterns: tuple[str, ...]) -> str:
    lowered = image_path.name.lower()
    if any(pattern.lower() in lowered for pattern in synthetic_patterns):
        return "synthetic_only"
    return "real_only"


def load_source_manifest(manifest_path: Path | None) -> dict[str, str]:
    if manifest_path is None:
        return {}
    manifest: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("image_name") or row.get("image_path") or row.get("path")
            source = row.get("source_slice") or row.get("source_type") or row.get("split")
            if not key or not source:
                continue
            source = source.strip().lower()
            if source in {"synthetic", "synthetic_only"}:
                manifest[Path(key).name] = "synthetic_only"
            elif source in {"real", "real_only"}:
                manifest[Path(key).name] = "real_only"
    return manifest


def collect_records(
    dataset_dir: Path,
    dataset_splits: list[str],
    source_manifest: dict[str, str],
    synthetic_patterns: tuple[str, ...],
    max_images: int | None,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for dataset_split in dataset_splits:
        image_dir = dataset_dir / dataset_split / "images"
        label_dir = dataset_dir / dataset_split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            source_slice = source_manifest.get(image_path.name) or infer_source_slice(image_path, synthetic_patterns)
            records.append(
                ImageRecord(
                    image_path=image_path,
                    label_path=label_path,
                    dataset_split=dataset_split,
                    source_slice=source_slice,
                )
            )
            if max_images is not None and len(records) >= max_images:
                return records
    return records


def parse_risk_targets(overrides: list[str]) -> dict[str, int]:
    targets = {normalize_class_name(key): value for key, value in DEFAULT_RISK_TARGETS.items()}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Risk target override must use class=percent format: {item}")
        class_name, percent_text = item.split("=", 1)
        targets[normalize_class_name(class_name)] = int(percent_text)
    return targets


def gt_image_risk(ground_truth: list[GroundTruthObject], risk_targets: dict[str, int]) -> int:
    return max((risk_targets.get(item.class_name, 0) for item in ground_truth), default=0)


def evaluate_records(
    records: list[ImageRecord],
    class_names: list[str],
    pipeline: LeakLockPipeline,
    risk_targets: dict[str, int],
    iou_threshold: float,
    cache_dir: Path | None,
    force_recompute: bool,
) -> dict[str, SliceAccumulator]:
    accumulators = {
        "synthetic_only": SliceAccumulator(class_names),
        "real_only": SliceAccumulator(class_names),
        "all_images": SliceAccumulator(class_names),
    }

    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record.dataset_split}/{record.image_path.name} ({record.source_slice})", flush=True)
        size = image_size(record.image_path)
        ground_truth = parse_yolo_label_file(record.label_path, class_names, size)
        result_dict, elapsed = run_or_load_prediction(
            pipeline=pipeline,
            record=record,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
        )
        predictions = prediction_objects_from_dict(result_dict)
        matches, false_negatives = match_predictions(ground_truth, predictions, iou_threshold)
        expected_image_risk = gt_image_risk(ground_truth, risk_targets)
        predicted_image_risk = int(result_dict.get("overall_risk_percent", 0))

        for slice_name in (record.source_slice, "all_images"):
            accumulators[slice_name].add_image(
                source_slice=record.source_slice,
                ground_truth=ground_truth,
                predictions=predictions,
                matches=matches,
                false_negatives=false_negatives,
                gt_image_risk=expected_image_risk,
                pred_image_risk=predicted_image_risk,
                processing_seconds=elapsed,
                risk_targets=risk_targets,
            )

    return accumulators


def run_or_load_prediction(
    *,
    pipeline: LeakLockPipeline,
    record: ImageRecord,
    cache_dir: Path | None,
    force_recompute: bool,
) -> tuple[dict[str, Any], float | None]:
    cache_path = prediction_cache_path(cache_dir, record) if cache_dir is not None else None
    if cache_path is not None and cache_path.exists() and not force_recompute:
        return json.loads(cache_path.read_text(encoding="utf-8")), None

    started_at = datetime.now()
    result = pipeline.analyze_image(record.image_path)
    elapsed = (datetime.now() - started_at).total_seconds()
    result_dict = result.to_dict()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")

    return result_dict, elapsed


def prediction_cache_path(cache_dir: Path | None, record: ImageRecord) -> Path | None:
    if cache_dir is None:
        return None
    key_source = f"{record.dataset_split}:{record.image_path.resolve()}"
    key = hashlib.sha1(key_source.encode("utf-8")).hexdigest()[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.image_path.stem)
    return cache_dir / record.dataset_split / f"{safe_stem}_{key}.json"


def write_reports(
    output_dir: Path,
    accumulators: dict[str, SliceAccumulator],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [accumulators[name].summary(name) for name in ("synthetic_only", "real_only", "all_images")]
    per_class_rows: list[dict[str, Any]] = []
    for name in ("synthetic_only", "real_only", "all_images"):
        per_class_rows.extend(accumulators[name].per_class_rows(name))

    (output_dir / "metrics_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary_rows, "per_class": per_class_rows}, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    write_csv(output_dir / "metrics_per_class.csv", per_class_rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LeakLock metrics by synthetic, real, and all image slices.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="YOLO dataset directory.")
    parser.add_argument(
        "--dataset-split",
        action="append",
        default=[],
        help="Dataset split to evaluate. Repeatable. Defaults to test.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results") / "metrics")
    parser.add_argument("--weights", type=Path, default=None, help="Optional YOLO weights override.")
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--source-manifest", type=Path, default=None, help="Optional CSV with image_name/source_type.")
    parser.add_argument(
        "--synthetic-pattern",
        action="append",
        default=[],
        help="Filename substring that marks synthetic images. Repeatable.",
    )
    parser.add_argument(
        "--risk-target",
        action="append",
        default=[],
        help="Override class risk target, e.g. passport=95. Repeatable.",
    )
    parser.add_argument(
        "--enable-document-ml-risk",
        action="store_true",
        help="Enable the slower zero-shot document-risk branch during evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Prediction cache directory. Defaults to OUTPUT_DIR/prediction_cache.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write cached pipeline predictions.")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore existing cached predictions.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    dataset_dir = args.dataset.resolve()
    data_yaml = dataset_dir / "data.yaml"
    class_names = parse_class_names(data_yaml)
    source_manifest = load_source_manifest(args.source_manifest)
    synthetic_patterns = tuple(args.synthetic_pattern) if args.synthetic_pattern else DEFAULT_SYNTHETIC_PATTERNS
    dataset_splits = args.dataset_split or ["test"]
    risk_targets = parse_risk_targets(args.risk_target)

    records = collect_records(
        dataset_dir=dataset_dir,
        dataset_splits=dataset_splits,
        source_manifest=source_manifest,
        synthetic_patterns=synthetic_patterns,
        max_images=args.max_images,
    )
    if not records:
        raise RuntimeError(f"No images were found under {dataset_dir} for splits {dataset_splits}")

    config = PipelineConfig(repo_root=Path(__file__).resolve().parent.parent)
    if args.weights is not None:
        config.yolo_weights_path = args.weights.resolve()
    config.enable_document_ml_risk = bool(args.enable_document_ml_risk)
    pipeline = LeakLockPipeline(config=config)
    cache_dir = None if args.no_cache else (args.cache_dir or args.output_dir / "prediction_cache")

    accumulators = evaluate_records(
        records=records,
        class_names=class_names,
        pipeline=pipeline,
        risk_targets=risk_targets,
        iou_threshold=args.iou_threshold,
        cache_dir=cache_dir,
        force_recompute=bool(args.force_recompute),
    )

    metadata = {
        "dataset": str(dataset_dir),
        "dataset_splits": dataset_splits,
        "images_evaluated": len(records),
        "iou_threshold": args.iou_threshold,
        "risk_targets": risk_targets,
        "synthetic_patterns": synthetic_patterns,
        "source_manifest": str(args.source_manifest) if args.source_manifest else None,
        "weights": str(config.yolo_weights_path),
        "document_ml_risk_enabled": config.enable_document_ml_risk,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "force_recompute": bool(args.force_recompute),
    }
    write_reports(args.output_dir, accumulators, metadata)
    print(f"Wrote metrics to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
