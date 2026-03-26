from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import LeakLockPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LeakLock layered image-risk analysis pipeline."
    )
    parser.add_argument("image_path", help="Path to the uploaded image to analyze.")
    parser.add_argument(
        "--weights",
        default="",
        help="Optional override for the YOLOv8 weights path.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig()
    if args.weights:
        config.yolo_weights_path = Path(args.weights).resolve()

    pipeline = LeakLockPipeline(config=config)
    result = pipeline.analyze_image(args.image_path)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
