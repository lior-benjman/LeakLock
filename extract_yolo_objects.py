import argparse
import json
import os

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract object cutouts and masks from a YOLO detection/segmentation dataset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing train/valid/test folders and optionally data.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root. Transparent cutouts go to cutouts/, masks to masks/.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Split to process. Repeat for multiple splits. Defaults to train, valid, test when present.",
    )
    parser.add_argument(
        "--class-id",
        action="append",
        type=int,
        default=[],
        help="Class id to extract. Repeat for multiple ids. If omitted, all classes are extracted.",
    )
    parser.add_argument(
        "--class-name",
        action="append",
        default=[],
        help="Class name to extract from data.yaml. Repeat for multiple names.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=64,
        help="Minimum mask area in pixels to keep.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=4,
        help="Extra pixels to keep around the extracted object crop.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_class_names(dataset_root):
    data_yaml = os.path.join(dataset_root, "data.yaml")
    if not os.path.exists(data_yaml):
        return {}

    names = {}
    in_names = False
    next_index = 0
    with open(data_yaml, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("names:"):
                after_colon = line.split(":", 1)[1].strip()
                if after_colon.startswith("[") and after_colon.endswith("]"):
                    values = [item.strip().strip("'\"") for item in after_colon[1:-1].split(",") if item.strip()]
                    return {index: value for index, value in enumerate(values)}
                in_names = True
                next_index = 0
                continue

            if in_names and line.startswith("- "):
                names[next_index] = line[2:].strip().strip("'\"")
                next_index += 1
                continue

            if in_names and ":" in line:
                left, right = line.split(":", 1)
                left = left.strip()
                right = right.strip().strip("'\"")
                if left.isdigit():
                    names[int(left)] = right
                    continue
                in_names = False

    return names


def resolve_target_class_ids(class_names, class_ids, class_name_filters):
    target_ids = set(class_ids)
    if class_name_filters:
        lookup = {name.lower(): class_id for class_id, name in class_names.items()}
        for class_name in class_name_filters:
            key = class_name.lower()
            if key not in lookup:
                raise ValueError(f"Class name '{class_name}' was not found in data.yaml")
            target_ids.add(lookup[key])
    return target_ids


def discover_splits(dataset_root, requested_splits):
    if requested_splits:
        return requested_splits

    found = []
    for split in ("train", "valid", "val", "test"):
        label_dir = os.path.join(dataset_root, split, "labels")
        image_dir = os.path.join(dataset_root, split, "images")
        if os.path.isdir(label_dir) and os.path.isdir(image_dir):
            found.append(split)
    return found


def find_image_for_label(image_dir, label_path):
    stem = os.path.splitext(os.path.basename(label_path))[0]
    for ext in IMAGE_EXTENSIONS:
        image_path = os.path.join(image_dir, stem + ext)
        if os.path.exists(image_path):
            return image_path
    return None


def segmentation_points_to_pixels(coords, width, height):
    points = []
    for index in range(0, len(coords), 2):
        x = min(max(coords[index] * width, 0.0), width - 1.0)
        y = min(max(coords[index + 1] * height, 0.0), height - 1.0)
        points.append([int(round(x)), int(round(y))])
    return np.array(points, dtype=np.int32)


def bbox_to_polygon(coords, width, height):
    cx, cy, bw, bh = coords
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    points = np.array(
        [
            [int(round(max(0.0, min(width - 1.0, x1)))), int(round(max(0.0, min(height - 1.0, y1))))],
            [int(round(max(0.0, min(width - 1.0, x2)))), int(round(max(0.0, min(height - 1.0, y1))))],
            [int(round(max(0.0, min(width - 1.0, x2)))), int(round(max(0.0, min(height - 1.0, y2))))],
            [int(round(max(0.0, min(width - 1.0, x1)))), int(round(max(0.0, min(height - 1.0, y2))))],
        ],
        dtype=np.int32,
    )
    return points


def parse_label_line(line, image_width, image_height):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    coords = [float(value) for value in parts[1:]]

    if len(coords) == 4:
        polygon = bbox_to_polygon(coords, image_width, image_height)
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        polygon = segmentation_points_to_pixels(coords, image_width, image_height)
    else:
        return None

    return class_id, polygon


def crop_from_polygon(image_bgr, polygon, padding, min_area):
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    area = int(cv2.countNonZero(mask))
    if area < min_area:
        return None

    xs = polygon[:, 0]
    ys = polygon[:, 1]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + padding + 1)
    y2 = min(h, int(ys.max()) + padding + 1)

    crop_bgr = image_bgr[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = crop_mask
    return rgba, crop_mask, (x1, y1, x2, y2), area


def save_metadata(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=True)


def main():
    args = parse_args()

    class_names = load_class_names(args.dataset_root)
    target_ids = resolve_target_class_ids(class_names, args.class_id, args.class_name)
    splits = discover_splits(args.dataset_root, args.split)
    if not splits:
        raise ValueError("No dataset splits found. Expected train/valid/test folders with images and labels.")

    cutout_dir = os.path.join(args.output_dir, "cutouts")
    mask_dir = os.path.join(args.output_dir, "masks")
    ensure_dir(cutout_dir)
    ensure_dir(mask_dir)

    metadata = []
    saved = 0

    for split in splits:
        image_dir = os.path.join(args.dataset_root, split, "images")
        label_dir = os.path.join(args.dataset_root, split, "labels")
        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            continue

        for label_name in sorted(os.listdir(label_dir)):
            if not label_name.lower().endswith(".txt"):
                continue

            label_path = os.path.join(label_dir, label_name)
            image_path = find_image_for_label(image_dir, label_path)
            if not image_path:
                continue

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            image_h, image_w = image.shape[:2]
            with open(label_path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()

            object_index = 0
            for line in lines:
                parsed = parse_label_line(line, image_w, image_h)
                if parsed is None:
                    continue

                class_id, polygon = parsed
                if target_ids and class_id not in target_ids:
                    continue

                cropped = crop_from_polygon(image, polygon, args.padding, args.min_area)
                if cropped is None:
                    continue

                rgba, binary_mask, bbox, area = cropped
                class_name = class_names.get(class_id, f"class_{class_id}")
                safe_class_name = class_name.replace(" ", "-")

                base_name = (
                    f"{safe_class_name}_{split}_"
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_"
                    f"{object_index:03d}"
                )
                cutout_path = os.path.join(cutout_dir, base_name + ".png")
                mask_path = os.path.join(mask_dir, base_name + ".png")

                cv2.imwrite(cutout_path, rgba)
                cv2.imwrite(mask_path, binary_mask)

                metadata.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "split": split,
                        "source_image": image_path,
                        "source_label": label_path,
                        "object_index": object_index,
                        "cutout_path": cutout_path,
                        "mask_path": mask_path,
                        "bbox_xyxy": list(bbox),
                        "mask_area_px": area,
                    }
                )
                saved += 1
                object_index += 1

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    save_metadata(metadata_path, metadata)
    print(f"Saved {saved} object cutouts to {args.output_dir}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
