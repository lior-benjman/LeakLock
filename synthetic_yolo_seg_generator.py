import argparse
import glob
import os
import random

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images and YOLO segmentation labels from object cutouts."
    )
    parser.add_argument("--background-dir", required=True, help="Directory with background images.")
    parser.add_argument(
        "--foreground-dir",
        required=True,
        help="Directory with object cutouts. PNG with alpha works best.",
    )
    parser.add_argument(
        "--mask-dir",
        default="",
        help="Optional directory with binary masks matching foreground file stems.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory. Images go to images/, labels go to labels/.",
    )
    parser.add_argument("--num-images", type=int, default=100, help="Number of synthetic samples.")
    parser.add_argument("--class-id", type=int, default=3, help="YOLO class id to write in labels.")
    parser.add_argument(
        "--min-width-ratio",
        type=float,
        default=0.08,
        help="Minimum object width relative to background width.",
    )
    parser.add_argument(
        "--max-width-ratio",
        type=float,
        default=0.22,
        help="Maximum object width relative to background width.",
    )
    parser.add_argument(
        "--whole-image-rotate",
        type=float,
        default=8.0,
        help="Maximum absolute rotation in degrees applied to the final image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Set to 0 to use a random seed each run.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_image_files(directory):
    files = []
    for name in os.listdir(directory):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            files.append(os.path.join(directory, name))
    files.sort()
    return files


def find_matching_mask(mask_dir, fg_path):
    if not mask_dir:
        return None

    stem = os.path.splitext(os.path.basename(fg_path))[0]
    matches = []
    for ext in IMAGE_EXTENSIONS:
        matches.extend(glob.glob(os.path.join(mask_dir, stem + ext)))
        matches.extend(glob.glob(os.path.join(mask_dir, stem + ext.upper())))
    return matches[0] if matches else None


def load_foreground_rgba(fg_path, mask_dir):
    fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    if fg is None:
        raise ValueError(f"Could not read foreground: {fg_path}")

    if fg.ndim == 2:
        fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGRA)

    if fg.shape[2] == 4:
        rgba = fg.copy()
    else:
        mask_path = find_matching_mask(mask_dir, fg_path)
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")
        else:
            gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        rgba = np.dstack([fg[:, :, :3], mask])

    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"Foreground has an empty mask: {fg_path}")

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return rgba[y1:y2, x1:x2]


def resize_foreground(rgba, bg_w, min_ratio, max_ratio):
    fg_h, fg_w = rgba.shape[:2]
    target_ratio = random.uniform(min_ratio, max_ratio)
    target_w = max(8, int(bg_w * target_ratio))
    scale = target_w / float(fg_w)
    target_h = max(8, int(fg_h * scale))
    return cv2.resize(rgba, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def random_perspective_rgba(rgba, max_margin_ratio=0.18):
    h, w = rgba.shape[:2]
    margin = max(2.0, min(h, w) * max_margin_ratio)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [random.uniform(0, margin), random.uniform(0, margin)],
            [w - 1 - random.uniform(0, margin), random.uniform(0, margin)],
            [w - 1 - random.uniform(0, margin), h - 1 - random.uniform(0, margin)],
            [random.uniform(0, margin), h - 1 - random.uniform(0, margin)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        rgba,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def rotate_rgba_keep_canvas(rgba, angle_degrees):
    h, w = rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(
        rgba,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def adjust_foreground_color(rgba):
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4]
    gain = random.uniform(0.85, 1.15)
    bias = random.uniform(-18.0, 18.0)
    rgb = np.clip(rgb * gain + bias, 0, 255)

    if random.random() < 0.35:
        rgb = cv2.GaussianBlur(rgb.astype(np.uint8), (3, 3), 0).astype(np.float32)

    return np.dstack([rgb.astype(np.uint8), alpha.astype(np.uint8)])


def add_shadow(background, mask, x, y):
    mask_u8 = mask.astype(np.uint8)
    offset_x = random.randint(3, 10)
    offset_y = random.randint(3, 10)
    blur_size = random.choice([11, 15, 19])

    shadow = np.zeros(background.shape[:2], dtype=np.uint8)
    h, w = mask_u8.shape
    y2 = min(background.shape[0], y + h + offset_y)
    x2 = min(background.shape[1], x + w + offset_x)
    src_h = y2 - (y + offset_y)
    src_w = x2 - (x + offset_x)
    if src_h <= 0 or src_w <= 0:
        return background

    shadow[y + offset_y : y2, x + offset_x : x2] = mask_u8[:src_h, :src_w]
    shadow = cv2.GaussianBlur(shadow, (blur_size, blur_size), 0)
    shadow_alpha = (shadow.astype(np.float32) / 255.0) * random.uniform(0.18, 0.35)

    out = background.astype(np.float32)
    for channel in range(3):
        out[:, :, channel] *= (1.0 - shadow_alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def alpha_blend(background, rgba, x, y):
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    alpha_3 = alpha[:, :, None]
    roi = background[y : y + h, x : x + w].astype(np.float32)
    fg_rgb = rgba[:, :, :3].astype(np.float32)
    blended = fg_rgb * alpha_3 + roi * (1.0 - alpha_3)
    background[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return background


def rotate_full_frame(image, mask, max_angle):
    if max_angle <= 0:
        return image, mask

    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_image, rotated_mask


def add_global_augmentations(image, label_mask, max_angle):
    image, label_mask = rotate_full_frame(image, label_mask, max_angle)

    if random.random() < 0.45:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if random.random() < 0.45:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    if random.random() < 0.8:
        noise = np.random.normal(0, random.uniform(5.0, 15.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        speckle = np.random.rand(*image.shape[:2])
        white_prob = random.uniform(0.003, 0.015)
        black_prob = random.uniform(0.002, 0.01)
        image[speckle < black_prob] = 0
        image[speckle > 1.0 - white_prob] = 255

    return image, label_mask


def contour_to_yolo_seg(mask, class_id):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 16:
        return None

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, 0.01 * perimeter)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    if len(polygon) < 3:
        return None

    h, w = mask.shape[:2]
    coords = []
    for point in polygon.reshape(-1, 2):
        x = min(max(float(point[0]), 0.0), float(w - 1))
        y = min(max(float(point[1]), 0.0), float(h - 1))
        coords.append(f"{x / w:.6f}")
        coords.append(f"{y / h:.6f}")

    return f"{class_id} " + " ".join(coords)


def make_sample(background_path, foreground_path, mask_dir, class_id, min_ratio, max_ratio, max_angle):
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"Could not read background: {background_path}")

    fg_rgba = load_foreground_rgba(foreground_path, mask_dir)
    bg_h, bg_w = background.shape[:2]
    fg_rgba = resize_foreground(fg_rgba, bg_w, min_ratio, max_ratio)
    fg_rgba = random_perspective_rgba(fg_rgba)
    fg_rgba = rotate_rgba_keep_canvas(fg_rgba, random.uniform(-25.0, 25.0))
    fg_rgba = adjust_foreground_color(fg_rgba)

    fg_h, fg_w = fg_rgba.shape[:2]
    if fg_h >= bg_h or fg_w >= bg_w:
        return None

    x = random.randint(0, bg_w - fg_w)
    y = random.randint(0, bg_h - fg_h)

    fg_mask = fg_rgba[:, :, 3]
    background = add_shadow(background, fg_mask, x, y)
    composed = alpha_blend(background.copy(), fg_rgba, x, y)

    label_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    label_mask[y : y + fg_h, x : x + fg_w] = np.maximum(
        label_mask[y : y + fg_h, x : x + fg_w], fg_mask
    )
    _, label_mask = cv2.threshold(label_mask, 1, 255, cv2.THRESH_BINARY)

    composed, label_mask = add_global_augmentations(composed, label_mask, max_angle)
    label_line = contour_to_yolo_seg(label_mask, class_id)
    if label_line is None:
        return None

    return composed, label_line


def main():
    args = parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.min_width_ratio <= 0 or args.max_width_ratio <= 0:
        raise ValueError("Width ratios must be greater than zero.")
    if args.min_width_ratio >= args.max_width_ratio:
        raise ValueError("min-width-ratio must be smaller than max-width-ratio.")

    background_files = list_image_files(args.background_dir)
    foreground_files = list_image_files(args.foreground_dir)
    if not background_files:
        raise ValueError(f"No background images found in {args.background_dir}")
    if not foreground_files:
        raise ValueError(f"No foreground images found in {args.foreground_dir}")

    image_dir = os.path.join(args.output_dir, "images")
    label_dir = os.path.join(args.output_dir, "labels")
    ensure_dir(image_dir)
    ensure_dir(label_dir)

    created = 0
    attempts = 0
    max_attempts = max(args.num_images * 20, 100)

    while created < args.num_images and attempts < max_attempts:
        attempts += 1
        background_path = random.choice(background_files)
        foreground_path = random.choice(foreground_files)

        try:
            sample = make_sample(
                background_path=background_path,
                foreground_path=foreground_path,
                mask_dir=args.mask_dir,
                class_id=args.class_id,
                min_ratio=args.min_width_ratio,
                max_ratio=args.max_width_ratio,
                max_angle=args.whole_image_rotate,
            )
        except ValueError as error:
            print(f"Skipping sample: {error}")
            continue

        if sample is None:
            continue

        image, label_line = sample
        name = f"{created:05d}"
        image_path = os.path.join(image_dir, name + ".jpg")
        label_path = os.path.join(label_dir, name + ".txt")

        cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(label_path, "w", encoding="utf-8") as handle:
            handle.write(label_line + "\n")
        created += 1

    print(f"Created {created} synthetic samples in {args.output_dir}")
    if created < args.num_images:
        print("Stopped early because too many attempts produced invalid placements.")


if __name__ == "__main__":
    main()
