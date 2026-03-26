import argparse
import os
import subprocess
import sys


def parse_args():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="One-command pipeline for extract -> background download -> synthetic dataset generation."
    )
    parser.add_argument(
        "--dataset-root",
        default=os.path.join(repo_root, "dataset"),
        help="Dataset root used for object extraction.",
    )
    parser.add_argument(
        "--class-name",
        default="license-plates",
        help="Class name to extract from data.yaml.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Class id to extract. Overrides class-name when provided.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=11000,
        help="How many synthetic images to generate.",
    )
    parser.add_argument(
        "--work-dir",
        default=os.path.join(repo_root, "ofek_data_aug"),
        help="Working directory for extracted objects and cached backgrounds.",
    )
    parser.add_argument( # needs to be named after the class name
        "--output-dir",
        default=os.path.join(repo_root, "synthetic_license-plates_output"),
        help="Final synthetic dataset output directory.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Background search query. Repeat for multiple queries.",
    )
    parser.add_argument(
        "--pexels-api-key",
        default=os.environ.get("PEXELS_API_KEY", ""),
        help="Pexels API key. Defaults to PEXELS_API_KEY env var.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=8,
        help="How many Pexels result pages to fetch per query.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=80,
        help="How many Pexels results per page.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip object extraction and reuse existing cutouts.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Pexels download and reuse existing cached backgrounds.",
    )
    parser.add_argument(
        "--min-width-ratio",
        type=float,
        default=0.08,
        help="Minimum pasted object width relative to background width.",
    )
    parser.add_argument(
        "--max-width-ratio",
        type=float,
        default=0.22,
        help="Maximum pasted object width relative to background width.",
    )
    parser.add_argument(
        "--whole-image-rotate",
        type=float,
        default=8.0,
        help="Maximum full-frame rotation for the synthetic image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the generator. Set 0 for non-deterministic runs.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def default_queries_for_class(class_name):
    normalized = class_name.strip().lower()
    if normalized == "license-plates":
        return ["truck trailer rear", "semi truck back", "trailer rear view"]
    if normalized == "card":
        return ["desk background", "wallet on table", "wooden table top","empty room background no people","empty office wall no people"]
    if normalized == "document":
        return ["office desk", "paper on desk", "workspace table"]
    if normalized == "face":
        return [
            "empty office wall no people",
            "empty room background no people",
            "street background no people",
            "park background no people",
        ]
    return ["neutral background", "indoor background", "outdoor background"]


def safe_name(value):
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in (" ", "-", "_"):
            cleaned.append("-")
    result = "".join(cleaned).strip("-")
    while "--" in result:
        result = result.replace("--", "-")
    return result or "objects"


def run_command(command):
    display_command = list(command)
    for index, item in enumerate(display_command[:-1]):
        if item == "--api-key":
            display_command[index + 1] = "***REDACTED***"

    print("")
    print("Running:")
    print(" ".join(f'"{item}"' if " " in item else item for item in display_command))
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    class_id_to_name = {0: "card", 1: "document", 2: "face", 3: "license-plates"}
    effective_class_name = class_id_to_name.get(args.class_id, args.class_name)

    class_label = f"class-{args.class_id}" if args.class_id is not None else effective_class_name
    class_slug = safe_name(class_label)

    work_dir = os.path.abspath(args.work_dir)
    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(work_dir)
    ensure_dir(output_dir)

    extract_dir = os.path.join(work_dir, f"extracted_{class_slug}")
    backgrounds_dir = os.path.join(work_dir, f"backgrounds_{class_slug}")

    extractor_script = os.path.join(repo_root, "extract_yolo_objects.py")
    downloader_script = os.path.join(repo_root, "download_pexels_backgrounds.py")
    generator_script = os.path.join(repo_root, "synthetic_yolo_seg_generator.py")

    if effective_class_name.lower() == "face":
        print("")
        print(
            "Warning: face compositing can look unrealistic if the backgrounds contain people. "
            "Use neutral no-person backgrounds or pass custom --query values."
        )

    if not args.skip_extract:
        extract_command = [
            sys.executable,
            extractor_script,
            "--dataset-root",
            os.path.abspath(args.dataset_root),
            "--output-dir",
            extract_dir,
        ]
        if args.class_id is not None:
            extract_command.extend(["--class-id", str(args.class_id)])
        else:
            extract_command.extend(["--class-name", args.class_name])
        run_command(extract_command)

    if not args.skip_download:
        api_key = args.pexels_api_key or os.environ.get("PEXELS_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Missing Pexels API key. Pass --pexels-api-key or set PEXELS_API_KEY."
            )

        per_page = min(args.per_page, 80)
        if args.per_page != per_page:
            print("")
            print(f"Adjusting --per-page from {args.per_page} to {per_page} because Pexels allows at most 80.")

        queries = args.query or default_queries_for_class(effective_class_name)
        download_command = [
            sys.executable,
            downloader_script,
            "--output-dir",
            backgrounds_dir,
            "--api-key",
            api_key,
            "--pages",
            str(args.pages),
            "--per-page",
            str(per_page),
            "--orientation",
            "landscape",
            "--size",
            "large",
        ]
        for query in queries:
            download_command.extend(["--query", query])
        run_command(download_command)

    generate_command = [
        sys.executable,
        generator_script,
        "--background-dir",
        os.path.join(backgrounds_dir, "images"),
        "--foreground-dir",
        os.path.join(extract_dir, "cutouts"),
        "--mask-dir",
        os.path.join(extract_dir, "masks"),
        "--output-dir",
        output_dir,
        "--num-images",
        str(args.num_images),
        "--min-width-ratio",
        str(args.min_width_ratio),
        "--max-width-ratio",
        str(args.max_width_ratio),
        "--whole-image-rotate",
        str(args.whole_image_rotate),
        "--seed",
        str(args.seed),
    ]
    if args.class_id is not None:
        generate_command.extend(["--class-id", str(args.class_id)])
    else:
        default_class_ids = {"card": 0, "document": 1, "face": 2, "license-plates": 3}
        class_id = default_class_ids.get(effective_class_name.lower())
        if class_id is None:
            raise ValueError(
                "Unknown class-name to class-id mapping. Pass --class-id explicitly."
            )
        generate_command.extend(["--class-id", str(class_id)])

    run_command(generate_command)

    print("")
    print(f"Synthetic dataset ready in: {output_dir}")
    print(f"Extracted objects cache: {extract_dir}")
    print(f"Background cache: {backgrounds_dir}")


if __name__ == "__main__":
    main()
