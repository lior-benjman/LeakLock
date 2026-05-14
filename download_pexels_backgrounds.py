import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


API_URL = "https://api.pexels.com/v1/search"
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a local cache of Pexels backgrounds for synthetic dataset generation."
    )
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Search query. Repeat the flag for multiple queries.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where images and metadata will be stored.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PEXELS_API_KEY", ""),
        help="Pexels API key. Defaults to PEXELS_API_KEY env var.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=40,
        help="Results per request. Pexels currently allows up to 80.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=3,
        help="How many result pages to fetch for each query.",
    )
    parser.add_argument(
        "--orientation",
        choices=["landscape", "portrait", "square"],
        default="landscape",
        help="Preferred photo orientation.",
    )
    parser.add_argument(
        "--size",
        choices=["large", "medium", "small"],
        default="large",
        help="Pexels search size filter.",
    )
    parser.add_argument(
        "--locale",
        default="en-US",
        help="Search locale, for example en-US.",
    )
    parser.add_argument(
        "--image-size",
        choices=["large2x", "large", "medium", "original", "landscape", "portrait"],
        default="large2x",
        help="Which returned image variant to download.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.4,
        help="Delay between API requests to avoid bursts.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=4,
        help="How many times to retry transient image-download failures before skipping a file.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="Base delay between download retries. Later retries back off automatically.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def slugify(value):
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in (" ", "-", "_"):
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "query"


def http_get_json(url, headers):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        return json.loads(data.decode("utf-8")), dict(response.headers.items())


def _is_retryable_download_error(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_HTTP_STATUS_CODES
    return isinstance(error, (urllib.error.URLError, TimeoutError))


def download_file(url, destination, max_attempts=4, retry_delay_seconds=2.0):
    request = urllib.request.Request(url, headers={"User-Agent": "LeakLock synthetic dataset builder"})
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            with open(destination, "wb") as handle:
                handle.write(content)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts or not _is_retryable_download_error(exc):
                raise

            delay_seconds = retry_delay_seconds * attempt
            print(
                f"Retrying download after {exc.__class__.__name__}: {exc}. "
                f"Attempt {attempt}/{max_attempts}, waiting {delay_seconds:.1f}s."
            )
            time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error


def load_existing_manifest(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(item["id"]): item for item in data}


def save_manifest(path, manifest_items):
    items = sorted(manifest_items.values(), key=lambda item: (item["query"], item["id"]))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2, ensure_ascii=True)


def build_search_url(query, page, per_page, orientation, size, locale):
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "orientation": orientation,
        "size": size,
        "locale": locale,
    }
    return API_URL + "?" + urllib.parse.urlencode(params)


def photo_extension(photo, image_url):
    parsed = urllib.parse.urlparse(image_url)
    path = parsed.path.lower()
    if path.endswith(".png"):
        return ".png"
    if path.endswith(".webp"):
        return ".webp"
    if photo.get("src", {}).get("original", "").lower().endswith(".png"):
        return ".png"
    return ".jpg"


def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing Pexels API key. Pass --api-key or set PEXELS_API_KEY.")
    if args.per_page < 1 or args.per_page > 80:
        raise ValueError("--per-page must be between 1 and 80.")
    if args.pages < 1:
        raise ValueError("--pages must be at least 1.")

    ensure_dir(args.output_dir)
    image_dir = os.path.join(args.output_dir, "images")
    ensure_dir(image_dir)

    manifest_path = os.path.join(args.output_dir, "metadata.json")
    manifest = load_existing_manifest(manifest_path)

    headers = {
        "Authorization": args.api_key,
        "User-Agent": "LeakLock synthetic dataset builder",
    }

    total_downloaded = 0
    total_skipped = 0

    for query in args.query:
        query_slug = slugify(query)
        for page in range(1, args.pages + 1):
            url = build_search_url(
                query=query,
                page=page,
                per_page=args.per_page,
                orientation=args.orientation,
                size=args.size,
                locale=args.locale,
            )
            payload, response_headers = http_get_json(url, headers=headers)
            photos = payload.get("photos", [])

            for photo in photos:
                photo_id = str(photo["id"])
                if photo_id in manifest:
                    manifest[photo_id]["queries"] = sorted(
                        set(manifest[photo_id].get("queries", [])) | {query}
                    )
                    continue

                src = photo.get("src", {})
                image_url = src.get(args.image_size) or src.get("large2x") or src.get("large")
                if not image_url:
                    continue

                ext = photo_extension(photo, image_url)
                filename = f"{query_slug}_{photo_id}{ext}"
                image_path = os.path.join(image_dir, filename)
                try:
                    download_file(
                        image_url,
                        image_path,
                        max_attempts=max(1, args.download_retries),
                        retry_delay_seconds=max(0.0, args.retry_delay_seconds),
                    )
                except Exception as exc:
                    total_skipped += 1
                    print(
                        f"Skipping photo {photo_id} for query '{query}' after download failure: "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                    continue

                manifest[photo_id] = {
                    "id": photo["id"],
                    "query": query,
                    "queries": [query],
                    "filename": filename,
                    "pexels_url": photo.get("url", ""),
                    "photographer": photo.get("photographer", ""),
                    "photographer_url": photo.get("photographer_url", ""),
                    "photographer_id": photo.get("photographer_id"),
                    "width": photo.get("width"),
                    "height": photo.get("height"),
                    "avg_color": photo.get("avg_color", ""),
                    "alt": photo.get("alt", ""),
                    "download_variant": args.image_size,
                    "download_url": image_url,
                    "downloaded_at_unix": int(time.time()),
                }
                total_downloaded += 1

            remaining = response_headers.get("X-Ratelimit-Remaining")
            if remaining is not None:
                print(f"Fetched page {page} for '{query}'. Remaining monthly requests: {remaining}")
            else:
                print(f"Fetched page {page} for '{query}'.")

            time.sleep(args.sleep_seconds)

    save_manifest(manifest_path, manifest)
    print(f"Downloaded {total_downloaded} new backgrounds into {args.output_dir}")
    if total_skipped:
        print(f"Skipped {total_skipped} backgrounds due to download failures")
    print(f"Metadata saved to {manifest_path}")


if __name__ == "__main__":
    main()
