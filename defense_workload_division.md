# LeakLock — Final Project Defense: Workload Division

> **Format:** 1-hour defense (≈ 10 min per member presentation + 20 min shared Q&A)
>
> Each member should master their assigned files, understand how their domain connects to the others, and prepare for cross-cutting questions.

---

## Project Summary (everyone should know this)

LeakLock is a **privacy-leak detection system** for images. A custom-trained YOLOv8 model detects sensitive content (faces, ID documents, license plates, cards) in an uploaded image. Detected objects are routed through specialized risk-analysis layers — face-age estimation, OCR text extraction + risk scoring, and license-plate flagging — producing a **0–100 risk score** with human-readable explanations. The system is exposed via a **FastAPI backend deployed on Google Cloud Run** and consumed by a **Chrome Manifest V3 extension** that intercepts image uploads on any website.

---

## Workload Division

| Member   | Assigned Topic                                        | Presentation Focus                                         |
|----------|------------------------------------------------------|------------------------------------------------------------|
| Member 1 | YOLO Model Training & Synthetic Data Pipeline        | How the training data was built and the model was trained   |
| Member 2 | Layered Risk-Analysis Pipeline (Core Engine)         | How detections are routed, scored, and explained            |
| Member 3 | Chrome Extension & User-Facing Experience            | How uploads are intercepted, analyzed, and redacted in-browser |
| Member 4 | API, Cloud Deployment, Docker & Testing              | How the backend serves requests, deploys, and is tested     |

---

## Member 1 — YOLO Model Training & Synthetic Data Pipeline

### Assigned Topic
The **data collection, synthetic data generation, data augmentation, YOLO model training, and model quality validation** workflow that produces the `best.pt` weights used by the rest of the system.

### Key Responsibilities
| Area | What to Explain |
|------|----------------|
| **Dataset composition** | ~2,000 YOLO-annotated images across 5 classes (QR/barcode, license plates, ID cards, credit cards, passports). Pilot on 200 images, then full dataset training. |
| **Object extraction** | How `extract_yolo_objects.py` reads YOLO polygon/box annotations, crops objects with alpha masks, and outputs transparent RGBA PNGs + binary masks for compositing. |
| **Background acquisition** | How `download_pexels_backgrounds.py` uses the Pexels REST API to download category-specific backgrounds (office, desk, street, etc.) with rate-limiting and deduplication. |
| **Synthetic data generation** | How `synthetic_yolo_seg_generator.py` composites RGBA cutouts onto random backgrounds with perspective warp (`cv2.warpPerspective`), rotation (`cv2.warpAffine`), drop shadows, color/lighting adjustments, and noise augmentation. Generates YOLO segmentation polygon labels. |
| **End-to-end automation** | How `data-aug.py` orchestrates the full extraction → download → synthesis pipeline via CLI, with class-specific query mappings (e.g., `license-plates` → truck trailers). |
| **Quality validation** | How `real_vs_synthetic_classifier.py` trains a `ShallowCardClassifier` (single-layer CNN with Gaussian noise + spatial dropout) to verify synthetic images are perceptually indistinguishable from real ones. Metrics: balanced accuracy, ROC-AUC, Macro F1, MCC. |
| **Training runs** | The `runs_sensitive/` directory structure (`yolov8`, `yolov82`, `yolov86`, etc.), iterative training progression, and why `yolov86/weights/best.pt` is the deployed model. |
| **YOLO specifics** | 4 deployed classes: `card`, `document`, `face`, `license-plates`. Confidence threshold 0.25. Note: original dataset had finer categories (credit card, passport, ID card) but the deployed model collapses them into `card`/`document` — downstream OCR disambiguates. |

### Files to Master
| File / Directory | Purpose |
|-----------------|---------|
| `data-aug.py` | Master pipeline automation CLI |
| `download_pexels_backgrounds.py` | Pexels API background downloader |
| `extract_yolo_objects.py` | Object cutout and mask extractor |
| `synthetic_yolo_seg_generator.py` | Compositing engine with transforms |
| `real_vs_synthetic_classifier.py` | Quality validation CNN classifier |
| `data_preprocess.ipynb` | Data preprocessing notebook |
| `runs_sensitive/` | All training run outputs and weights |
| `dataset/` | Training dataset |
| `README.md` | Dataset description and class counts |

### Likely Q&A Questions to Prepare For
1. Why use synthetic data instead of only real annotated images?
2. How do you ensure synthetic images are realistic enough for training? (→ real vs. synthetic classifier)
3. What augmentations were applied and why? (perspective warp, shadows, noise)
4. Why does the deployed model have only 4 classes when the dataset has more categories?
5. How many training iterations/experiments were needed? What improved between `yolov8` and `yolov86`?
6. What is the model's mAP / precision / recall on the validation set?

---

## Member 2 — Layered Risk-Analysis Pipeline (Core Engine)

### Assigned Topic
The **`leaklock/` Python package architecture**: the pipeline orchestrator, routing layer, all specialized risk layers (face-age, OCR, license plate), the service adapters (age estimators, OCR providers, text classifiers), and the configuration/data model system.

### Key Responsibilities
| Area | What to Explain |
|------|----------------|
| **Pipeline orchestrator** | `LeakLockPipeline.analyze_image()` flow: YOLO detect → route → layer-specific risk evaluation → text-region gate → aggregate risk via `max()`. |
| **Data models** | `BoundingBox`, `Detection`, `AgeEstimate`, `OcrExtraction`, `RiskResult`, `DetectionAnalysis`, `ImageAnalysisResult` — how data flows through the system. |
| **Detection layer** | `YoloDetectionLayer` wrapping Ultralytics YOLO: lazy model loading, confidence threshold, normalization of raw boxes into `Detection` objects. |
| **Routing** | `DetectionRouter` maps class names → processing layers: `face` → `face_age_layer`, `card/document/id/passport` → `ocr_extraction_layer`, `license-plates` → `license_plate_risk_layer`, unknown → `unsupported_layer`. |
| **Face-age risk** | `FaceAgeRiskLayer` + `FallbackAgeEstimator` with 4-tier fallback chain: ONNX HF age-gender → ViT classifier → DeepFace → Unavailable. Configurable `AgeRiskBand`s: ≤11 → 60%, 12–15 → 30%, 16+ → 20%. |
| **OCR extraction** | `OcrExtractionLayer` with tiered OCR: RapidOCR (fast/eager) → EasyOCR (lazy) → Tesseract (lazy) → TrOCR (lazy). `GOOD_ENOUGH_SCORE` stops escalation. Fast variant for real-time mode. |
| **OCR risk scoring** | `OcrRiskEvaluationLayer`: 60% rule-based (keyword/regex: credentials, PII, financial data, medical terms) + 40% ML (`facebook/bart-large-mnli` zero-shot document classification). SpaCy NER for entity extraction. Blended risk with credential override. |
| **Text-region gate** | `TextRegionGate`: OpenCV adaptive thresholding (dark-on-light + light-on-dark), morphological close/open, contour filtering by aspect ratio/density/area to find text blocks that YOLO missed (e.g., sticky notes, handwritten credentials). |
| **Configuration** | `PipelineConfig` dataclass: feature flags (`enable_slow_ocr_fallbacks`, `enable_document_ml_risk`, `enable_text_region_gate`), all tunable thresholds and weights. |
| **Risk levels** | 0–33 LOW (green), 34–66 MEDIUM (orange), 67–100 HIGH (red). Overall = `max()` across all detections. |
| **Warm-up** | Pre-loading models at startup (`warm_up()`) to avoid cold-start latency. |

### Files to Master
| File / Directory | Purpose |
|-----------------|---------|
| `leaklock/pipeline.py` | Core orchestrator |
| `leaklock/config.py` | All configuration and thresholds |
| `leaklock/models.py` | Domain data models |
| `leaklock/layers/detection.py` | YOLO detection wrapper |
| `leaklock/layers/routing.py` | Class → layer routing |
| `leaklock/layers/face_age.py` | Face-age risk evaluation |
| `leaklock/layers/ocr.py` | Tiered OCR extraction |
| `leaklock/layers/ocr_risk.py` | Rule + ML text risk scoring |
| `leaklock/layers/license_plate.py` | Fixed 75% license plate risk |
| `leaklock/layers/unsupported.py` | Fallback 0% risk |
| `leaklock/services/age_estimators.py` | 4-tier age estimation chain |
| `leaklock/services/ocr_providers.py` | 4-tier OCR provider chain |
| `leaklock/services/text_classifiers.py` | Rule-based keyword classifier |
| `leaklock/services/text_region_gate.py` | OpenCV text region finder |
| `leaklock/services/reason_generators.py` | Human-readable reason templates |
| `leaklock/services/image_tools.py` | Bounding box cropping utility |
| `docs/layered-risk-pipeline-plan.md` | Original architecture design document |

### Likely Q&A Questions to Prepare For
1. Why use `max()` aggregation instead of additive risk? What are the trade-offs?
2. Why have a 4-tier OCR fallback? Isn't one OCR engine enough?
3. How does the zero-shot BART classifier improve over rule-based scoring alone?
4. What happens when age estimation fails on all providers? (→ 20% baseline risk)
5. How does the text-region gate find text that YOLO missed?
6. Why is the 16+ age band assigned 20% risk instead of 0%? (→ recent bug fix)
7. Could an adversary craft an image that bypasses the pipeline?

---

## Member 3 — Chrome Extension & User-Facing Experience

### Assigned Topic
The **Chrome Manifest V3 extension** (upload interception, risk overlay, in-browser redaction/blurring), the **standalone web UI** (`web_app.py`), the **Jupyter notebook interface**, and the overall **user-facing experience** across all frontends.

### Key Responsibilities
| Area | What to Explain |
|------|----------------|
| **Extension architecture** | Manifest V3: `background.js` (service worker for state init), `content.js` (injected on all URLs), `popup.html/css/js` (toggle UI). Permissions: `storage`, `tabs`, `<all_urls>`. |
| **Upload interception** | Capture-phase `change` listener on `<input type="file">` + `MutationObserver` for dynamically created inputs. `event.stopImmediatePropagation()` blocks the host page (e.g., Gmail) from reading the file before LeakLock can analyze it. |
| **API communication** | Extension → Cloud Run (`https://leaklock-api-799658247857.us-central1.run.app/analyze-image`) via `fetch()` with 45s timeout. Fail-open: if API is unreachable, user can proceed. |
| **Risk overlay UI** | Shadow DOM (`#leaklock-overlay-host`) renders an isolated modal with risk meter, detection breakdown, explanations, and action buttons. Handles single-image and batch (up to 10 files). |
| **In-browser redaction** | Client-side Canvas pixelation: `blurImage()`, `pixelateRegion()`, `collectRiskyBoxes()` — redacts sensitive bounding boxes locally without re-uploading. User can toggle blur, review redacted preview. |
| **User actions** | "Upload Anyway" / "Cancel Upload" / "Continue Upload". On proceed: `DataTransfer`-constructed `FileList` re-dispatches `change` event so the host page processes the file normally. |
| **Recent bug fixes** | (1) Cancel didn't actually block Gmail uploads — fixed with capture-phase listener + `stopImmediatePropagation()`. (2) Adult faces returned 0% risk and were invisible — fixed with 20% baseline. |
| **Standalone web UI** | `web_app.py`: Python `ThreadingHTTPServer` on port 8765 serving embedded HTML with image preview, SHA-1 deduplication, analysis display, and result export (JSON + OCR text + summary CSV). |
| **Jupyter notebook** | `leaklock_upload_pipeline.ipynb`: `ipywidgets` GUI running the full pipeline in-kernel with risk cards, detection details, and automatic result export. |
| **Known limitations** | Only intercepts `<input type="file">` — misses drag-and-drop and clipboard-paste. Only analyzes `files[0]` in multi-select. Filters by MIME type only. |

### Files to Master
| File / Directory | Purpose |
|-----------------|---------|
| `extension/manifest.json` | Extension configuration and permissions |
| `extension/content.js` | Core 800-line content script (interception, overlay, redaction) |
| `extension/background.js` | Service worker for state initialization |
| `extension/popup.html` | Extension popup markup |
| `extension/popup.css` | Popup styling |
| `extension/popup.js` | Toggle logic and tab messaging |
| `extension/icons/` | Extension icon assets |
| `leaklock/web_app.py` | Standalone local web UI and HTTP server |
| `notebooks/leaklock_upload_pipeline.ipynb` | Interactive Jupyter analysis UI |
| `leaklock/cli.py` | CLI interface for single-image analysis |

### Likely Q&A Questions to Prepare For
1. Why use Shadow DOM for the overlay? (→ CSS isolation from host page)
2. How do you prevent Gmail/other apps from processing the file before LeakLock? (→ capture phase + stopImmediatePropagation)
3. Why does the extension fail-open instead of fail-closed?
4. How does in-browser blurring work without sending the image back to the server?
5. What upload methods are NOT covered? (→ drag-and-drop, paste, multi-file)
6. How is state synchronized between popup and content script? (→ `chrome.storage.local` + `chrome.tabs.sendMessage`)
7. How do you handle the extension on sites with restrictive CSP?

---

## Member 4 — API, Cloud Deployment, Docker & Testing

### Assigned Topic
The **FastAPI backend** (`api.py`), **Docker containerization**, **GitHub Actions CI/CD pipeline to Google Cloud Run**, **testing strategy**, and **model evaluation/metrics**.

### Key Responsibilities
| Area | What to Explain |
|------|----------------|
| **FastAPI backend** | `api.py`: `POST /analyze-image` (multipart upload → pipeline → JSON response), `GET /health` (service status, model metadata, runtime deps). CORS `allow_origins=["*"]`. |
| **Thread safety** | `_pipeline_lock` serializes access to non-thread-safe model objects. `asyncio.wait_for()` + `run_in_executor()` keeps the event loop unblocked with a 60s timeout. |
| **Realtime mode** | `LEAKLOCK_REALTIME_MODE=1` env var disables slow OCR fallbacks and ML document risk for faster browser-extension response times. |
| **Startup warm-up** | `lifespan()` context manager pre-loads YOLO, age estimator, and BART classifier off the event loop. Skippable via `LEAKLOCK_SKIP_STARTUP_WARMUP`. |
| **Explanation builder** | `_build_explanations()` converts raw `DetectionAnalysis` objects into user-friendly strings (face detected, sensitive OCR keywords, financial info, etc.). |
| **Dockerfile** | `python:3.10-slim` base, OpenCV system deps (`libgl1`, `libglib2.0-0`), copies only `leaklock/` package + `runs_sensitive/yolov86/weights/best.pt`. CMD runs uvicorn on `$PORT`. |
| **`.dockerignore`** | Excludes 61 GB training dataset, notebooks, all weights except deployed `best.pt`, notebooks, `.ipynb`, `.crx`, `.pem`. |
| **GitHub Actions** | `deploy-cloud-run.yml`: triggers on push to `main` or manual dispatch. Authenticates via **Workload Identity Federation** (no stored secrets). Deploys to Cloud Run with `--memory=8Gi --cpu=2 --min-instances=1 --allow-unauthenticated`. Smoke test via `curl /health`. Concurrency group cancels in-progress deploys. |
| **GCP architecture** | Project `finalproj-orianaziz`, region `us-central1`, service `leaklock-api`. Workload Identity Pool `github-actions/providers/leaklock`. Deploy SA vs Runtime SA separation. |
| **Testing** | `tests/test_text_region_gate.py`: tests text-region discovery, OCR risk pipeline integration, and `RuleBasedSensitiveTextClassifier` for Wi-Fi/password detection. Uses synthetic rendered images. |
| **Metrics & evaluation** | `leaklock/metrics.py`: batch evaluation against ground-truth YOLO annotations. IoU matching, per-class precision/recall/F1, MAE/RMSE on risk scores, risk-band accuracy, OCR success rate, age estimation rate. Separate `synthetic_only` / `real_only` / `all_images` report slices. JSON prediction caching. |
| **Result storage** | `analysis_results/`: JSON results, OCR text exports, `summary.csv`, metrics reports. |

### Files to Master
| File / Directory | Purpose |
|-----------------|---------|
| `leaklock/api.py` | FastAPI backend (285 lines) |
| `.github/workflows/deploy-cloud-run.yml` | CI/CD pipeline to Cloud Run |
| `Dockerfile` | Container definition |
| `.dockerignore` | Build exclusions |
| `requirements.txt` | Python dependency manifest |
| `tests/test_text_region_gate.py` | Unit and integration tests |
| `leaklock/metrics.py` | Model evaluation and benchmarking (708 lines) |
| `analysis_results/` | Result storage (JSON, CSV, OCR text, metrics) |
| `CLAUDE.md` | Full project documentation (known limitations, recent fixes) |

### Likely Q&A Questions to Prepare For
1. Why use Workload Identity Federation instead of a service account key? (→ keyless, more secure)
2. Why `--min-instances=1`? (→ avoids cold-start latency for YOLO/HF model loading)
3. How is thread safety handled for concurrent requests? (→ `_pipeline_lock` + `run_in_executor`)
4. What happens if the API times out? (→ HTTP 504 with descriptive error)
5. Why pin `transformers==4.51.3` and `huggingface_hub==0.30.2`? (→ ONNX/HF age estimator compatibility)
6. What metrics validate that the system actually works? (→ per-class P/R/F1, risk MAE/RMSE)
7. What are the known limitations of the current system? (→ synchronous handler, no caching, no request timeout on OCR)

---

## Cross-Cutting Topics (everyone should be ready)

These questions can be directed at **any** team member:

| Topic | Key Points |
|-------|-----------|
| **End-to-end data flow** | User selects file → extension intercepts → API receives → YOLO detects → router dispatches → layers score risk → API returns → extension shows overlay → user decides |
| **Why YOLO?** | Real-time single-pass detection, good for edge/server deployment, well-supported ecosystem (Ultralytics) |
| **Privacy by design** | Analysis happens server-side but redaction can happen client-side (Canvas). Images are ephemeral (temp files deleted after analysis). |
| **Risk score philosophy** | `max()` aggregation — one high-risk detection flags the entire image. Conservative by design. |
| **Fallback resilience** | Every ML component has a fallback chain: OCR (4 providers), age estimation (4 estimators), text classification (rule-based if ML fails). Extension fails-open if API unreachable. |
| **Known limitations** | Only intercepts `<input type="file">`, OCR fallback chain is slow, no request caching, risk is `max()` not additive, only 4 YOLO classes |
