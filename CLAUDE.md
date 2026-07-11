# LeakLock

Privacy-leak detection for images. A YOLO-based pipeline detects sensitive content
(faces, ID documents, license plates) in an image, runs OCR/age-estimation on what
it finds, and produces a 0-100 risk score. Exposed via a FastAPI backend and consumed
by a Chrome extension that intercepts image uploads on any website.

## Architecture

```
extension/ (MV3 Chrome extension)
  content.js    — intercepts <input type="file"> changes, shows risk overlay, blocks/resumes upload
  popup.js/html — ON/OFF toggle, synced via chrome.storage.local + chrome.tabs.sendMessage
  background.js — service worker, initializes default storage state

leaklock/
  api.py              — FastAPI app: POST /analyze-image, GET /health. CORS open (allow_origins=["*"])
  pipeline.py          — LeakLockPipeline.analyze_image(path): YOLO detect → route → risk-evaluate
  config.py             — PipelineConfig: thresholds, age risk bands, OCR keyword weights
  layers/
    detection.py        — YoloDetectionLayer, runs ultralytics YOLO, conf threshold 0.25
    routing.py           — DetectionRouter: face → face_age_layer, card/document/id/passport → ocr_extraction_layer,
                            license-plates → license_plate_risk_layer
    face_age.py          — FaceAgeRiskLayer: age estimate → risk band (config.age_risk_bands)
    ocr.py                — OcrExtractionLayer: crops detection box, runs OCR fallback chain
    ocr_risk.py           — OcrRiskEvaluationLayer: rule-based keywords (60%) blended with zero-shot
                            document classification via facebook/bart-large-mnli (40%)
    license_plate.py     — fixed risk_percent from config (75%)
  services/
    age_estimators.py    — fallback chain: ONNX (onnx-community/age-gender-prediction-ONNX) →
                            HF Transformers (nateraw/vit-age-classifier) → DeepFace → Unavailable
    ocr_providers.py     — fallback chain: TrOCR → RapidOCR → EasyOCR → Tesseract
    image_tools.py        — crop_detection_to_temp_file: crops to bounding box, saves as PNG temp file
```

**YOLO model in use**: `runs_sensitive/yolov86/weights/best.pt`, 4 classes only —
`card`, `document`, `face`, `license-plates` (see `sync_eval.yaml`/`real_eval.yaml`).
Note: README mentions training data included QR/barcode, credit card, and passport as
separate categories — if those were meant to be distinct YOLO classes, the currently
deployed weights don't reflect that; everything funnels through `card`/`document` and
gets disambiguated downstream by OCR text content instead.

**Risk levels**: 0-33 LOW (green) · 34-66 MEDIUM (orange) · 67-100 HIGH (red).
Overall score = `max()` across all detections' individual risk_percent (not additive).

## Running locally

```bash
pip install -r requirements.txt
python -m uvicorn leaklock.api:app --port 8000   # NOT plain `uvicorn` — not on PATH here
```

Load `extension/` as an unpacked extension via `chrome://extensions` → Developer mode →
Load unpacked. Full walkthrough in `extension/INSTRUCTIONS.md`.

Branch: `api_extension` (not merged to `main`). Remote: `https://github.com/lior-benjman/LeakLock`.

## Recent fixes (2026-06)

1. **Adult faces returned 0% risk and were invisible to the user.**
   `config.py`'s age band for 16+ was `risk_percent=0`, and `api.py`'s
   `_build_explanations` skips any analysis with `risk_percent == 0`. So an image
   containing only an adult face came back as `risk_score=0, explanations=[]` even
   though YOLO detected the face. Same zero-risk fallback applied when age estimation
   failed outright (`face_age.py`). Fixed by giving adult faces and estimation-failure
   paths a baseline `risk_percent=20` instead of 0, so they now surface as LOW with a
   "Face detected" explanation rather than being silently dropped. Still a product
   knob — bump above 33 if adult faces should force a MEDIUM-risk Cancel/Upload-Anyway
   choice instead of a single Continue button.

2. **"Cancel Upload" didn't actually block the upload (e.g. on Gmail).**
   `content.js` listened for `change` in the default bubble phase, so Gmail's own
   handler ran first and read/stored the file before LeakLock could intervene.
   Clearing `input.value` afterward had no effect on Gmail's already-captured state.
   Fixed by registering the listener in the **capture phase** and calling
   `event.stopImmediatePropagation()` immediately, fully blocking the host page from
   seeing the file until the user decides. On "Upload Anyway" / "Continue Upload" (and
   on backend-unavailable "Close"), a new `change` event is re-dispatched via a
   `DataTransfer`-constructed `FileList` so the host page processes it normally.

## Known limitations / improvement ideas

**Model performance**
- Only 4 YOLO classes; OCR has to infer document subtype (passport vs ID vs card) from text.
- OCR runs a 4-provider sequential fallback chain (TrOCR→RapidOCR→EasyOCR→Tesseract) — slow.
- `facebook/bart-large-mnli` zero-shot classifier is heavy for real-time use; a small
  fine-tuned classifier on the actual document labels would be faster and likely more accurate.
- Risk aggregation is `max()`, not additive — multiple medium-risk detections in one
  image don't compound.

**System/API**
- `analyze_image()` runs synchronously inside an async FastAPI handler, blocking the
  event loop under load. Should run via `run_in_executor`.
- OCR/age models lazy-load on first request (YOLO is the only one pre-warmed in
  `lifespan`), causing a slow first real request.
- No caching — identical re-uploaded images re-run the full pipeline.
- No request timeout for pathological images/slow OCR providers.

**Extension**
- Only intercepts `<input type="file">` `change` events — misses drag-and-drop and
  clipboard-paste upload flows.
- Filters by `file.type.startsWith('image/')`; files with empty/wrong MIME type are
  silently skipped (no extension-based fallback).
- `<input multiple>` selections: only `files[0]` is analyzed, rest pass through unchecked.

## Working notes

- `python -m uvicorn`, not plain `uvicorn` (not in PATH on this machine).
- `leak_lock/` (underscore, not the `leaklock/` package) is an unrelated Python venv directory — ignore it.
- `__pycache__/` and `*.pyc` are gitignored; if pulls fail with "untracked files would be
  overwritten," delete local `__pycache__` dirs and retry.
- requirements.txt pins `transformers==4.51.3` and `huggingface_hub==0.30.2` — these
  versions matter for the ONNX/HF age estimators in `age_estimators.py`.
