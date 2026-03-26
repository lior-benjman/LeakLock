# Layered Risk Pipeline Plan

## Goal

Turn LeakLock from a training-focused repository into a layered image-risk analysis pipeline.

The pipeline should:

1. Run a trained YOLOv8 model on an uploaded image.
2. Route each detected object to the most relevant downstream layer.
3. Calculate risk and explain why the risk was assigned.

## Proposed Layers

### 1. Detection Layer

Responsibility:

- Load the trained YOLOv8 weights.
- Detect supported classes in the uploaded image.
- Return normalized detection results with class name, confidence, and bounding box.

Current source for weights:

- `runs_sensitive/yolov82/weights/best.pt`

### 2. Routing Layer

Responsibility:

- Decide which specialized layer handles each detection.

Routing rules:

- `face` -> Face Age Layer
- `card`, `document`, `id` -> OCR Extraction Layer -> OCR Risk Evaluation Layer
- `license-plates` -> License Plate Risk Layer

### 3. Face Age Layer

Responsibility:

- Estimate the age of the detected face.
- Convert the age estimate into a risk score.

Requested rules:

- `16+` years old -> `0%` risk
- `12-15` years old -> `30%` risk

Open point:

- The user did not define the risk for `0-11`.
- Current implementation assumption: `0-11` -> `60%` risk.
- This must remain configurable.

### 4. OCR Extraction Layer

Responsibility:

- Crop the relevant object region.
- Extract OCR text from `card`, `document`, or `id` detections.

### 5. OCR Risk Evaluation Layer

Responsibility:

- Inspect OCR text.
- Estimate how sensitive the extracted information is.
- Produce a risk score and machine-readable evidence.

Target future implementation:

- A trained text classifier specialized for sensitive/personal-data detection.

Short-term implementation:

- Rule-based baseline using sensitive-data patterns and keyword matches.

### 6. Reason Generation Layer

Responsibility:

- Convert risk evidence into a human-readable reason.

Target future implementation:

- A generative model that writes a natural explanation.

Short-term implementation:

- Template-based reason generation.

### 7. License Plate Risk Layer

Responsibility:

- Assign fixed risk when a license plate is detected.

Requested rule:

- `license-plates` -> `75%` risk
- Reason: `"License-plate was detected"`

## Execution Plan

### Phase 1: Pipeline Skeleton

- Add a dedicated `leaklock` Python package.
- Add shared models and configuration.
- Add a CLI entrypoint for local analysis.

### Phase 2: Detection and Routing

- Wrap YOLOv8 inference behind a provider interface.
- Add routing logic per detected class.

### Phase 3: Specialized Risk Layers

- Add face age risk layer with pluggable age estimator.
- Add OCR extraction layer with pluggable OCR provider.
- Add OCR risk evaluation layer with pluggable text classifier.
- Add license plate rule layer.

### Phase 4: Explanation Layer

- Add templated reason generation now.
- Add interface for future generative-AI replacement.

### Phase 5: Model Upgrades

- Train or integrate a face-age estimation model.
- Train or integrate a sensitive-text classifier for OCR results.
- Optionally replace template reasons with an LLM-backed reason generator.

## Deliverables Added In This Iteration

- Layered architecture package
- Detection and routing pipeline
- Baseline OCR risk evaluator
- Fixed license plate risk layer
- Configurable face-age risk rules
- CLI entrypoint for single-image analysis

## Known Gaps After This Iteration

- No trained face-age model is present in the repo yet
- No trained OCR-sensitive-text classifier is present in the repo yet
- The active shell environment does not currently expose a real Python interpreter on PATH
