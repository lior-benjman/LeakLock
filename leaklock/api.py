"""LeakLock FastAPI backend — exposes the analysis pipeline for the Chrome extension."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from leaklock.config import PipelineConfig
from leaklock.pipeline import LeakLockPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent

# A request can't actually interrupt the underlying worker thread once started
# (Python threads aren't preemptible), so this bounds how long the HTTP
# response waits, not how long the model computation itself runs. It exists
# so an abandoned request (closed tab) or a pathologically slow image
# produces a clean error instead of the client hanging indefinitely.
ANALYSIS_TIMEOUT_SECONDS = 60.0

_pipeline: LeakLockPipeline | None = None
_pipeline_config: PipelineConfig | None = None
# Serializes access to the shared pipeline's model objects (YOLO, OCR engines,
# age estimators, zero-shot classifier). None of these are documented as safe
# for concurrent inference calls on the same instance from multiple threads,
# so concurrent requests queue here instead of racing on shared model state.
_pipeline_lock = threading.Lock()


def _get_pipeline_config() -> PipelineConfig:
    global _pipeline_config
    if _pipeline_config is None:
        # Match notebooks/leaklock_upload_pipeline.ipynb so the extension backend
        # uses the same configured model weights as the notebook pipeline.
        _pipeline_config = PipelineConfig(repo_root=REPO_ROOT)
    return _pipeline_config


def _get_pipeline() -> LeakLockPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LeakLockPipeline(config=_get_pipeline_config())
    return _pipeline


def _run_pipeline_locked(image_path: str):
    """Runs inference under _pipeline_lock. Called via run_in_executor, so the
    lock-wait happens on a worker thread and never blocks the event loop."""
    with _pipeline_lock:
        return _get_pipeline().analyze_image(image_path)


def _model_metadata() -> dict[str, object]:
    config = _get_pipeline_config()
    return {
        "repo_root": str(config.repo_root),
        "yolo_weights": str(config.yolo_weights_path),
        "yolo_weights_exists": config.yolo_weights_path.exists(),
        "detection_confidence_threshold": config.detection_confidence_threshold,
        "onnx_age_model_repo": config.hf_age_model_repo_id,
        "transformers_age_model": config.hf_age_classifier_model_id,
    }


def _runtime_metadata() -> dict[str, object]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "age_dependencies": {
            "huggingface_hub": importlib.util.find_spec("huggingface_hub") is not None,
            "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
            "deepface": importlib.util.find_spec("deepface") is not None,
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = _get_pipeline()
    # Force every layer's model to load now, off the event loop, so the
    # first real request doesn't pay model-loading latency.
    await asyncio.get_event_loop().run_in_executor(None, pipeline.warm_up)
    yield


app = FastAPI(title="LeakLock API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _risk_level(score: int) -> str:
    if score <= 33:
        return "low"
    if score <= 66:
        return "medium"
    return "high"


def _build_explanations(analyses: list) -> list[str]:
    explanations: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        if text not in seen:
            seen.add(text)
            explanations.append(text)

    for analysis in analyses:
        if analysis.risk.risk_percent == 0:
            continue

        class_name = analysis.detection.class_name.lower().replace("-", "_")
        evidence = analysis.risk.evidence or {}
        reason = analysis.risk.reason or ""

        # Class-based explanations
        if class_name == "face":
            add("Face detected")
            if "younger" in reason or "11 years" in reason:
                add("Minor (under 12) detected — elevated privacy risk")
            elif "12-15" in reason:
                add("Teen (12–15) face detected")
        elif class_name in ("license_plates", "license_plate"):
            add("License plate detected")
        elif class_name == "passport":
            add("Passport detected")
            add("Personal information detected")
        elif class_name in ("id", "id_card"):
            add("ID card detected")
            add("Personal information detected")
        elif class_name == "document":
            add("Document detected")
        elif class_name == "card":
            add("Card document detected")
        elif class_name == "text_region":
            add("Sensitive visible text detected")
        else:
            add(f"{class_name.replace('_', ' ').title()} detected")

        # OCR rule-based matched features
        classification = evidence.get("classification") or {}
        matched = classification.get("matched_features") or []
        if matched:
            add("Sensitive OCR keywords detected")
            for feat in matched:
                if feat == "credential":
                    add("Password or credential text detected")
                elif feat == "wifi_network":
                    add("WiFi network information detected")
                elif feat in ("passport", "identity", "identification"):
                    add("Identity document indicators detected")
                elif feat in ("credit card", "card number", "iban", "account"):
                    add("Financial information detected")
                elif feat == "name":
                    add("Personal name detected in text")
                elif feat == "email":
                    add("Email address detected in text")
                elif feat == "long_digit_sequence":
                    add("Sensitive number sequences detected")

        # ML document analysis enrichment
        doc_ml = evidence.get("document_ml_analysis") or {}
        doc_type = doc_ml.get("document_type", "")
        patterns = doc_ml.get("patterns") or {}

        if doc_type == "medical document":
            add("Medical document detected")
        elif doc_type == "government id":
            add("Government ID detected")
        elif doc_type == "financial document":
            add("Financial document detected")

        if patterns.get("emails"):
            add("Email address detected in text")
        if patterns.get("phones"):
            add("Phone number detected in text")

    return explanations


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": "ready" if _pipeline else "initializing",
        "model": _model_metadata(),
        "runtime": _runtime_metadata(),
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    original_name = file.filename or "upload.jpg"
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}:
        suffix = ".jpg"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file received")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _run_pipeline_locked, tmp_path),
                timeout=ANALYSIS_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Analysis did not complete within {ANALYSIS_TIMEOUT_SECONDS:.0f}s. "
                    "The image may be unusually large or complex — try again or use a smaller image."
                ),
            ) from exc

        risk_score = result.overall_risk_percent
        detections = [
            {
                "class_name": a.detection.class_name,
                "confidence": round(a.detection.confidence, 3),
            }
            for a in result.analyses
        ]
        explanations = _build_explanations(result.analyses)

        if risk_score > 0 and not explanations:
            explanations.append("Sensitive content detected")

        return {
            "risk_score": risk_score,
            "risk_level": _risk_level(risk_score),
            "detections": detections,
            "analyses": [a.to_dict() for a in result.analyses],
            "result": result.to_dict(),
            "explanations": explanations,
            "model": _model_metadata(),
            "runtime": _runtime_metadata(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
