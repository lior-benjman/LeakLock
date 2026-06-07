"""LeakLock FastAPI backend — exposes the analysis pipeline for the Chrome extension."""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from leaklock.config import PipelineConfig
from leaklock.pipeline import LeakLockPipeline

_pipeline: LeakLockPipeline | None = None


def _get_pipeline() -> LeakLockPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LeakLockPipeline(config=PipelineConfig())
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_pipeline()
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
        else:
            add(f"{class_name.replace('_', ' ').title()} detected")

        # OCR rule-based matched features
        classification = evidence.get("classification") or {}
        matched = classification.get("matched_features") or []
        if matched:
            add("Sensitive OCR keywords detected")
            for feat in matched:
                if feat in ("passport", "identity", "identification"):
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
    return {"status": "ok", "pipeline": "ready" if _pipeline else "initializing"}


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
        result = _get_pipeline().analyze_image(tmp_path)

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
            "explanations": explanations,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
