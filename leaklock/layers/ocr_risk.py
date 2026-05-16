from __future__ import annotations

import json
import re
from typing import Any

try:
    import spacy
except ImportError:  # pragma: no cover - depends on runtime
    spacy = None

from ..config import PipelineConfig
from ..models import OcrExtraction, RiskResult
from ..services.reason_generators import ReasonGenerator, TemplateReasonGenerator
from ..services.text_classifiers import (
    RuleBasedSensitiveTextClassifier,
    SensitiveTextClassifier,
)


RULE_BASED_WEIGHT = 0.6
DOCUMENT_ML_WEIGHT = 0.4


def _clamp_percent(value: float | int) -> int:
    return max(0, min(100, int(round(value))))


def _score_to_percent(score: float) -> int:
    return _clamp_percent(score * 100.0)


class OcrRiskEvaluationLayer:
    """Scores extracted OCR text for privacy risk."""

    def __init__(
        self,
        config: PipelineConfig,
        classifier: SensitiveTextClassifier | None = None,
        reason_generator: ReasonGenerator | None = None,
    ) -> None:
        self._config = config
        self._classifier = classifier or RuleBasedSensitiveTextClassifier(config)
        self._reason_generator = reason_generator or TemplateReasonGenerator()

    def evaluate(self, routed_from_class: str, extraction: OcrExtraction) -> RiskResult:
        rule_classification = self._classifier.classify(extraction.text)
        try:
            rule_risk_percent = _clamp_percent(float(rule_classification.get("risk_percent", 0)))
        except (TypeError, ValueError):
            rule_risk_percent = 0

        document_ml_analysis: dict[str, Any] | None = None
        ml_risk_percent: int | None = None
        fallback_reason: str | None = None

        if self._config.enable_document_ml_risk:
            try:
                document_ml_analysis = evaluate_document_risk(extraction.text)
                ml_risk_percent = _clamp_percent(document_ml_analysis["risk"]["risk_percent"])
            except Exception as exc:  # noqa: BLE001 - keep OCR path stable if ML branch fails.
                fallback_reason = f"{exc.__class__.__name__}: {exc}"
        else:
            fallback_reason = "Document ML risk branch disabled by configuration"

        if ml_risk_percent is None:
            blended_risk_percent = rule_risk_percent
        else:
            blended_risk_percent = _clamp_percent(
                (rule_risk_percent * RULE_BASED_WEIGHT)
                + (ml_risk_percent * DOCUMENT_ML_WEIGHT)
            )

        classification: dict[str, Any] = {
            **rule_classification,
            "risk_percent": blended_risk_percent,
            "rule_based_risk_percent": rule_risk_percent,
            "ml_risk_percent": ml_risk_percent,
            "blend_weights": {
                "rule_based": RULE_BASED_WEIGHT,
                "document_ml": DOCUMENT_ML_WEIGHT,
            },
        }
        if fallback_reason:
            classification["ml_fallback_reason"] = fallback_reason

        reason = self._reason_generator.generate_ocr_reason(
            routed_from_class=routed_from_class,
            extraction=extraction,
            classification=classification,
        )
        return RiskResult(
            layer_name="ocr_risk_evaluation_layer",
            routed_from_class=routed_from_class,
            risk_percent=classification["risk_percent"],
            reason=reason,
            evidence={
                "ocr": extraction.to_dict(),
                "classification": classification,
                "document_ml_analysis": document_ml_analysis,
                "digit_sequences": re.findall(r"\d{4,}", extraction.text),
            },
        )


# Load NLP Models
if spacy is None:
    nlp = None
else:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Keep pipeline operational even when full spaCy model is unavailable.
        nlp = spacy.blank("en")


def _load_zero_shot_classifier():
    try:
        from transformers import pipeline as transformers_pipeline
    except ImportError:
        return None

    model_name = "facebook/bart-large-mnli"
    try:
        # Prefer local cache first so analysis works in restricted/no-network runtimes.
        return transformers_pipeline(
            "zero-shot-classification",
            model=model_name,
            local_files_only=True,
        )
    except Exception:
        try:
            return transformers_pipeline(
                "zero-shot-classification",
                model=model_name,
                local_files_only=False,
            )
        except Exception:
            return None


classifier = None
classifier_loaded = False


def _get_zero_shot_classifier():
    global classifier, classifier_loaded

    if not classifier_loaded:
        classifier = _load_zero_shot_classifier()
        classifier_loaded = True

    return classifier

# Config
DOCUMENT_LABELS = [
    "government id",
    "financial document",
    "medical document",
    "contact information",
    "generic document",
    "non sensitive document"
]

DOCUMENT_TYPE_TO_BASE_RISK = {
    "government id": 0.75,
    "financial document": 0.65,
    "medical document": 0.70,
    "contact information": 0.45,
    "generic document": 0.20,
    "non sensitive document": 0.05
}

# Text Normalization
def normalize_ocr_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()

# Pattern Detection
EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
PHONE_REGEX = r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
DATE_REGEX = r'\b(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})\b'
LONG_NUMBER_REGEX = r'\b\d{6,16}\b'


DOCUMENT_KEYWORDS = {
    "government_id_keywords": [
        "passport",
        "identity",
        "id",
        "nationality",
        "date of birth",
        "license"
    ],
    "financial_keywords": [
        "card",
        "credit",
        "bank",
        "iban",
        "account"
    ],
    "medical_keywords": [
        "patient",
        "hospital",
        "doctor",
        "diagnosis",
        "medical",
        "health record",
        "allergy",
        "allergies",
        "immunization",
        "immunizations",
        "vaccine",
        "treatment",
        "prescription",
    ]
}


def detect_patterns(text: str) -> dict[str, Any]:

    text_lower = text.lower()

    emails = re.findall(EMAIL_REGEX, text)
    phones = re.findall(PHONE_REGEX, text)
    dates = re.findall(DATE_REGEX, text)
    long_numbers = re.findall(LONG_NUMBER_REGEX, text)

    keyword_hits = {}

    for category, keywords in DOCUMENT_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in text_lower]
        keyword_hits[category] = hits

    return {
        "emails": emails,
        "phones": phones,
        "dates": dates,
        "long_numbers": long_numbers,
        "keyword_hits": keyword_hits
    }

# Entity Extraction
def extract_sensitive_entities(text: str) -> list[dict[str, str]]:
    if nlp is None:
        return []

    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities

# Document Classification
def classify_document(text: str) -> dict[str, Any]:
    classifier = _get_zero_shot_classifier()
    if classifier is None:
        # Conservative fallback when zero-shot model isn't available.
        return {
            "document_type": "generic document",
            "confidence": 0.30,
        }

    result = classifier(text, DOCUMENT_LABELS)

    return {
        "document_type": result["labels"][0],
        "confidence": result["scores"][0]
    }

# Risk Calculation
def score_to_risk_level(risk_percent: int) -> str:

    if risk_percent > 75:
        return "high"

    if risk_percent > 40:
        return "medium"

    return "low"


def aggregate_risk(
    document_type: str,
    confidence: float,
    entities: list[dict[str, str]],
    pattern_matches: dict[str, Any],
) -> dict[str, Any]:

    score = DOCUMENT_TYPE_TO_BASE_RISK.get(document_type, 0.10)

    score += confidence * 0.2

    if pattern_matches["emails"]:
        score += 0.1

    if pattern_matches["phones"]:
        score += 0.1

    if pattern_matches["dates"]:
        score += 0.05

    if pattern_matches["long_numbers"]:
        score += 0.15

    keyword_hits = pattern_matches.get("keyword_hits", {})
    government_id_hits = keyword_hits.get("government_id_keywords", [])
    financial_hits = keyword_hits.get("financial_keywords", [])
    medical_hits = keyword_hits.get("medical_keywords", [])

    if government_id_hits:
        score += min(len(government_id_hits) * 0.08, 0.25)

    if financial_hits:
        score += min(len(financial_hits) * 0.06, 0.20)

    if medical_hits:
        score += min(len(medical_hits) * 0.08, 0.25)

    score = min(score, 1.0)
    risk_percent = _score_to_percent(score)

    return {
        "risk_percent": risk_percent,
        "risk_score": score,
        "risk_level": score_to_risk_level(risk_percent)
    }

# Full Pipeline
def evaluate_document_risk(text: str) -> dict[str, Any]:

    text = normalize_ocr_text(text)

    patterns = detect_patterns(text)

    entities = extract_sensitive_entities(text)

    classification = classify_document(text)

    risk = aggregate_risk(
        classification["document_type"],
        classification["confidence"],
        entities,
        patterns
    )

    return {
        "document_type": classification["document_type"],
        "confidence": classification["confidence"],
        "entities": entities,
        "patterns": patterns,
        "risk": risk
    }


# Test
if __name__ == "__main__":

    test_text = """
    Passport
    Name: John Smith
    Passport Number: 123456789
    Date of Birth: 12/08/2003
    """

    result = evaluate_document_risk(test_text)

    print(json.dumps(result, indent=2))
