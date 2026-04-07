"""Run Tamil-first event prediction for a Tamil sentence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from preprocess import clean_tamil_text, tokenize_tamil_text
from translate import translate_tamil_to_english


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"
LOW_CONFIDENCE_THRESHOLD = 0.45


CATEGORY_DETAILS = {
    "Crime": [
        {
            "subtype": "Arrest / Police Action",
            "keywords": ["கைது", "போலீசார்", "காவல்துறை", "குற்றவாளி"],
            "description": "The sentence appears to describe a criminal arrest or police enforcement action.",
        },
        {
            "subtype": "Robbery",
            "keywords": ["கொள்ளை", "அபகரித்த", "வங்கி", "நகை கடை"],
            "description": "The sentence appears to describe a robbery involving money, a bank, or a shop.",
        },
        {
            "subtype": "Fraud / Scam",
            "keywords": ["மோசடி", "கள்ளநோட்டு", "ஏமாற்று", "இணைய"],
            "description": "The sentence appears to describe fraud, counterfeit activity, or an online scam.",
        },
        {
            "subtype": "Theft",
            "keywords": ["திருட்டு", "திருடப்பட்டது", "பணம் திருடப்பட்டது", "வீட்டில்"],
            "description": "The sentence appears to describe theft of property or money.",
        },
        {
            "subtype": "Snatching / Extortion",
            "keywords": ["பறிப்பு", "பணம் பறித்த", "மிரட்டி", "சங்கிலி"],
            "description": "The sentence appears to describe extortion or snatching of valuables.",
        },
    ],
    "Weather": [
        {
            "subtype": "Heavy Rain",
            "keywords": ["மழை", "கனமழை", "இடி"],
            "description": "The sentence appears to describe rain or storm-related weather.",
        },
        {
            "subtype": "Cyclone / Wind",
            "keywords": ["புயல்", "காற்று", "எச்சரிக்கை"],
            "description": "The sentence appears to describe cyclone conditions or strong winds.",
        },
        {
            "subtype": "Heat / Temperature",
            "keywords": ["வெப்பநிலை", "வெப்பஅலை", "சூடு"],
            "description": "The sentence appears to describe hot weather or a temperature increase.",
        },
    ],
    "Accident": [
        {
            "subtype": "Road Accident",
            "keywords": ["கார்", "பேருந்து", "லாரி", "பைக்", "சாலை"],
            "description": "The sentence appears to describe a road traffic accident.",
        },
        {
            "subtype": "Train Accident",
            "keywords": ["ரயில்", "தடம் புரண்டது"],
            "description": "The sentence appears to describe a train accident or derailment.",
        },
        {
            "subtype": "Industrial Accident",
            "keywords": ["தொழிற்சாலை", "வெடிப்பு", "தொழிலாளி"],
            "description": "The sentence appears to describe an industrial or workplace accident.",
        },
    ],
    "Sports": [
        {
            "subtype": "Match Result",
            "keywords": ["வெற்றி", "கோப்பை", "இறுதி", "முன்னிலை"],
            "description": "The sentence appears to describe a sports result or tournament outcome.",
        },
        {
            "subtype": "Player Performance",
            "keywords": ["கோல்கள்", "தங்கம்", "சாம்பியன்", "வீரர்", "வீராங்கனை"],
            "description": "The sentence appears to describe an athlete's performance or achievement.",
        },
    ],
    "Politics": [
        {
            "subtype": "Government Announcement",
            "keywords": ["அரசு", "அறிவித்தார்", "திட்டம்", "கொள்கை"],
            "description": "The sentence appears to describe a government or ministerial announcement.",
        },
        {
            "subtype": "Election / Party Activity",
            "keywords": ["தேர்தல்", "கட்சித் தலைவர்", "வேட்பாளர்", "பிரச்சாரம்"],
            "description": "The sentence appears to describe election activity or party politics.",
        },
        {
            "subtype": "Legislative Action",
            "keywords": ["பாராளுமன்றம்", "சட்டமன்றம்", "மசோதா"],
            "description": "The sentence appears to describe parliamentary or assembly proceedings.",
        },
    ],
    "Entertainment": [
        {
            "subtype": "Movie Release",
            "keywords": ["திரைப்படம்", "படம்", "வெளியானது"],
            "description": "The sentence appears to describe a movie release or film update.",
        },
        {
            "subtype": "Music / Stage Event",
            "keywords": ["இசை", "பாடல்", "பாடகர்", "நிகழ்ச்சி"],
            "description": "The sentence appears to describe a music or cultural performance.",
        },
        {
            "subtype": "TV / Web Series",
            "keywords": ["தொலைக்காட்சி", "தொடர்", "வெப் தொடர்", "சீசன்"],
            "description": "The sentence appears to describe a television or web-series update.",
        },
    ],
    "Education": [
        {
            "subtype": "Exam / Academic Schedule",
            "keywords": ["தேர்வு", "அட்டவணை", "செமஸ்டர்"],
            "description": "The sentence appears to describe an exam or academic schedule.",
        },
        {
            "subtype": "School / College Event",
            "keywords": ["பள்ளி", "கல்லூரி", "கண்காட்சி", "வினாடி வினா"],
            "description": "The sentence appears to describe a school or college academic event.",
        },
        {
            "subtype": "Curriculum / Learning",
            "keywords": ["பாடத்திட்டம்", "கற்றல்", "பயிற்சி", "வகுப்புகள்"],
            "description": "The sentence appears to describe learning, coaching, or curriculum changes.",
        },
    ],
}


def load_model_artifacts(model_path: Path = DEFAULT_MODEL_PATH) -> Tuple[object, Dict]:
    """Load saved model and metadata."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}. Run train_model.py first."
        )

    model = joblib.load(model_path)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def detect_event_subtype(cleaned_sentence: str, predicted_category: str) -> Dict[str, str]:
    """Return a simple subtype and explanation using keyword matching."""
    category_rules = CATEGORY_DETAILS.get(predicted_category, [])

    for rule in category_rules:
        if any(keyword in cleaned_sentence for keyword in rule["keywords"]):
            return {
                "event_subtype": rule["subtype"],
                "event_information": rule["description"],
            }

    return {
        "event_subtype": "General Event",
        "event_information": f"The sentence appears to be a general {predicted_category.lower()}-related event.",
    }


def predict_event(tamil_sentence: str) -> Dict[str, str]:
    """Predict the event category from Tamil text."""
    cleaned_sentence = clean_tamil_text(tamil_sentence)
    tokens = tokenize_tamil_text(cleaned_sentence)

    if not cleaned_sentence or not tokens:
        return {
            "tamil_text": tamil_sentence,
            "english_translation": "",
            "predicted_category": "Unable to classify",
            "confidence_note": "Input is empty after cleaning. Please enter a meaningful Tamil sentence.",
            "translation_note": "No translation generated.",
            "cleaned_tamil_text": "",
            "tokens": [],
        }

    processed_tamil_text = " ".join(tokens)
    english_translation = translate_tamil_to_english(cleaned_sentence)
    model, metadata = load_model_artifacts()
    labels = metadata.get("labels", [])
    best_model_name = metadata.get("best_model_name", "unknown")
    model_classes = list(getattr(model, "classes_", labels))

    prediction = model.predict([processed_tamil_text])[0]
    confidence_note = "Prediction generated successfully."
    translation_note = "Translation generated successfully."
    if not english_translation:
        translation_note = (
            "Live translation is unavailable in the current environment. "
            "Classification used Tamil text directly."
        )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([processed_tamil_text])[0]
        best_score = float(np.max(probabilities))
        probability_pairs = sorted(
            zip(model_classes, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        top_predictions: List[Dict[str, str]] = [
            {
                "category": label,
                "probability": f"{score:.2f}",
            }
            for label, score in probability_pairs[:3]
        ]
        if best_score < LOW_CONFIDENCE_THRESHOLD:
            confidence_note = (
                f"Low confidence prediction ({best_score:.2f}). "
                "The sentence may be outside the training examples."
            )
    else:
        best_score = None
        top_predictions = []

    result = {
        "tamil_text": tamil_sentence,
        "cleaned_tamil_text": cleaned_sentence,
        "tokens": tokens,
        "english_translation": english_translation,
        "predicted_category": prediction if prediction in labels else "Unable to classify",
        "confidence_note": confidence_note,
        "translation_note": translation_note,
        "model_used": best_model_name,
        "dataset_size": metadata.get("dataset_size", "unknown"),
        "feature_strategy": metadata.get("feature_strategy", "unknown"),
        "class_counts": metadata.get("class_counts", {}),
        "top_predictions": top_predictions,
    }

    result.update(detect_event_subtype(cleaned_sentence, result["predicted_category"]))

    if best_score is not None:
        result["confidence_score"] = f"{best_score:.2f}"

    return result


def format_prediction_report(result: Dict[str, object]) -> str:
    """Create a short and informative prediction report for CLI output."""
    translation_text = result["english_translation"] or "Unavailable"

    lines = [
        f'Input: "{result["tamil_text"]}"',
        f'English Translation: "{translation_text}"',
        f'Predicted Category: {result["predicted_category"]}',
        f'Event Type: {result["event_subtype"]}',
        f'Info: {result["event_information"]}',
    ]

    if "confidence_score" in result:
        lines.append(f'Confidence: {result["confidence_score"]}')

    if result["confidence_note"] != "Prediction generated successfully.":
        lines.append(f'Note: {result["confidence_note"]}')

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict event category from Tamil text.")
    parser.add_argument(
        "--text",
        type=str,
        help="Tamil sentence for prediction. If omitted, you will be prompted.",
    )
    return parser.parse_args()


def ensure_utf8_output() -> None:
    """Make Tamil text printable in Windows terminals when possible."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ensure_utf8_output()
    args = parse_args()
    tamil_sentence = args.text or input("Enter a Tamil sentence: ").strip()
    result = predict_event(tamil_sentence)
    print(format_prediction_report(result))


if __name__ == "__main__":
    main()
