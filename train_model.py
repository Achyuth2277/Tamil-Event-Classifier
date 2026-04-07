"""Train Tamil event classification models from dataset.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

from preprocess import clean_tamil_text, tokenize_tamil_text


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_tamil.csv"
MODELS_DIR = BASE_DIR / "models"
SUPPORTED_LABELS = [
    "Sports",
    "Politics",
    "Weather",
    "Accident",
    "Entertainment",
    "Education",
    "Crime",
]


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    required_columns = {"tamil_text", "english_text", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {sorted(missing_columns)}")

    df = df.dropna(subset=["tamil_text", "english_text", "label"]).copy()
    df["tamil_text"] = df["tamil_text"].astype(str).map(clean_tamil_text)
    df["tokens"] = df["tamil_text"].map(tokenize_tamil_text)
    df["processed_tamil_text"] = df["tokens"].map(" ".join)
    df["english_text"] = df["english_text"].astype(str).fillna("")
    df["label"] = df["label"].astype(str).str.strip()

    invalid_labels = sorted(set(df["label"]) - set(SUPPORTED_LABELS))
    if invalid_labels:
        raise ValueError(f"Unsupported labels found: {invalid_labels}")

    return df


def build_models() -> Dict[str, Pipeline]:
    """Create Tamil-first machine learning pipelines."""
    vectorizer = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1),
            ),
        ]
    )

    models = {
        "svm": Pipeline(
            steps=[
                ("tfidf", vectorizer),
                ("classifier", SVC(kernel="linear", probability=True, random_state=42)),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("tfidf", vectorizer),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    return models


def evaluate_model(name: str, model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> Tuple[float, pd.DataFrame]:
    """Print model metrics and return accuracy with confusion matrix."""
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=SUPPORTED_LABELS)
    matrix_df = pd.DataFrame(matrix, index=SUPPORTED_LABELS, columns=SUPPORTED_LABELS)

    print(f"\n{name.upper()} RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(matrix_df)

    return accuracy, matrix_df


def save_artifacts(models: Dict[str, Pipeline], best_model_name: str) -> None:
    """Save all models and metadata with joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    class_counts = df["label"].value_counts().reindex(SUPPORTED_LABELS, fill_value=0).to_dict()

    for model_name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{model_name}_model.joblib")

    best_model = models[best_model_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(
        {
            "best_model_name": best_model_name,
            "labels": SUPPORTED_LABELS,
            "dataset_path": str(DATASET_PATH),
            "dataset_size": int(len(df)),
            "class_counts": class_counts,
            "feature_strategy": "Tamil TF-IDF with word n-grams (1,2) and char_wb n-grams (3,5)",
        },
        MODELS_DIR / "metadata.joblib",
    )


def train() -> None:
    """Main training function."""
    df = load_dataset()
    print(f"Loaded {len(df)} rows from {DATASET_PATH.name}")

    x_train, x_test, y_train, y_test = train_test_split(
        df["processed_tamil_text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    models = build_models()
    scores = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        accuracy, _ = evaluate_model(name, model, x_test, y_test)
        scores[name] = accuracy

    best_model_name = max(scores, key=scores.get)
    print(f"\nBest model: {best_model_name} with accuracy {scores[best_model_name]:.4f}")

    save_artifacts(models, best_model_name)
    print(f"Saved models to: {MODELS_DIR}")


if __name__ == "__main__":
    train()
