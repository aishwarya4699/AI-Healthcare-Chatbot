from pathlib import Path
from typing import List, Tuple

import joblib

DOCUMENT_CLASSES = [
    "request_form",
    "clinical_note",
    "therapy_note",
    "policy_document",
]

DEFAULT_MODEL_PATH = Path("models/document_classifier.joblib")
DEFAULT_CLASSIFIER_DATA_DIR = Path("classifier_data")


def load_labeled_texts(
    data_dir: Path = DEFAULT_CLASSIFIER_DATA_DIR,
) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for label in DOCUMENT_CLASSES:
        class_dir = data_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing classifier data folder: {class_dir}")
        for path in sorted(class_dir.glob("*.txt")):
            texts.append(path.read_text(encoding="utf-8"))
            labels.append(label)
    if not texts:
        raise FileNotFoundError(f"No .txt training files found under {data_dir}")
    return texts, labels


def load_classifier(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier not found at {model_path}. "
            "Train it first with: python train_classifier.py"
        )
    return joblib.load(model_path)


def save_classifier(model, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def predict_page(model, text: str) -> Tuple[str, float]:
    probabilities = model.predict_proba([text or ""])[0]
    best_index = probabilities.argmax()
    label = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return label, confidence
