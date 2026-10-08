from pathlib import Path
from typing import List, Tuple

import joblib

# The 4 document types the classifier can predict (also used for the UI dropdown)
DOCUMENT_CLASSES = [
    "request_form",
    "clinical_note",
    "therapy_note",
    "policy_document",
]

# Where the trained model is saved, and where the training documents live
DEFAULT_MODEL_PATH = Path("models/document_classifier.joblib")
DEFAULT_CLASSIFIER_DATA_DIR = Path("classifier_data")


# Read all training documents; the folder name is the label (used for training)
def load_labeled_texts(
    data_dir: Path = DEFAULT_CLASSIFIER_DATA_DIR,
) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for label in DOCUMENT_CLASSES:
        class_dir = data_dir / label
        # Stop with a clear error if a type's folder is missing
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing classifier data folder: {class_dir}")
        # Add every .txt file in this folder with its label
        for path in sorted(class_dir.glob("*.txt")):
            texts.append(path.read_text(encoding="utf-8"))
            labels.append(label)
    # Stop if no training files were found at all
    if not texts:
        raise FileNotFoundError(f"No .txt training files found under {data_dir}")
    return texts, labels


# Load the saved classifier from disk (tells you to train it first if missing)
def load_classifier(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier not found at {model_path}. "
            "Train it first with: python train_classifier.py"
        )
    return joblib.load(model_path)


# Save the trained classifier to disk (creates the models/ folder if needed)
def save_classifier(model, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


# Predict the document type and the confidence, e.g. ("clinical_note", 0.47)
def predict_page(model, text: str) -> Tuple[str, float]:
    # Get a probability for each of the 4 types
    probabilities = model.predict_proba([text or ""])[0]
    # Pick the type with the highest probability
    best_index = probabilities.argmax()
    label = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return label, confidence
