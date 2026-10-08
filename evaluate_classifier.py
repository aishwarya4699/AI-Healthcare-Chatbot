"""Evaluate the page classifier on a synthetic held-out split.

Metrics on this demo set are not representative of production clinical performance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.classification import DOCUMENT_CLASSES, load_classifier, load_labeled_texts
from train_classifier import RANDOM_STATE, TEST_SIZE

REPORTS_DIR = Path("reports")


def main() -> None:
    texts, labels = load_labeled_texts()
    _, X_test, _, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    model = load_classifier()
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=DOCUMENT_CLASSES,
        digits=3,
        zero_division=0,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "classification_report.txt"
    caveat = (
        "NOTE: Evaluation uses a small synthetic, non-PHI dataset. "
        "These metrics demonstrate the pipeline only and are not "
        "representative of production clinical performance.\n\n"
    )
    report_path.write_text(
        caveat
        + f"accuracy: {accuracy:.3f}\n"
        + f"macro_precision: {precision:.3f}\n"
        + f"macro_recall: {recall:.3f}\n"
        + f"macro_f1: {f1:.3f}\n\n"
        + report,
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_test, y_pred, labels=DOCUMENT_CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=DOCUMENT_CLASSES,
    ).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Synthetic demo confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(report_path.read_text(encoding="utf-8"))
    print(f"Wrote {report_path}")
    print(f"Wrote {REPORTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
