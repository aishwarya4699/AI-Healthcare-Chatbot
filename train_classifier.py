"""Train a TF-IDF + Logistic Regression page classifier on synthetic demo texts."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.classification import load_labeled_texts, save_classifier

RANDOM_STATE = 42
TEST_SIZE = 0.25
# Regularization strength is chosen by 5-fold cross-validation (log-loss) on the
# training split only. The grid is capped at 30 because synthetic data is easy to
# separate, and a weaker penalty would make the model overconfident on real docs.
C_GRID = [1, 3, 10, 30]


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    texts, labels = load_labeled_texts()
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    search = GridSearchCV(
        build_model(),
        {"clf__C": C_GRID},
        scoring="neg_log_loss",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Selected C={search.best_params_['clf__C']} by 5-fold CV log-loss")
    save_classifier(model)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Saved model to models/document_classifier.joblib")
    print(f"Train accuracy (synthetic): {train_acc:.3f}")
    print(f"Held-out accuracy (synthetic): {test_acc:.3f}")
    print(
        "These scores are for the demo dataset only and are not "
        "representative of production clinical performance."
    )


if __name__ == "__main__":
    main()
