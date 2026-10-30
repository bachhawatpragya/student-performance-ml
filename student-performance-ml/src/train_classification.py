
from datetime import datetime, timezone
from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import (
    create_feature_target,
    create_performance_category,
    create_preprocessor,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.evaluate import evaluate_classification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def train_classification(df, run_id):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, final_grades = create_feature_target(df)

    # The target is derived from G3, but G3 is not in X.
    y = create_performance_category(final_grades)
    student_ids = df["student_id"].reset_index(drop=True)

    X_train, X_test, y_train, y_test, _, test_ids = (
        train_test_split(
            X,
            y,
            student_ids,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
    }

    evaluations = []
    prediction_records = []
    best_score = float("-inf")
    best_model = None

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = evaluate_classification(y_test, predictions)

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{model_name}.joblib",
        )

        evaluation = {
            "run_id": run_id,
            "task": "classification",
            "model_name": model_name,
            "metrics": metrics,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "created_at": datetime.now(timezone.utc),
        }
        evaluations.append(evaluation)

        for student_id, actual, predicted in zip(
            test_ids,
            y_test,
            predictions,
        ):
            prediction_records.append(
                {
                    "run_id": run_id,
                    "task": "classification",
                    "model_name": model_name,
                    "student_id": str(student_id),
                    "actual": str(actual),
                    "predicted": str(predicted),
                    "created_at": datetime.now(timezone.utc),
                }
            )

        print(f"\nClassification model: {model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model = pipeline

    if best_model is None:
        raise RuntimeError("No classification model was trained.")

    joblib.dump(
        best_model,
        MODEL_DIR / "best_classification.joblib",
    )

    return evaluations, prediction_records
