
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import (
    create_feature_target,
    create_preprocessor,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.evaluate import evaluate_regression, to_python_scalar
from src.visualize import (
    plot_actual_vs_predicted,
    plot_feature_impacts,
    plot_linear_coefficients,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def train_regression(df, run_id):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = create_feature_target(df)
    student_ids = df["student_id"].reset_index(drop=True)

    X_train, X_test, y_train, y_test, _, test_ids = (
        train_test_split(
            X,
            y,
            student_ids,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    evaluations = []
    prediction_records = []
    best_score = float("-inf")
    best_model = None

    # Generate feature impacts plot from full dataset
    plot_feature_impacts(df)

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = evaluate_regression(y_test, predictions)

        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)

        evaluation = {
            "run_id": run_id,
            "task": "regression",
            "model_name": model_name,
            "metrics": metrics,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "created_at": datetime.now(timezone.utc),
        }

        # For LinearRegression, extract coefficients and generate visualizations
        if model_name == "LinearRegression":
            preprocessor = pipeline.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
            clean_names = [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names]
            linear_model = pipeline.named_steps["model"]
            coefs = linear_model.coef_.tolist()
            intercept = float(linear_model.intercept_)

            evaluation["intercept"] = intercept
            coef_list = [
                {"feature": f, "coefficient": float(c)}
                for f, c in zip(clean_names, coefs)
            ]
            coef_list.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
            evaluation["top_coefficients"] = coef_list[:15]
            evaluation["all_coefficients"] = coef_list

            # Plot actual vs predicted for Linear Regression
            plot_actual_vs_predicted(
                y_test.values,
                predictions,
                r2=metrics["r2_score"],
                mae=metrics["mae"],
                model_name="Linear Regression",
                output_filename="actual_vs_predicted.png",
            )

            # Plot coefficients
            plot_linear_coefficients(
                clean_names,
                coefs,
                intercept,
                output_filename="linear_coefficients.png",
                top_n=12,
            )

        evaluations.append(evaluation)

        for student_id, actual, predicted in zip(
            test_ids,
            y_test,
            predictions,
        ):
            prediction_records.append(
                {
                    "run_id": run_id,
                    "task": "regression",
                    "model_name": model_name,
                    "student_id": str(student_id),
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "created_at": datetime.now(timezone.utc),
                }
            )

        print(f"\nRegression model: {model_name}")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics.get('rmse', 0):.4f}")

        if metrics["r2_score"] > best_score:
            best_score = metrics["r2_score"]
            best_model = pipeline

    if best_model is None:
        raise RuntimeError("No regression model was trained.")

    joblib.dump(
        best_model,
        MODEL_DIR / "best_regression.joblib",
    )

    return evaluations, prediction_records

