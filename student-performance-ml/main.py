
import json
from datetime import datetime, timezone
from pathlib import Path

from pymongo import UpdateOne

from src.db import get_database, close_connection
from src.load_data import load_students, load_students_dataframe
from src.train_regression import train_regression
from src.train_classification import train_classification

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def save_records(collection, records, key_fields):
    if not records:
        return

    operations = []

    for record in records:
        filter_query = {
            field: record[field]
            for field in key_fields
        }

        operations.append(
            UpdateOne(
                filter_query,
                {"$set": record},
                upsert=True,
            )
        )

    collection.bulk_write(operations, ordered=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    try:
        print("Connecting to MongoDB...")
        db = get_database()
        print(f"Connected to database: {db.name}")

        print("\nLoading dataset into MongoDB...")
        load_students()

        df = load_students_dataframe()
        print(f"Loaded {len(df)} student records from MongoDB.")

        print("\nTraining regression models...")
        regression_evaluations, regression_predictions = (
            train_regression(df, run_id)
        )

        print("\nTraining classification models...")
        classification_evaluations, classification_predictions = (
            train_classification(df, run_id)
        )

        all_evaluations = (
            regression_evaluations
            + classification_evaluations
        )

        all_predictions = (
            regression_predictions
            + classification_predictions
        )

        save_records(
            db.model_evaluations,
            all_evaluations,
            ["run_id", "task", "model_name"],
        )

        save_records(
            db.predictions,
            all_predictions,
            ["run_id", "task", "model_name", "student_id"],
        )

        report = {
            "run_id": run_id,
            "dataset": "UCI Student Performance - Mathematics",
            "student_count": len(df),
            "classification_categories": {
                "Low": "G3 < 10",
                "Medium": "10 <= G3 < 15",
                "High": "G3 >= 15",
            },
            "regression": [
                {
                    "model_name": item["model_name"],
                    "metrics": item["metrics"],
                }
                for item in regression_evaluations
            ],
            "classification": [
                {
                    "model_name": item["model_name"],
                    "metrics": item["metrics"],
                }
                for item in classification_evaluations
            ],
        }

        report_path = RESULTS_DIR / "evaluation_results.json"

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\nPipeline completed successfully.")
        print(f"Evaluation report: {report_path}")
        print(f"Models saved in: {PROJECT_ROOT / 'models'}")
        print(f"MongoDB database: {db.name}")
        print(f"Run ID: {run_id}")

    finally:
        close_connection()


if __name__ == "__main__":
    main()
