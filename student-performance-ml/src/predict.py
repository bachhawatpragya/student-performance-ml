
from pathlib import Path

import joblib

from src.db import get_database, close_connection
from src.load_data import load_students_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def predict_student(student_id):
    db = get_database()

    student = db.students.find_one(
        {"student_id": student_id},
        {"_id": 0},
    )

    if student is None:
        raise ValueError(
            f"Student not found: {student_id}"
        )

    import pandas as pd

    # Remove identifiers and the target grade before prediction.
    features = pd.DataFrame(
        [{
            key: value
            for key, value in student.items()
            if key not in {"student_id", "G3"}
        }]
    )

    regression_model = joblib.load(
        MODEL_DIR / "best_regression.joblib"
    )
    classification_model = joblib.load(
        MODEL_DIR / "best_classification.joblib"
    )

    predicted_grade = float(
        regression_model.predict(features)[0]
    )
    predicted_category = str(
        classification_model.predict(features)[0]
    )

    return {
        "student_id": student_id,
        "predicted_final_grade": round(predicted_grade, 2),
        "predicted_performance_category": predicted_category,
    }


if __name__ == "__main__":
    try:
        df = load_students_dataframe()

        if not (
            (MODEL_DIR / "best_regression.joblib").exists()
            and (MODEL_DIR / "best_classification.joblib").exists()
        ):
            raise FileNotFoundError(
                "Train the models first by running: python main.py"
            )

        sample_id = str(df.iloc[0]["student_id"])
        print(predict_student(sample_id))

    finally:
        close_connection()
