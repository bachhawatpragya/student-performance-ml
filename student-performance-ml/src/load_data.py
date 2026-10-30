
from pathlib import Path

import pandas as pd
from pymongo import UpdateOne

from src.db import get_database
from src.download_data import download_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "student-mat.csv"


def read_dataset():
    if not DATA_FILE.exists():
        download_dataset()

    # The original UCI file is semicolon-separated.
    df = pd.read_csv(DATA_FILE, sep=";")

    if "G3" not in df.columns:
        raise ValueError("Dataset must contain the G3 final-grade column.")

    if df.empty:
        raise ValueError("The dataset contains no student records.")

    return df


def load_students():
    df = read_dataset()
    db = get_database()

    operations = []

    for index, row in df.iterrows():
        student_id = f"student-mat-{index:05d}"

        record = row.to_dict()
        record["student_id"] = student_id

        operations.append(
            UpdateOne(
                {"student_id": student_id},
                {"$set": record},
                upsert=True,
            )
        )

    if operations:
        result = db.students.bulk_write(
            operations,
            ordered=False,
        )
        print(
            f"Student records processed: {len(operations)} | "
            f"Inserted: {result.upserted_count} | "
            f"Updated: {result.modified_count}"
        )

    print(f"Total student records in dataset: {len(df)}")
    return len(df)


def load_students_dataframe():
    db = get_database()

    records = list(
        db.students.find({}, {"_id": 0})
    )

    if not records:
        load_students()
        records = list(
            db.students.find({}, {"_id": 0})
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No student records could be loaded.")

    return df


if __name__ == "__main__":
    load_students()
