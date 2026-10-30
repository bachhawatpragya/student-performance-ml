
import os

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017/",
)

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "student_performance_db",
)

_client = None


def get_database():
    global _client

    if _client is None:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
        )

    # Force a connection check before returning the database.
    _client.admin.command("ping")

    db = _client[MONGODB_DATABASE]

    db.students.create_index(
        [("student_id", ASCENDING)],
        unique=True,
    )

    db.predictions.create_index(
        [
            ("run_id", ASCENDING),
            ("task", ASCENDING),
            ("model_name", ASCENDING),
            ("student_id", ASCENDING),
        ],
        unique=True,
    )

    db.model_evaluations.create_index(
        [
            ("run_id", ASCENDING),
            ("task", ASCENDING),
            ("model_name", ASCENDING),
        ],
        unique=True,
    )

    return db


def close_connection():
    global _client

    if _client is not None:
        _client.close()
        _client = None
