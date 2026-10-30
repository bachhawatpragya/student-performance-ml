import json
import os
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

from src.db import get_database, close_connection
from src.load_data import load_students, load_students_dataframe
from src.train_regression import train_regression
from src.train_classification import train_classification

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app)


def get_active_model(model_name="LinearRegression"):
    model_file = MODELS_DIR / f"{model_name}.joblib"
    if not model_file.exists():
        # Fallback to best_regression if needed
        model_file = MODELS_DIR / "best_regression.joblib"
    if not model_file.exists():
        return None
    return joblib.load(model_file)


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    try:
        db = get_database()
        db_ping = db.command("ping")
        student_count = db.students.count_documents({})
        pred_count = db.predictions.count_documents({})
        eval_count = db.model_evaluations.count_documents({})

        linear_model_exists = (MODELS_DIR / "LinearRegression.joblib").exists()

        return jsonify({
            "status": "online",
            "mongodb": {
                "connected": True,
                "database": db.name,
                "collections": {
                    "students": student_count,
                    "predictions": pred_count,
                    "model_evaluations": eval_count,
                }
            },
            "model_ready": linear_model_exists,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "mongodb": {"connected": False, "error": str(e)},
            "model_ready": False,
        }), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    try:
        report_file = RESULTS_DIR / "evaluation_results.json"
        if not report_file.exists():
            return jsonify({"error": "Evaluation report not found. Train models first."}), 404

        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Get latest LinearRegression evaluation from MongoDB
        db = get_database()
        lr_eval = db.model_evaluations.find_one(
            {"model_name": "LinearRegression"},
            {"_id": 0},
            sort=[("created_at", -1)]
        )

        rf_eval = db.model_evaluations.find_one(
            {"model_name": "RandomForestRegressor"},
            {"_id": 0},
            sort=[("created_at", -1)]
        )

        return jsonify({
            "report": data,
            "linear_regression_details": lr_eval,
            "random_forest_details": rf_eval,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart-data", methods=["GET"])
def get_chart_data():
    try:
        db = get_database()
        students = list(db.students.find({}, {"_id": 0}))
        if not students:
            return jsonify({"error": "No students found in MongoDB"}), 404

        df = pd.DataFrame(students)

        # 1. Study time vs Mean G3
        study_stats = df.groupby("studytime")["G3"].agg(["mean", "count"]).reset_index()
        study_data = {
            "labels": [f"{row.studytime} (<{int(row.studytime*2.5)}h/wk)" for _, row in study_stats.iterrows()],
            "values": [round(float(val), 2) for val in study_stats["mean"]],
            "counts": [int(val) for val in study_stats["count"]],
        }

        # 2. Absences vs G3 scatter (sampled 60 points)
        sample_df = df.sample(min(80, len(df)), random_state=42)
        absences_scatter = [
            {"x": int(row["absences"]), "y": float(row["G3"]), "id": row["student_id"]}
            for _, row in sample_df.iterrows()
        ]

        # 3. G1, G2, G3 correlation samples
        grades_scatter = [
            {"g1": float(row["G1"]), "g2": float(row["G2"]), "g3": float(row["G3"]), "id": row["student_id"]}
            for _, row in sample_df.iterrows()
        ]

        # 4. Failures vs G3
        failure_stats = df.groupby("failures")["G3"].mean().reset_index()
        failure_data = {
            "labels": [f"{int(row.failures)} failures" for _, row in failure_stats.iterrows()],
            "values": [round(float(val), 2) for val in failure_stats["G3"]],
        }

        # 5. Actual vs Predicted for LinearRegression test set from predictions collection
        pred_records = list(db.predictions.find(
            {"model_name": "LinearRegression"},
            {"_id": 0, "actual": 1, "predicted": 1, "student_id": 1}
        ).limit(100))

        actual_vs_pred = [
            {"actual": round(p["actual"], 1), "predicted": round(p["predicted"], 2), "id": p["student_id"]}
            for p in pred_records
        ]

        return jsonify({
            "study_data": study_data,
            "absences_scatter": absences_scatter,
            "grades_scatter": grades_scatter,
            "failure_data": failure_data,
            "actual_vs_pred": actual_vs_pred,
            "total_records": len(df),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/students", methods=["GET"])
def get_students():
    try:
        db = get_database()
        limit = min(int(request.args.get("limit", 20)), 100)
        skip = int(request.args.get("skip", 0))
        search = request.args.get("search", "").strip()

        query = {}
        if search:
            query = {
                "$or": [
                    {"student_id": {"$regex": search, "$options": "i"}},
                    {"school": {"$regex": search, "$options": "i"}},
                ]
            }

        total = db.students.count_documents(query)
        cursor = db.students.find(query, {"_id": 0}).skip(skip).limit(limit)
        students = list(cursor)

        return jsonify({
            "total": total,
            "skip": skip,
            "limit": limit,
            "students": students,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing JSON input"}), 400

        # Load Linear Regression pipeline
        lr_pipeline = get_active_model("LinearRegression")
        if lr_pipeline is None:
            return jsonify({"error": "Linear Regression model is not trained yet."}), 500

        # Default template for all 30 features of the UCI dataset
        sample_student = {
            "school": data.get("school", "GP"),
            "sex": data.get("sex", "F"),
            "age": int(data.get("age", 16)),
            "address": data.get("address", "U"),
            "famsize": data.get("famsize", "GT3"),
            "Pstatus": data.get("Pstatus", "T"),
            "Medu": int(data.get("Medu", 3)),
            "Fedu": int(data.get("Fedu", 3)),
            "Mjob": data.get("Mjob", "other"),
            "Fjob": data.get("Fjob", "other"),
            "reason": data.get("reason", "course"),
            "guardian": data.get("guardian", "mother"),
            "traveltime": int(data.get("traveltime", 1)),
            "studytime": int(data.get("studytime", 2)),
            "failures": int(data.get("failures", 0)),
            "schoolsup": data.get("schoolsup", "no"),
            "famsup": data.get("famsup", "yes"),
            "paid": data.get("paid", "no"),
            "activities": data.get("activities", "yes"),
            "nursery": data.get("nursery", "yes"),
            "higher": data.get("higher", "yes"),
            "internet": data.get("internet", "yes"),
            "romantic": data.get("romantic", "no"),
            "famrel": int(data.get("famrel", 4)),
            "freetime": int(data.get("freetime", 3)),
            "goout": int(data.get("goout", 3)),
            "Dalc": int(data.get("Dalc", 1)),
            "Walc": int(data.get("Walc", 1)),
            "health": int(data.get("health", 4)),
            "absences": int(data.get("absences", 4)),
            "G1": float(data.get("G1", 12.0)),
            "G2": float(data.get("G2", 12.0)),
        }

        input_df = pd.DataFrame([sample_student])

        # Predict with Linear Regression
        predicted_grade = float(lr_pipeline.predict(input_df)[0])
        # Clip to valid grade range [0, 20]
        predicted_grade_clamped = max(0.0, min(20.0, predicted_grade))

        # Determine Performance Tier
        if predicted_grade_clamped < 10.0:
            category = "Low"
            category_label = "Low Performance (< 10)"
            category_color = "#ef4444"
        elif predicted_grade_clamped < 15.0:
            category = "Medium"
            category_label = "Medium Performance (10 - 14)"
            category_color = "#f59e0b"
        else:
            category = "High"
            category_label = "High Performance (>= 15)"
            category_color = "#10b981"

        # Also get classification model prediction if available
        clf_prediction = None
        clf_file = MODELS_DIR / "best_classification.joblib"
        if clf_file.exists():
            clf_pipeline = joblib.load(clf_file)
            clf_prediction = str(clf_pipeline.predict(input_df)[0])

        return jsonify({
            "predicted_grade": round(predicted_grade_clamped, 2),
            "raw_grade": round(predicted_grade, 2),
            "category": category,
            "category_label": category_label,
            "category_color": category_color,
            "classifier_tier": clf_prediction,
            "inputs": sample_student,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        db = get_database()
        load_students()
        df = load_students_dataframe()

        reg_evals, reg_preds = train_regression(df, run_id)
        clf_evals, clf_preds = train_classification(df, run_id)

        all_evals = reg_evals + clf_evals
        all_preds = reg_preds + clf_preds

        # Upsert model evaluations
        for eval_record in all_evals:
            db.model_evaluations.update_one(
                {"run_id": run_id, "task": eval_record["task"], "model_name": eval_record["model_name"]},
                {"$set": eval_record},
                upsert=True,
            )

        # Upsert predictions
        for pred_record in all_preds:
            db.predictions.update_one(
                {
                    "run_id": run_id,
                    "task": pred_record["task"],
                    "model_name": pred_record["model_name"],
                    "student_id": pred_record["student_id"],
                },
                {"$set": pred_record},
                upsert=True,
            )

        report = {
            "run_id": run_id,
            "dataset": "UCI Student Performance - Mathematics",
            "student_count": len(df),
            "regression": [
                {"model_name": item["model_name"], "metrics": item["metrics"]}
                for item in reg_evals
            ],
            "classification": [
                {"model_name": item["model_name"], "metrics": item["metrics"]}
                for item in clf_evals
            ],
        }

        with open(RESULTS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return jsonify({
            "success": True,
            "run_id": run_id,
            "message": "Model retrained and evaluated successfully with updated visualizations and MongoDB records.",
            "report": report,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

