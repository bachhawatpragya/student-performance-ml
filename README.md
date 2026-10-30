# Student Performance Prediction & Classification System

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Community%20Server-green?logo=mongodb&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end Machine Learning and Database project integrating **MongoDB (PyMongo)** with **Scikit-Learn** pipelines and an interactive **Flask Web UI** to predict continuous student final grades and classify academic performance tiers.

---

## 📌 Project Overview & Objectives

- **Database Management:** Store, index, and query 395 UCI student records in MongoDB with unique identifiers.
- **Regression Modeling:** Train a **Linear Regression** model ($R^2 = 0.7241$) to predict final mathematics grades ($G3$, range 0–20) alongside a Random Forest benchmark.
- **Classification Modeling:** Categorize students into performance tiers (**Low**, **Medium**, **High**) with **Logistic Regression** ($86.08\%$ accuracy) and Decision Tree.
- **Evaluation Tracking:** Persist all run predictions, train/test distributions, and evaluation metrics back into MongoDB collections.
- **Interactive Web Dashboard:** A responsive Flask web application with a Light/Dark theme toggle, factor impact charts, live prediction radial gauge, and MongoDB explorer.

---

## 📊 Model Evaluation Results

### 1. Regression Models (Final Grade Prediction: 0–20 Scale)
Evaluated on the held-out 20% test dataset ($N = 79$):

| Model | $R^2$ Score (Variance Explained) | MAE (Mean Absolute Error) | RMSE | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | **0.7241** (72.4%) | **1.6467** grade pts | **2.3784** | **Target Regressor** |
| **Random Forest Regressor** | **0.8071** (80.7%) | **1.2033** grade pts | **1.9886** | Benchmark Regressor |

### 2. Classification Models (Academic Tiers: Low / Medium / High)
- **Low:** $G3 < 10$
- **Medium:** $10 \le G3 < 15$
- **High:** $G3 \ge 15$

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **86.08%** | **0.8600** | **0.8685** | **0.8629** | **Selected Best Classifier** |
| **Decision Tree Classifier** | **84.81%** | **0.8504** | **0.8745** | **0.8610** | Rule-Based Classifier |

---

## 🗄️ MongoDB Architecture

The project connects to MongoDB via `pymongo` and manages three primary collections:

```text
student_performance_db
├── students           (395 documents: raw demographics, study habits, grades)
├── predictions        (632 documents: test set actual vs predicted values)
└── model_evaluations  (8+ documents: R², MAE, RMSE, accuracy, F1, run timestamps)
```

- **Idempotent Ingestion:** Uses `UpdateOne(..., upsert=True)` bulk writes to prevent duplicate data insertion.
- **Compound Unique Indexes:** Indexed on `(run_id, task, model_name, student_id)` to ensure integrity across runs.

---

## 📁 Repository Structure

```text
mongo_CA2/
├── .gitignore                         # Configured ignores for virtualenvs, caches, secrets
├── README.md                          # Comprehensive project documentation
├── report/
│   └── PROJECT_REPORT.md              # Detailed CA2 academic report
├── screenshots/                       # Output captures and visual evidence
│   ├── 01_mongodb_connection_and_main_output.png
│   ├── 02_regression_metrics.png
│   ├── 03_classification_metrics.png
│   ├── 04_mongodb_students_compass.png
│   ├── 05_mongodb_predictions_compass.png
│   ├── 06_predict_cli_output.png
│   ├── 07_evaluation_results_json.png
│   ├── actual_vs_predicted.png
│   ├── feature_impacts.png
│   └── linear_coefficients.png
└── student-performance-ml/            # Application & pipeline package
    ├── .env.example                   # Environment configuration template
    ├── requirements.txt               # Pinned project dependencies
    ├── main.py                        # Complete end-to-end training pipeline
    ├── server.py                      # Flask API and Web Dashboard server
    ├── data/
    │   └── student-mat.csv            # UCI Mathematics cohort dataset
    ├── models/                        # Serialized Scikit-Learn pipelines (.joblib)
    ├── results/
    │   └── evaluation_results.json    # JSON report of all evaluated models
    ├── src/
    │   ├── db.py                      # PyMongo connection & collection indexing
    │   ├── download_data.py           # Automated dataset downloader
    │   ├── evaluate.py                # Regression & classification metric functions
    │   ├── load_data.py               # MongoDB dataset ingestion & loading
    │   ├── predict.py                 # CLI single-student inference script
    │   ├── preprocess.py              # ColumnTransformer & feature engineering
    │   ├── train_classification.py    # LogisticRegression & DecisionTree trainer
    │   ├── train_regression.py        # LinearRegression & RandomForest trainer
    │   └── visualize.py               # Matplotlib analysis plot generator
    └── static/                        # Frontend Web Dashboard assets
        ├── index.html                 # Modern glassmorphism UI & tabs
        ├── css/
        │   └── style.css              # Custom styling + Light/Dark mode themes
        └── js/
            └── app.js                 # Chart.js charts, gauge & MongoDB explorer
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11 or 3.12
- MongoDB Community Server running locally on port `27017` (or MongoDB Atlas connection URI)

### 2. Environment Setup
```powershell
cd student-performance-ml

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run Pipeline (Ingest Data, Train & Evaluate Models)
```powershell
python main.py
```

### 4. Launch Web Application Dashboard
```powershell
python server.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser to access:
- **Model Overview:** $R^2$, MAE, and side-by-side model comparison cards.
- **Visualizations:** Interactive Chart.js plots and factor impacts.
- **Interactive Predictor:** Live student grade calculation with radial score gauge.
- **MongoDB Explorer:** Live paginated documents viewer with JSON inspector.
- **Theme Switcher:** Toggle between Dark and Light mode.

---

## 📜 License
This project is open-source under the MIT License.
