# Student Performance Prediction & Classification System

**Course / Assessment:** Continuous Assessment 2 (CA2) — MongoDB & Applied Machine Learning  
**Project Directory:** `student-performance-ml`  
**Database:** `student_performance_db` (MongoDB Community Server / PyMongo)  
**Latest Run ID:** `20261029T073103025121Z`  
**Date:** October 2026  

---

## 1. Introduction
This project implements an end-to-end Machine Learning and Database system for predicting student academic performance. Using MongoDB as the persistent storage layer and Scikit-Learn for modeling, the system predicts continuous final grades and classifies academic performance tiers based on demographic, behavioural, and previous academic attributes.

---

## 2. Project Objective
1. **Database Integration:** Ingest and manage student demographic and academic records in MongoDB using PyMongo.
2. **Regression Modeling:** Train a **Linear Regression** model to predict final mathematics grades ($G3$, 0–20 scale) and benchmark against Random Forest.
3. **Classification Modeling:** Train **Logistic Regression** and **Decision Tree** models to classify students into performance tiers (Low, Medium, High).
4. **Evaluation Tracking:** Evaluate models using **$R^2$ Score, MAE, RMSE, Accuracy, Precision, Recall, and F1-Score**, and persist evaluation metrics back into MongoDB.
5. **Interactive Interface:** Deliver a responsive web frontend (Flask) with light/dark theme support, live predictions, and database exploration.

---

## 3. Dataset Description
The system uses the **UCI Student Performance (Mathematics)** dataset (`student-mat.csv`):
- **Total Records:** 395 students.
- **Attributes:** 30 predictor attributes (demographics, family background, weekly study time, school support, failures, and absences) plus 2 interim period grades ($G1$, $G2$).
- **Target Variable (Regression):** Final grade $G3$ (continuous numeric value from 0 to 20).
- **Target Categories (Classification):**
  - **Low Performance:** $G3 < 10$
  - **Medium Performance:** $10 \le G3 < 15$
  - **High Performance:** $G3 \ge 15$

---

## 4. MongoDB / PyMongo Implementation
The system interacts with MongoDB via `pymongo` with automatic connection handling and indexing across three collections:

| Collection Name | Document Count | Purpose & Indexing Strategy |
| :--- | :---: | :--- |
| `students` | 395 | Stores raw student profiles; unique index on `student_id`. |
| `predictions` | 632 | Stores individual test-set predictions; compound unique index on `(run_id, task, model_name, student_id)`. |
| `model_evaluations` | 8+ | Tracks run metrics and parameters; compound unique index on `(run_id, task, model_name)`. |

*Bulk Operations:* Data ingestion uses idempotent `UpdateOne(..., upsert=True)` bulk writes, allowing safe re-runs without duplicate documents.

---

## 5. Data Preprocessing
An 80/20 train-test split was executed using `random_state=42` (316 train records, 79 test records). Preprocessing was encapsulated inside Scikit-Learn `ColumnTransformer` pipelines to avoid data leakage:
- **Numeric Features** (`age`, `absences`, `studytime`, `G1`, `G2`, etc.): Handled missing values using median imputation (`SimpleImputer`) and scaled via `StandardScaler`.
- **Categorical Features** (`school`, `sex`, `address`, `Mjob`, `Fjob`, `guardian`, `schoolsup`, etc.): Imputed with most-frequent strategy and transformed using `OneHotEncoder(handle_unknown='ignore')`.
- Target column $G3$ and identifier `student_id` were dropped from the input feature matrix.

---

## 6. Model Implementation

### 6.1 Regression Models (Continuous Grade Prediction)
- **Linear Regression:** Baseline regressor modeling linear relationships between encoded predictors and final grade $G3$:
  $$\hat{G3} = \beta_0 + \sum_{i=1}^{58} \beta_i X_i \quad (\beta_0 \approx 10.80)$$
- **Random Forest Regressor:** Non-linear ensemble model configured with 150 estimators, `max_depth=10`, and `random_state=42`.

### 6.2 Classification Models (Performance Tiers)
- **Logistic Regression:** Multinomial linear classifier trained with balanced preprocessing.
- **Decision Tree Classifier:** Tree-based rule classifier with stratified sampling preserving class ratios.

---

## 7. Evaluation Results

Models were evaluated on the held-out 20% test dataset ($N=79$):

### 7.1 Regression Evaluation Metrics ($R^2$, MAE, RMSE)
| Model | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | **0.7241** | **1.6467** | **2.3784** |
| **Random Forest Regressor** | **0.8071** | **1.2033** | **1.9886** |

- **Interpretation:** Linear Regression explains **72.41%** of variance in final student grades with an average error of only 1.65 grade points (out of 20). Random Forest further reduces MAE to 1.20 grade points.
- **Key Feature Coefficients:** Second period grade ($G2$, $+0.87$) and first period grade ($G1$, $+0.15$) are the strongest positive predictors; past failures ($-0.32$) and absences are the strongest negative factors.

### 7.2 Classification Evaluation Metrics (Accuracy, Precision, Recall, F1)
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **86.08%** | **0.8600** | **0.8685** | **0.8629** |
| **Decision Tree Classifier** | **84.81%** | **0.8504** | **0.8745** | **0.8610** |

- **Interpretation:** Logistic Regression achieved **86.08% accuracy** and balanced class performance with an **F1-score of 0.8629**.

---

## 8. Conclusion
The project successfully demonstrates an integrated MongoDB and Machine Learning pipeline:
- Ingested and indexed student records seamlessly in MongoDB via PyMongo.
- Trained a **Linear Regression** model achieving a strong **$R^2$ score of 0.7241** and low MAE of 1.6467.
- Classified student tiers with **86.08% accuracy** using Logistic Regression.
- Persisted test predictions and model runs idempotently in MongoDB.
- Deployed a full-stack Flask web application featuring a dynamic theme changer (light/dark), factor correlation charts, and an interactive prediction gauge.
