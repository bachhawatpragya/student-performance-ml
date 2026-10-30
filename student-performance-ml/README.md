
# Student Performance Prediction System

An academic project combining MongoDB, PyMongo, and machine learning
to predict student final grades and classify academic performance.

## Features

- Imports the UCI Student Performance mathematics dataset.
- Stores student records in MongoDB.
- Predicts final grades using Linear Regression and Random Forest.
- Classifies performance using Logistic Regression and Decision Tree.
- Calculates R2, MAE, accuracy, precision, recall, and F1-score.
- Stores predictions and evaluation results in MongoDB.
- Saves trained models and a JSON evaluation report.

## Requirements

- Python 3.11 or 3.12
- MongoDB Community Server or MongoDB Atlas
- Python dependencies in requirements.txt

## Setup

Create and activate a virtual environment, install dependencies,
and configure the `.env` file using `.env.example`.

Download the dataset:

```powershell
python -m src.download_data
```

Run the complete pipeline:

```powershell
python main.py
```

Run the prediction demonstration after training:

```powershell
python -m src.predict
```

## MongoDB collections

- `students`
- `predictions`
- `model_evaluations`

## Generated outputs

- `models/`: saved ML pipelines
- `results/evaluation_results.json`: model comparison metrics

## Performance categories

- Low: final grade below 10
- Medium: final grade from 10 to below 15
- High: final grade 15 or above

## Evaluation

Regression uses R2 Score and MAE.

Classification uses accuracy, macro precision, macro recall,
and macro F1-score.

The metrics are calculated on held-out test data.
