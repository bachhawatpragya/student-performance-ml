
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def create_feature_target(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError("The dataset must contain G3.")

    # G3 is the target, never an input feature.
    X = df.drop(
        columns=[TARGET_COLUMN, "student_id"],
        errors="ignore",
    ).copy()

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise",
    )

    # Reject invalid or missing target values.
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    return X, y


def create_performance_category(grades):
    """
    Categories:
      Low:    grade < 10
      Medium: 10 <= grade < 15
      High:   grade >= 15
    """
    return pd.Series(
        grades,
        copy=False,
    ).map(
        lambda grade: (
            "Low" if grade < 10
            else "Medium" if grade < 15
            else "High"
        )
    )


def create_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def split_regression_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def split_classification_data(X, y):
    # Stratification helps preserve category proportions.
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
