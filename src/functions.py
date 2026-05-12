import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    f1_score, recall_score, precision_score,
    average_precision_score, confusion_matrix,
)


def load_data(file_path: str) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    return pd.read_csv(file_path)

def compute_metrics(y_true, y_pred, y_proba):
    """Return a dict of classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "BA": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "PRAUC": average_precision_score(y_true, y_proba),
    }


def build_pipeline(estimator, categorical_cols, numerical_cols):
    """Handles different preprocessing for numerical vs categorical columns."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numerical_cols),
            ("cat", cat_pipe, categorical_cols)
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", estimator),
    ])


def bootstrap_median_ci(values, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    boots = [np.median(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lo), float(hi)