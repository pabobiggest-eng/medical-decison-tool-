"""Single-file, leakage-aware diabetes classification workflow.

Run from this folder:
    python diabetes_ai_project.py
    python diabetes_ai_project.py --predict  (after training)

Input:
    diabetes_prediction_dataset.csv

Outputs are written to results/ and graphs/.
The model is an educational decision-support prototype, not a diagnostic device.
"""
from __future__ import annotations

import json
import warnings
import sys
from pathlib import Path
from itertools import product

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------------------------------------------------------
# SHARED FEATURE DEFINITIONS (inlined so this is a SINGLE-FILE project)
# -----------------------------------------------------------------------------
RAW_FEATURES = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi"
]
ENGINEERED_FEATURES = [
    "log_bmi", "comorbidity_count", "obese", "age_x_bmi", "age_group"
]
NUMERIC_FEATURES = [
    "age", "bmi", "log_bmi", "hypertension", "heart_disease",
    "comorbidity_count", "obese", "age_x_bmi"
]
CATEGORICAL_FEATURES = ["gender", "smoking_history", "age_group"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_DIAGNOSTIC_FEATURES = ["HbA1c_level", "blood_glucose_level"]
TARGET = "diabetes"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create row-level features only; never reads target or dataset-wide stats."""
    missing = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required raw features: {missing}")
    out = frame.loc[:, RAW_FEATURES].copy()
    bmi = pd.to_numeric(out["bmi"], errors="coerce")
    age = pd.to_numeric(out["age"], errors="coerce")
    out["log_bmi"] = np.log1p(bmi.clip(lower=0))
    out["comorbidity_count"] = (
        pd.to_numeric(out["hypertension"], errors="coerce")
        + pd.to_numeric(out["heart_disease"], errors="coerce")
    )
    out["obese"] = np.where(bmi.notna(), (bmi >= 30).astype(float), np.nan)
    out["age_x_bmi"] = age * bmi / 100.0
    out["age_group"] = pd.cut(
        age, bins=[-np.inf, 29, 44, 59, np.inf],
        labels=["under_30", "30_44", "45_59", "60_plus"], right=True
    ).astype("object")
    return out.loc[:, MODEL_FEATURES]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer: training and prediction use identical features."""
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURES)
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Input is missing required raw features: {missing}")
        self.n_features_in_ = len(RAW_FEATURES)
        self.feature_names_in_ = np.asarray(RAW_FEATURES, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURES)
        return add_features(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MODEL_FEATURES, dtype=object)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
DATA_FILE = Path(__file__).resolve().parent / "diabetes_prediction_dataset.csv"
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"
RESULTS_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def log(message: str) -> None:
    print(f"\n{'=' * 78}\n{message}\n{'=' * 78}", flush=True)


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def build_pipeline(classifier) -> Pipeline:
    """All learned preprocessing is inside the CV-fold-fitted pipeline."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessing", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "positive_rate_actual": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(predictions)),
        "majority_class_accuracy_baseline": float(max(np.mean(y_true), 1 - np.mean(y_true))),
        "accuracy_minus_majority_baseline": float(
            accuracy_score(y_true, predictions) - max(np.mean(y_true), 1 - np.mean(y_true))
        ),
    }


def validate_and_clean(raw: pd.DataFrame):
    required = set(RAW_FEATURES + EXCLUDED_DIAGNOSTIC_FEATURES + [TARGET])
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = raw.copy()
    raw_rows, raw_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    missing_before = int(df.isna().sum().sum())

    # Normalize whitespace without changing the label or inventing values.
    for col in ["gender", "smoking_history"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Numeric conversion is explicit; invalid text becomes missing and is later
    # handled inside the training-only imputation pipeline.
    numeric_cols = [
        "age", "hypertension", "heart_disease", "bmi",
        "HbA1c_level", "blood_glucose_level", TARGET,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[TARGET].isna().any():
        raise ValueError("Target 'diabetes' contains missing/non-numeric values.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target 'diabetes' must contain only 0 and 1.")

    # Domain validation: flag impossible/unsupported values instead of silently
    # clipping them or deleting rows. Missing numeric values may be imputed later
    # inside the training-only pipeline.
    domain_checks = []

    def record_domain_check(column, rule, invalid_mask):
        invalid_count = int(invalid_mask.fillna(False).sum())
        domain_checks.append({
            "column": column,
            "expected_rule": rule,
            "invalid_count": invalid_count,
            "status": "PASS" if invalid_count == 0 else "FAIL",
        })

    record_domain_check("age", "0 <= age <= 120 years",
                        df["age"].notna() & ~df["age"].between(0, 120))
    record_domain_check("bmi", "0 < BMI <= 100",
                        df["bmi"].notna() & ~df["bmi"].between(0, 100, inclusive="right"))
    record_domain_check("HbA1c_level", "0 < HbA1c <= 20 (audit-only field)",
                        df["HbA1c_level"].notna() & ~df["HbA1c_level"].between(0, 20, inclusive="right"))
    record_domain_check("blood_glucose_level", "0 < glucose <= 1000 (audit-only field)",
                        df["blood_glucose_level"].notna() & ~df["blood_glucose_level"].between(0, 1000, inclusive="right"))
    for col in ["hypertension", "heart_disease"]:
        record_domain_check(col, "binary values {0, 1}",
                            df[col].notna() & ~df[col].isin([0, 1]))
    record_domain_check("gender", "Female, Male or Other",
                        df["gender"].notna() & ~df["gender"].isin(["Female", "Male", "Other"]))
    record_domain_check(
        "smoking_history",
        "never, former, current, not current, ever or No Info",
        df["smoking_history"].notna() & ~df["smoking_history"].isin(
            ["never", "former", "current", "not current", "ever", "No Info"]
        ),
    )
    failed_checks = [item for item in domain_checks if item["status"] == "FAIL"]
    if failed_checks:
        raise ValueError(f"Domain validation failed. Inspect these checks: {failed_checks}")

    # Exact duplicate rows are removed before the split to avoid train/test copies.
    df = df.drop_duplicates().reset_index(drop=True)

    # Do not delete rare 'Other' gender rows; they are retained and rare categories
    # are grouped by OneHotEncoder(min_frequency=20), fitted on training folds.
    if df["gender"].isna().all() or df["smoking_history"].isna().all():
        raise ValueError("A required categorical input column is entirely missing.")

    audit = {
        "raw_rows": int(raw_rows),
        "raw_columns": int(raw_cols),
        "exact_duplicate_rows_removed": duplicate_rows,
        "rows_after_duplicate_removal": int(len(df)),
        "missing_cells_before_pipeline_imputation": int(df.isna().sum().sum()),
        "missing_cells_in_original_file": missing_before,
        "gender_other_rows_retained": int((df["gender"] == "Other").sum()),
        "smoking_no_info_rows_retained": int((df["smoking_history"] == "No Info").sum()),
        "rows_removed_for_missing_target": 0,
        "domain_validation": domain_checks,
    }
    return df, audit


def create_eda(df: pd.DataFrame, audit: dict) -> None:
    log("STEP 2 - DATA QUALITY AND EDA")
    print("Cleaned shape:", df.shape)
    print("Target counts:\n", df[TARGET].value_counts().sort_index().to_string())
    print("Missing cells after duplicate removal:", int(df.isna().sum().sum()))

    quality_rows = []
    for col in df.columns:
        quality_rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_percent": float(df[col].isna().mean() * 100),
                "unique_values": int(df[col].nunique(dropna=True)),
            }
        )
    pd.DataFrame(quality_rows).to_csv(RESULTS_DIR / "data_quality_audit.csv", index=False)
    pd.DataFrame(audit.get("domain_validation", [])).to_csv(
        RESULTS_DIR / "domain_validation_audit.csv", index=False
    )

    class_counts = df[TARGET].value_counts().sort_index()
    class_table = pd.DataFrame(
        {
            "class": [0, 1],
            "label": ["No diabetes", "Diabetes"],
            "count": [int(class_counts.get(0, 0)), int(class_counts.get(1, 0))],
        }
    )
    class_table["percentage"] = class_table["count"] / len(df) * 100
    class_table.to_csv(RESULTS_DIR / "class_distribution.csv", index=False)
    majority = int(class_counts.max())
    minority = int(class_counts.min())
    imbalance_ratio = majority / minority if minority else None

    # Numeric statistical profile, including distribution shape.
    numeric_eda = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    stats = df[numeric_eda].describe().T
    stats["skewness"] = df[numeric_eda].apply(lambda s: skew(s.dropna(), bias=False))
    stats["kurtosis_excess"] = df[numeric_eda].apply(lambda s: kurtosis(s.dropna(), fisher=True, bias=False))
    stats.to_csv(RESULTS_DIR / "eda_numeric_summary.csv", float_format="%.5f")

    # BMI transform assessment. Log BMI is used as an additional feature,
    # not as a replacement for original BMI.
    log_bmi = np.log1p(df["bmi"].clip(lower=0))
    skew_table = pd.DataFrame(
        [
            {"feature": "bmi", "skewness": float(skew(df["bmi"], bias=False)),
             "kurtosis_excess": float(kurtosis(df["bmi"], fisher=True, bias=False))},
            {"feature": "log_bmi", "skewness": float(skew(log_bmi, bias=False)),
             "kurtosis_excess": float(kurtosis(log_bmi, fisher=True, bias=False))},
        ]
    )
    skew_table.to_csv(RESULTS_DIR / "skewness_kurtosis.csv", index=False)

    # IQR flags are reported, not automatically deleted: unusual BMI can be real.
    outlier_rows = []
    for col in numeric_eda:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_rows.append(
            {"feature": col, "q1": q1, "q3": q3, "iqr": iqr,
             "lower_fence": lower, "upper_fence": upper,
             "outlier_count": count, "outlier_percent": count / len(df) * 100,
             "action": "reported_only_not_removed"}
        )
    pd.DataFrame(outlier_rows).to_csv(RESULTS_DIR / "outlier_summary.csv", index=False)

    # Correlations are restricted to numeric fields; nominal categories are not
    # assigned arbitrary integer codes for correlation.
    corr_cols = ["age", "bmi", "hypertension", "heart_disease",
                 "HbA1c_level", "blood_glucose_level", TARGET]
    corr = df[corr_cols].corr(method="spearman")
    corr.to_csv(RESULTS_DIR / "spearman_correlation.csv", float_format="%.5f")

    # Explicit leakage audit: diagnostic measurements are measured against label
    # but are prohibited from model features.
    leakage_tables = []
    for col, bins, labels in [
        ("HbA1c_level", [-np.inf, 5.6, 7.0, np.inf], ["<=5.6", "5.6-7.0", ">7.0"]),
        ("blood_glucose_level", [-np.inf, 100, 200, np.inf], ["<=100", "100-200", ">200"]),
    ]:
        bucket = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        group = df.assign(_band=bucket).groupby("_band", observed=False)[TARGET].agg(
            patients="count", diabetes_rate="mean"
        ).reset_index()
        group.insert(0, "diagnostic_feature", col)
        group.rename(columns={"_band": "value_band"}, inplace=True)
        group["model_input"] = False
        leakage_tables.append(group)
    leakage_df = pd.concat(leakage_tables, ignore_index=True)
    leakage_df.to_csv(RESULTS_DIR / "leakage_band_analysis.csv", index=False)

    feature_decisions = pd.DataFrame(
        [
            {"feature": "gender", "decision": "include", "reason": "Demographic input; rare category retained and grouped if infrequent."},
            {"feature": "age", "decision": "include", "reason": "Core demographic risk factor."},
            {"feature": "hypertension", "decision": "include", "reason": "Recorded comorbidity."},
            {"feature": "heart_disease", "decision": "include", "reason": "Recorded comorbidity."},
            {"feature": "smoking_history", "decision": "include", "reason": "Medical/lifestyle history; 'No Info' kept as explicit category."},
            {"feature": "bmi", "decision": "include", "reason": "Body-mass indicator; extremes are flagged, not automatically deleted."},
            {"feature": "log_bmi", "decision": "engineer", "reason": "Row-level transform to represent right-skewed BMI pattern."},
            {"feature": "comorbidity_count", "decision": "engineer", "reason": "Sum of hypertension and heart-disease flags (0-2)."},
            {"feature": "obese", "decision": "engineer", "reason": "BMI >= 30 indicator; adds an interpretable threshold feature."},
            {"feature": "age_x_bmi", "decision": "engineer", "reason": "Row-level age/BMI interaction to capture combined pattern."},
            {"feature": "age_group", "decision": "engineer", "reason": "Age bands under 30, 30-44, 45-59 and 60+; raw age is also retained."},
            {"feature": "HbA1c_level", "decision": "exclude", "reason": "Diagnostic biomarker is strongly label-separating; excluded to avoid diagnostic target leakage."},
            {"feature": "blood_glucose_level", "decision": "exclude", "reason": "Diagnostic biomarker is strongly label-separating; excluded to avoid diagnostic target leakage."},
            {"feature": "diabetes", "decision": "target_only", "reason": "Outcome label; never used as an input."},
        ]
    )
    feature_decisions.to_csv(RESULTS_DIR / "feature_decisions.csv", index=False)

    # Graph 1: class distribution.
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET, hue=TARGET, legend=False, palette=["#4776A8", "#D65F5F"])
    for bar in ax.patches:
        ax.annotate(f"{int(bar.get_height()):,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1], ["No diabetes (0)", "Diabetes (1)"])
    ax.set_title(f"Target class distribution | ratio {imbalance_ratio:.2f}:1")
    ax.set_xlabel("")
    ax.set_ylabel("Patient records")
    save_plot("01_class_imbalance.png")

    # Graph 2: numeric distributions and reported skew/kurtosis.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), numeric_eda):
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color="steelblue")
        ax.set_title(f"{col} | skew={skew(df[col], bias=False):.2f}, kurt={kurtosis(df[col], fisher=True, bias=False):.2f}")
    save_plot("02_numeric_distributions.png")

    # Graph 3: outlier boxplots.
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, col in zip(axes, numeric_eda):
        sns.boxplot(y=df[col], ax=ax, color="orange")
        ax.set_title(col)
    save_plot("03_boxplots_outliers.png")

    # Graph 4: numeric Spearman correlation.
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Spearman correlation (numeric variables only)")
    save_plot("04_spearman_correlation.png")

    # Graph 5: target rate by grouped age and BMI.
    grouped = df.copy()
    grouped["age_group_eda"] = pd.cut(
        grouped["age"], [-np.inf, 29, 44, 59, np.inf],
        labels=["under 30", "30-44", "45-59", "60+"],
    )
    grouped["bmi_group_eda"] = pd.cut(
        grouped["bmi"], [0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        include_lowest=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    age_rate = grouped.groupby("age_group_eda", observed=False)[TARGET].mean() * 100
    bmi_rate = grouped.groupby("bmi_group_eda", observed=False)[TARGET].mean() * 100
    age_rate.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Diabetes share by age group")
    axes[0].set_ylabel("Diabetes records (%)")
    bmi_rate.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Diabetes share by BMI group")
    axes[1].set_ylabel("Diabetes records (%)")
    save_plot("05_diabetes_rate_by_groups.png")

    # Graph 6: label rate across diagnostic bands; explanatory evidence only.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["HbA1c_level", "blood_glucose_level"]):
        sub = leakage_df[leakage_df["diagnostic_feature"] == col]
        sns.barplot(data=sub, x="value_band", y="diabetes_rate", ax=ax, color="steelblue")
        ax.set_title(f"Label rate by {col} band (EXCLUDED from model)")
        ax.set_ylabel("Diabetes share")
        ax.set_xlabel("Value band")
    save_plot("06_diagnostic_leakage_evidence.png")

    clean_info = {
        **audit,
        "clean_rows": int(len(df)),
        "clean_columns": int(df.shape[1]),
        "class_0_count": int(class_counts.get(0, 0)),
        "class_1_count": int(class_counts.get(1, 0)),
        "class_0_percent": float(class_counts.get(0, 0) / len(df) * 100),
        "class_1_percent": float(class_counts.get(1, 0) / len(df) * 100),
        "class_imbalance_ratio_majority_to_minority": float(imbalance_ratio),
        "raw_model_inputs": RAW_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "diagnostic_features_excluded": EXCLUDED_DIAGNOSTIC_FEATURES,
        "missing_handling": "Missing values, if encountered, are imputed inside CV-fitted pipelines; no full-data imputation.",
        "outlier_handling": "IQR outliers reported but not automatically removed.",
        "imbalance_handling": "class_weight='balanced' or 'balanced_subsample' for supported models; no pre-split resampling.",
    }
    with open(RESULTS_DIR / "data_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(clean_info, f, indent=2)

    print("Class ratio (majority/minority): %.2f:1" % imbalance_ratio)
    print("Class 1 share after cleaning: %.2f%%" % (clean_info["class_1_percent"]))
    print("EDA tables and graphs saved.")
    return clean_info


def main() -> None:
    log("STEP 1 - LOAD DATA")
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE.name}. Place it in the same folder as train_model.py."
        )
    raw = pd.read_csv(DATA_FILE, low_memory=False)
    print("Dataset file:", DATA_FILE.name)
    print("Raw shape:", raw.shape)
    print("Raw columns:", raw.columns.tolist())

    log("STEP 2 - CLEANING AND EDA")
    df, audit = validate_and_clean(raw)
    data_info = create_eda(df, audit)

    # The two lab diagnostics and the target are not in X by construction.
    X = df.loc[:, RAW_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    assert not set(EXCLUDED_DIAGNOSTIC_FEATURES).intersection(X.columns)
    assert TARGET not in X.columns

    log("STEP 3 - STRATIFIED TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    split_table = pd.DataFrame(
        [
            {"split": "train", "rows": len(y_train), "diabetes_0": int((y_train == 0).sum()),
             "diabetes_1": int((y_train == 1).sum()), "diabetes_share_percent": float(y_train.mean() * 100)},
            {"split": "test", "rows": len(y_test), "diabetes_0": int((y_test == 0).sum()),
             "diabetes_1": int((y_test == 1).sum()), "diabetes_share_percent": float(y_test.mean() * 100)},
        ]
    )
    split_table.to_csv(RESULTS_DIR / "train_test_split_summary.csv", index=False)
    print(split_table.to_string(index=False))

    log("STEP 4 - MODEL COMPARISON (STRATIFIED CV ON TRAIN ONLY)")
    models = {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=12, min_samples_leaf=25,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.07, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=2.0,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }
    comparison_rows = []
    fitted_candidates = {}
    for name, estimator in models.items():
        print(f"Cross-validating: {name}", flush=True)
        pipe = build_pipeline(estimator)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=1, return_train_score=False, error_score="raise",
        )
        row = {"model": name}
        for metric in scoring:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1))
        comparison_rows.append(row)
        fitted_candidates[name] = pipe
        print(
            f"  PR-AUC={row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f} | "
            f"ROC-AUC={row['roc_auc_mean']:.4f} | F1={row['f1_mean']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("pr_auc_mean", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison_cv.csv", index=False)
    non_baseline = comparison[comparison["model"] != "Dummy baseline"]
    best_name = str(non_baseline.iloc[0]["model"])
    best_base_pipe = fitted_candidates[best_name]
    print("Selected candidate by mean CV PR-AUC:", best_name)

    # Train-only, compact hyperparameter search for the selected candidate.
    log("STEP 5 - HYPERPARAMETER TUNING (TRAIN ONLY)")
    param_spaces = {
        "Logistic Regression": {"classifier__C": [0.05, 0.1, 0.5, 1.0, 3.0]},
        "Decision Tree": {
            "classifier__max_depth": [4, 5, 6, 8],
            "classifier__min_samples_leaf": [30, 50, 80],
        },
        "Random Forest": {
            "classifier__n_estimators": [150, 180, 220],
            "classifier__max_depth": [8, 10, 12],
            "classifier__min_samples_leaf": [20, 25, 40],
        },
        "HistGradientBoosting": {
            "classifier__max_iter": [100, 140, 180],
            "classifier__learning_rate": [0.04, 0.06, 0.08],
            "classifier__max_leaf_nodes": [15, 23, 31],
            "classifier__l2_regularization": [1.0, 3.0, 5.0],
            "classifier__min_samples_leaf": [30, 40, 60],
        },
    }
    search = RandomizedSearchCV(
        estimator=best_base_pipe,
        param_distributions=param_spaces[best_name],
        n_iter=3,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    final_model = search.best_estimator_
    pd.DataFrame(search.cv_results_).to_csv(RESULTS_DIR / "hyperparameter_search_results.csv", index=False)
    print("Best parameters:", search.best_params_)
    print("Best tuning CV PR-AUC: %.4f" % search.best_score_)

    log("STEP 6 - THRESHOLD SELECTION USING TRAIN OOF PREDICTIONS")
    oof_probabilities = cross_val_predict(
        final_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    oof_precision, oof_recall, oof_thresholds = precision_recall_curve(y_train, oof_probabilities)
    if len(oof_thresholds):
        oof_f1 = (
            2 * oof_precision[:-1] * oof_recall[:-1]
            / (oof_precision[:-1] + oof_recall[:-1] + 1e-12)
        )
        best_idx = int(np.nanargmax(oof_f1))
        threshold_f1 = float(oof_thresholds[best_idx])
    else:
        threshold_f1 = 0.5

    # A second, explicitly labelled operating point trades some sensitivity
    # for higher accuracy. It is selected ONLY from out-of-fold TRAIN scores.
    # The recall floor prevents the trivial all-negative solution.
    MIN_RECALL_FOR_ACCURACY_POINT = 0.30
    threshold_grid = np.unique(np.r_[np.linspace(0.05, 0.99, 189), threshold_f1])
    threshold_rows = []
    for candidate_threshold in threshold_grid:
        candidate_pred = (oof_probabilities >= candidate_threshold).astype(int)
        candidate_recall = recall_score(y_train, candidate_pred, zero_division=0)
        candidate_accuracy = accuracy_score(y_train, candidate_pred)
        candidate_precision = precision_score(y_train, candidate_pred, zero_division=0)
        candidate_f1 = f1_score(y_train, candidate_pred, zero_division=0)
        threshold_rows.append({
            "threshold": float(candidate_threshold),
            "oof_accuracy": float(candidate_accuracy),
            "oof_precision": float(candidate_precision),
            "oof_recall": float(candidate_recall),
            "oof_f1": float(candidate_f1),
            "meets_minimum_recall": bool(candidate_recall >= MIN_RECALL_FOR_ACCURACY_POINT),
        })
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(RESULTS_DIR / "threshold_tradeoff_oof_train_only.csv", index=False)
    eligible = threshold_table[threshold_table["meets_minimum_recall"]]
    if eligible.empty:
        threshold_accuracy = threshold_f1
        print("WARNING: no threshold met the recall floor; using F1 threshold.")
    else:
        # Prefer accuracy; ties prefer higher recall, then higher F1.
        chosen = eligible.sort_values(
            ["oof_accuracy", "oof_recall", "oof_f1"], ascending=[False, False, False]
        ).iloc[0]
        threshold_accuracy = float(chosen["threshold"])

    print("OOF threshold maximizing F1:", round(threshold_f1, 5))
    print(
        f"OOF accuracy-oriented threshold (recall >= {MIN_RECALL_FOR_ACCURACY_POINT:.0%}):",
        round(threshold_accuracy, 5),
    )
    threshold = threshold_f1  # primary threshold continues to prioritize F1

    log("STEP 7 - FINAL FIT AND ONE-TIME HELD-OUT TEST EVALUATION")
    final_model.fit(X_train, y_train)
    encoded_feature_count = len(final_model.named_steps["preprocessing"].get_feature_names_out())
    train_probabilities = final_model.predict_proba(X_train)[:, 1]
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    train_metrics = calculate_metrics(y_train, train_probabilities, threshold_f1)
    test_metrics = calculate_metrics(y_test, test_probabilities, threshold_f1)
    train_metrics_accuracy_point = calculate_metrics(y_train, train_probabilities, threshold_accuracy)
    test_metrics_accuracy_point = calculate_metrics(y_test, test_probabilities, threshold_accuracy)
    print("Final model:", best_name)
    print("PRIMARY operating point (F1 threshold) - train:", json.dumps(train_metrics, indent=2))
    print("PRIMARY operating point (F1 threshold) - held-out test:", json.dumps(test_metrics, indent=2))
    print("SECONDARY operating point (accuracy threshold) - held-out test:",
          json.dumps(test_metrics_accuracy_point, indent=2))

    report = classification_report(
        y_test,
        (test_probabilities >= threshold).astype(int),
        labels=[0, 1],
        target_names=["No diabetes", "Diabetes"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(RESULTS_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, (test_probabilities >= threshold_f1).astype(int), labels=[0, 1])
    pd.DataFrame(
        cm, index=["actual_no_diabetes", "actual_diabetes"],
        columns=["predicted_no_diabetes", "predicted_diabetes"],
    ).to_csv(RESULTS_DIR / "confusion_matrix_f1_threshold.csv")
    cm_accuracy = confusion_matrix(
        y_test, (test_probabilities >= threshold_accuracy).astype(int), labels=[0, 1]
    )
    pd.DataFrame(
        cm_accuracy, index=["actual_no_diabetes", "actual_diabetes"],
        columns=["predicted_no_diabetes", "predicted_diabetes"],
    ).to_csv(RESULTS_DIR / "confusion_matrix_accuracy_threshold.csv")

    predictions = pd.DataFrame(
        {
            "test_row_index": X_test.index,
            "actual_diabetes": y_test.to_numpy(),
            "model_score": test_probabilities,
            "predicted_diabetes_at_selected_threshold": (test_probabilities >= threshold).astype(int),
        }
    )
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    gap_metrics = []
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "f1", "roc_auc", "pr_auc_average_precision"]:
        gap_metrics.append(
            {
                "metric": metric,
                "train_in_sample": train_metrics[metric],
                "test_held_out": test_metrics[metric],
                "absolute_gap": abs(train_metrics[metric] - test_metrics[metric]),
            }
        )
    pd.DataFrame(gap_metrics).to_csv(RESULTS_DIR / "train_test_overfit_gap.csv", index=False)

    # Test set is not used to select/tune the model. Permutation importance is
    # calculated only after the final model and metrics are fixed, for reporting.
    log("STEP 8 - INTERPRETABILITY AND FINAL GRAPHS")
    importance_sample_n = min(2500, len(X_test))
    sample_positions = np.random.RandomState(RANDOM_STATE).choice(
        len(X_test), size=importance_sample_n, replace=False
    )
    X_imp = X_test.iloc[sample_positions]
    y_imp = y_test.iloc[sample_positions]
    perm = permutation_importance(
        final_model, X_imp, y_imp, scoring="roc_auc",
        n_repeats=3, random_state=RANDOM_STATE, n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    importance.to_csv(RESULTS_DIR / "permutation_feature_importance.csv", index=False)

    # Graph 7: CV comparison (PR-AUC is primary under class imbalance).
    plot_cv = comparison[comparison["model"] != "Dummy baseline"].sort_values("pr_auc_mean")
    plt.figure(figsize=(9, 5))
    plt.barh(plot_cv["model"], plot_cv["pr_auc_mean"], xerr=plot_cv["pr_auc_std"], capsize=3)
    plt.xlabel("Cross-validated Average Precision (PR-AUC)")
    plt.title("Model comparison on training folds")
    save_plot("07_model_comparison_pr_auc.png")

    # Graph 8: ROC curve.
    fpr, tpr, _ = roc_curve(y_test, test_probabilities)
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {test_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - held-out test set")
    plt.legend()
    save_plot("08_roc_curve_test.png")

    # Graph 9: precision-recall curve.
    pr, rc, _ = precision_recall_curve(y_test, test_probabilities)
    plt.figure(figsize=(5.5, 5))
    plt.plot(rc, pr, label=f"Average precision = {test_metrics['pr_auc_average_precision']:.3f}")
    plt.axhline(y_test.mean(), linestyle="--", color="grey", label=f"positive prevalence = {y_test.mean():.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve - held-out test set")
    plt.legend()
    save_plot("09_precision_recall_curve_test.png")

    # Graph 10: confusion matrix.
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No diabetes", "Diabetes"],
                yticklabels=["No diabetes", "Diabetes"])
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"Confusion matrix | threshold={threshold:.3f}")
    save_plot("10_confusion_matrix_test.png")

    # Graph 11: feature importance.
    plt.figure(figsize=(8, 5))
    imp_plot = importance.sort_values("permutation_importance_mean")
    plt.barh(imp_plot["feature"], imp_plot["permutation_importance_mean"],
             xerr=imp_plot["permutation_importance_std"], capsize=3)
    plt.xlabel("Decrease in test ROC-AUC after shuffling")
    plt.title("Permutation feature importance (raw model inputs)")
    save_plot("11_permutation_feature_importance.png")

    log("STEP 9 - SAVE MODEL AND FINAL METRICS")
    model_path = RESULTS_DIR / "model.joblib"
    joblib.dump(final_model, model_path, compress=3)
    final_output = {
        "project": "Leakage-aware diabetes classification (tabular ML)",
        "selected_model": best_name,
        "best_hyperparameters": search.best_params_,
        "threshold_selected_on_training_oof_for_max_f1": threshold_f1,
        "threshold_selected_on_training_oof_for_accuracy_with_recall_floor": threshold_accuracy,
        "minimum_recall_constraint_for_accuracy_threshold": MIN_RECALL_FOR_ACCURACY_POINT,
        "primary_selection_metric": "mean cross-validated average precision (PR-AUC); primary threshold maximizes F1 on training OOF predictions",
        "train_metrics_in_sample": train_metrics,
        "test_metrics_held_out": test_metrics,
        "train_metrics_accuracy_oriented_threshold": train_metrics_accuracy_point,
        "test_metrics_accuracy_oriented_threshold": test_metrics_accuracy_point,
        "tuning_cv_best_pr_auc": float(search.best_score_),
        "test_set_used_for_model_selection_or_tuning": False,
        "raw_model_inputs": RAW_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "engineered_feature_count": len(ENGINEERED_FEATURES),
        "total_pre_encoding_features": len(RAW_FEATURES) + len(ENGINEERED_FEATURES),
        "encoded_model_matrix_feature_count": int(encoded_feature_count),
        "diagnostic_features_excluded": EXCLUDED_DIAGNOSTIC_FEATURES,
        "class_imbalance_handling": "class_weight in estimators; stratified splitting/CV; no SMOTE",
        "data_audit": data_info,
        "warning": (
            "Educational ML output only. This dataset has no patient IDs or visit dates, "
            "so patient-level independence/longitudinal history cannot be verified. "
            "The score is not a clinical diagnosis or calibrated medical risk estimate."
        ),
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, default=float)

    print("\nPROJECT COMPLETED")
    print("Model:", model_path)
    print("Metrics:", RESULTS_DIR / "metrics.json")
    print("Tables:", RESULTS_DIR)
    print("Graphs:", GRAPHS_DIR)
    print("\nFINAL TEST SUMMARY - PRIMARY F1 THRESHOLD")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "f1", "roc_auc", "pr_auc_average_precision"]:
        print(f"{key:30s}: {test_metrics[key]:.4f}")
    print("\nFINAL TEST SUMMARY - SECONDARY ACCURACY-ORIENTED THRESHOLD")
    print("threshold:", round(threshold_accuracy, 5))
    for key in ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "f1", "roc_auc", "pr_auc_average_precision"]:
        print(f"{key:30s}: {test_metrics_accuracy_point[key]:.4f}")
    print("\nThe accuracy-oriented threshold was selected on training OOF predictions only, with a 30% recall floor.")
    print("Accuracy must still be read alongside recall, F1 and PR-AUC because the target is imbalanced.")



def write_final_report() -> None:
    """Create a self-contained Markdown report from this run's saved outputs."""
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    audit = data["data_audit"]
    test = data["test_metrics_held_out"]
    train = data["train_metrics_in_sample"]
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison_cv.csv")
    importance = pd.read_csv(RESULTS_DIR / "permutation_feature_importance.csv")
    leakage = pd.read_csv(RESULTS_DIR / "leakage_band_analysis.csv")
    class_dist = pd.read_csv(RESULTS_DIR / "class_distribution.csv")
    gaps = pd.read_csv(RESULTS_DIR / "train_test_overfit_gap.csv")

    class_rows = "\n".join(
        f"| {r.label} | {int(r.count):,} | {r.percentage:.2f}% |"
        for r in class_dist.itertuples()
    )
    cv_rows = "\n".join(
        f"| {r.model} | {r.pr_auc_mean:.4f} | {r.roc_auc_mean:.4f} | {r.f1_mean:.4f} |"
        for r in comparison.sort_values("pr_auc_mean", ascending=False).itertuples()
    )
    leak_rows = "\n".join(
        f"| {r.diagnostic_feature} | {r.value_band} | {int(r.patients):,} | {r.diabetes_rate:.1%} |"
        for r in leakage.itertuples()
    )
    gap_rows = "\n".join(
        f"| {r.metric} | {r.train_in_sample:.4f} | {r.test_held_out:.4f} | {r.absolute_gap:.4f} |"
        for r in gaps.itertuples()
    )
    importance_rows = "\n".join(
        f"| {r.feature} | {r.permutation_importance_mean:.5f} |"
        for r in importance.sort_values("permutation_importance_mean", ascending=False).itertuples()
    )
    threshold = data["threshold_selected_on_training_oof_for_max_f1"]
    report = f"""# Final Project Report — AI-Powered Diabetes Risk Classification

## 1. Problem, purpose and target

**Problem:** The dataset has a strongly imbalanced diabetes label and diagnostic laboratory columns that show suspiciously strong label separation. A naive model could report misleading accuracy or exploit diagnostic leakage.

**Purpose:** Build a reproducible Python machine-learning classification workflow using demographic and non-laboratory history fields, while auditing imbalance, leakage and generalization.

**Target:** `diabetes` (`0` = no diabetes, `1` = diabetes). The target is used only as the outcome, never as an input.

**Value:** An educational decision-support prototype that estimates the dataset label and makes false positives and missed positive cases measurable. It is not a medical diagnosis, treatment recommender, longitudinal patient tracker or live doctor-alert system.

## 2. Dataset and cleaning

- Input file: `diabetes_prediction_dataset.csv`
- Raw shape: **{audit['raw_rows']:,} rows × {audit['raw_columns']} columns**
- Exact duplicate rows removed before splitting: **{audit['exact_duplicate_rows_removed']:,}**
- Clean rows: **{audit['rows_after_duplicate_removal']:,}**
- Missing cells in original CSV: **{audit['missing_cells_in_original_file']:,}**
- `Other` gender rows retained: **{audit['gender_other_rows_retained']:,}**
- `No Info` smoking-history rows retained as a category: **{audit['smoking_no_info_rows_retained']:,}**
- Domain checks: `results/domain_validation_audit.csv`

IQR outliers are reported, not automatically deleted. Extreme measurements are not automatically data errors.

## 3. Class imbalance

| Class | Rows after duplicate removal | Share |
|---|---:|---:|
{class_rows}

Majority/minority ratio: **{audit['class_imbalance_ratio_majority_to_minority']:.2f}:1**. Stratified splitting and cross-validation preserve the class proportions. Class weights are used for supported estimators; no synthetic SMOTE rows are generated in the primary workflow. PR-AUC, recall, F1 and balanced accuracy accompany accuracy.

## 4. Feature count and decisions

- Original input columns: **6** — `{', '.join(data['raw_model_inputs'])}`.
- Engineered features added: **5** — `{', '.join(data['engineered_features'])}`.
- Total pre-encoding features: **{len(data['raw_model_inputs']) + len(data['engineered_features'])}**.
- Encoded matrix columns: **{data.get('encoded_model_matrix_feature_count', 'see metrics.json')}**.
- Excluded diagnostic fields: `{', '.join(data['diagnostic_features_excluded'])}`.

Engineered feature meanings:
- `log_bmi`: log transformation of BMI.
- `comorbidity_count`: hypertension plus heart-disease count.
- `obese`: indicator for BMI >= 30.
- `age_x_bmi`: row-wise age/BMI interaction.
- `age_group`: age bands under 30, 30–44, 45–59 and 60+.

## 5. Leakage safeguards

`HbA1c_level` and `blood_glucose_level` are excluded from every model input. They are examined only in an explanatory label-band audit:

| Diagnostic field | Band | Rows | Observed diabetes share |
|---|---|---:|---:|
{leak_rows}

This is strong evidence of possible diagnostic/label leakage, not proof of how the dataset creator generated the label. `diabetes` is target-only. Exact duplicates are removed before the split. Imputation, scaling and one-hot encoding are fitted inside pipelines on training folds only. Hyperparameter tuning and threshold selection use training data only. The held-out test set is reserved for final evaluation.

## 6. Workflow

1. Load and validate CSV schema and target.
2. Clean text/numeric types and perform domain checks.
3. Remove exact duplicate records before splitting.
4. EDA: missingness, target balance, descriptive statistics, skewness, kurtosis, IQR outliers, correlations and label rates by groups.
5. Audit suspicious diagnostic variables and exclude them from predictors.
6. Engineer five row-level features.
7. Stratified 80/20 train/test split.
8. Compare baseline and four ML classifiers using stratified cross-validation on training data.
9. Tune the best candidate using training folds only.
10. Select threshold using out-of-fold training predictions.
11. Evaluate once on held-out test data.
12. Save model, metrics, tables, predictions and graphs.

## 7. Model comparison (training CV)

| Model | Mean PR-AUC | Mean ROC-AUC | Mean F1 |
|---|---:|---:|---:|
{cv_rows}

Selected model: **{data['selected_model']}**. Selection metric: mean CV PR-AUC. Best parameters: `{data['best_hyperparameters']}`.

## 8. Final held-out test results

| Metric | Value |
|---|---:|
| Accuracy | {test['accuracy']:.4f} |
| Majority-class accuracy baseline | {test['majority_class_accuracy_baseline']:.4f} |
| Balanced accuracy | {test['balanced_accuracy']:.4f} |
| Precision | {test['precision']:.4f} |
| Recall / sensitivity | {test['recall_sensitivity']:.4f} |
| Specificity | {test['specificity']:.4f} |
| F1 | {test['f1']:.4f} |
| F2 | {test['f2']:.4f} |
| ROC-AUC | {test['roc_auc']:.4f} |
| PR-AUC / Average Precision | {test['pr_auc_average_precision']:.4f} |
| Training-selected F1 threshold | {threshold:.4f} |

Confusion matrix: TN={test['tn']:,}, FP={test['fp']:,}, FN={test['fn']:,}, TP={test['tp']:,}.

### Secondary accuracy-oriented operating point

A second threshold was selected from out-of-fold TRAIN predictions only to maximize accuracy subject to a minimum 30% recall constraint. Its held-out test results are:

| Metric | Accuracy-oriented test result |
|---|---:|
| Threshold | {data['threshold_selected_on_training_oof_for_accuracy_with_recall_floor']:.4f} |
| Accuracy | {data['test_metrics_accuracy_oriented_threshold']['accuracy']:.4f} |
| Precision | {data['test_metrics_accuracy_oriented_threshold']['precision']:.4f} |
| Recall | {data['test_metrics_accuracy_oriented_threshold']['recall_sensitivity']:.4f} |
| F1 | {data['test_metrics_accuracy_oriented_threshold']['f1']:.4f} |

This is a threshold trade-off, not a better underlying model. It may raise accuracy by reducing positive predictions, while missing more diabetic cases. The primary reporting point remains the F1-selected threshold. Both thresholds were selected without using the held-out test labels.

Accuracy is compared with the majority-class baseline because predicting all cases as non-diabetic can produce high accuracy while missing every diabetic case. Accuracy must always be read with recall, F1 and PR-AUC.

## 9. Overfitting / generalization check

| Metric | Train in-sample | Held-out test | Absolute gap |
|---|---:|---:|---:|
{gap_rows}

These gaps are diagnostic, not proof of no overfitting. In-sample training metrics are expected to be optimistic. External validation is required to establish generalization. Moderate precision/recall also mean underfitting or limited signal cannot be ruled out solely from this split.

Permutation importance:

| Feature | Mean ROC-AUC decrease when shuffled |
|---|---:|
{importance_rows}

## 10. Outputs

- `results/model.joblib` — trained end-to-end preprocessing/model pipeline.
- `results/metrics.json` — metrics, selected model, threshold and audit details.
- `results/*.csv` — EDA, cleaning, leakage, comparison and evaluation tables.
- `graphs/*.png` — EDA and evaluation plots.
- `FINAL_PROJECT_REPORT.md` — this report.

## 11. Limitations

The CSV has no patient identifier or visit timestamp. Patient-level split and true longitudinal tracking therefore cannot be verified. The public file does not establish a verified clinical cohort. This model is not clinically validated or calibrated and must not be used for diagnosis or treatment. External validation, fairness analysis, privacy controls and clinician oversight would be necessary before any real clinical use.
"""
    (BASE_DIR / "FINAL_PROJECT_REPORT.md").write_text(report, encoding="utf-8")
    print("Final report saved:", BASE_DIR / "FINAL_PROJECT_REPORT.md")


def run_prediction_demo() -> None:
    """Optional terminal example; educational output, not a diagnosis."""
    model_path = RESULTS_DIR / "model.joblib"
    metrics_path = RESULTS_DIR / "metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Train the model first: python diabetes_ai_project.py")
    model = joblib.load(model_path)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    threshold = float(metrics["threshold_selected_on_training_oof_for_max_f1"])

    def ask_choice(prompt, choices):
        valid = {v.lower(): v for v in choices}
        while True:
            value = input(f"{prompt} ({', '.join(choices)}): ").strip().lower()
            if value in valid:
                return valid[value]
            print("Please select one of the listed values.")

    def ask_number(prompt, low, high, integer=False):
        while True:
            try:
                value = float(input(f"{prompt} ({low} to {high}): ").strip())
            except ValueError:
                print("Please enter a number.")
                continue
            if not low <= value <= high or (integer and not value.is_integer()):
                print("Value outside the allowed range or not an integer.")
                continue
            return int(value) if integer else value

    patient = pd.DataFrame([{
        "gender": ask_choice("Gender", ["Female", "Male", "Other"]),
        "age": ask_number("Age", 0, 120),
        "hypertension": ask_number("Hypertension (0=no, 1=yes)", 0, 1, True),
        "heart_disease": ask_number("Heart disease (0=no, 1=yes)", 0, 1, True),
        "smoking_history": ask_choice("Smoking history", ["never", "former", "current", "not current", "ever", "No Info"]),
        "bmi": ask_number("BMI", 1, 100),
    }])
    score = float(model.predict_proba(patient)[0, 1])
    print("\nModel score (not a clinically calibrated probability):", round(score, 4))
    print("Training-selected threshold:", round(threshold, 4))
    print("Model flag:", "POSITIVE - seek clinician assessment" if score >= threshold else "NEGATIVE - this does not rule out diabetes")
    print("This educational model is not a medical diagnosis or treatment recommendation.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--predict":
        run_prediction_demo()
    else:
        main()
        write_final_report()
