# Final Project Report — AI-Powered Diabetes Risk Classification

## 1. Problem, purpose and target

**Problem:** The dataset has a strongly imbalanced diabetes label and diagnostic laboratory columns that show suspiciously strong label separation. A naive model could report misleading accuracy or exploit diagnostic leakage.

**Purpose:** Build a reproducible Python machine-learning classification workflow using demographic and non-laboratory history fields, while auditing imbalance, leakage and generalization.

**Target:** `diabetes` (`0` = no diabetes, `1` = diabetes). The target is used only as the outcome, never as an input.

**Value:** An educational decision-support prototype that estimates the dataset label and makes false positives and missed positive cases measurable. It is not a medical diagnosis, treatment recommender, longitudinal patient tracker or live doctor-alert system.

## 2. Dataset and cleaning

- Input file: `diabetes_prediction_dataset.csv`
- Raw shape: **100,000 rows × 9 columns**
- Exact duplicate rows removed before splitting: **3,854**
- Clean rows: **96,146**
- Missing cells in original CSV: **0**
- `Other` gender rows retained: **18**
- `No Info` smoking-history rows retained as a category: **32,887**
- Domain checks: `results/domain_validation_audit.csv`

IQR outliers are reported, not automatically deleted. Extreme measurements are not automatically data errors.

## 3. Class imbalance

| Class | Rows after duplicate removal | Share |
|---|---:|---:|
| No diabetes | 87,664 | 91.18% |
| Diabetes | 8,482 | 8.82% |

Majority/minority ratio: **10.34:1**. Stratified splitting and cross-validation preserve the class proportions. Class weights are used for supported estimators; no synthetic SMOTE rows are generated in the primary workflow. PR-AUC, recall, F1 and balanced accuracy accompany accuracy.

## 4. Feature count and decisions

- Original input columns: **6** — `gender, age, hypertension, heart_disease, smoking_history, bmi`.
- Engineered features added: **5** — `log_bmi, comorbidity_count, obese, age_x_bmi, age_group`.
- Total pre-encoding features: **11**.
- Encoded matrix columns: **21**.
- Excluded diagnostic fields: `HbA1c_level, blood_glucose_level`.

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
| HbA1c_level | <=5.6 | 36,265 | 0.0% |
| HbA1c_level | 5.6-7.0 | 57,268 | 10.2% |
| HbA1c_level | >7.0 | 2,613 | 100.0% |
| blood_glucose_level | <=100 | 27,014 | 0.0% |
| blood_glucose_level | 100-200 | 65,868 | 7.9% |
| blood_glucose_level | >200 | 3,264 | 100.0% |

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
| HistGradientBoosting | 0.3251 | 0.8334 | 0.3352 |
| Logistic Regression | 0.3219 | 0.8331 | 0.3381 |
| Random Forest | 0.3194 | 0.8306 | 0.3444 |
| Decision Tree | 0.2974 | 0.8266 | 0.3224 |
| Dummy baseline | 0.0882 | 0.5000 | 0.0000 |

Selected model: **HistGradientBoosting**. Selection metric: mean CV PR-AUC. Best parameters: `{'classifier__min_samples_leaf': 60, 'classifier__max_leaf_nodes': 31, 'classifier__max_iter': 140, 'classifier__learning_rate': 0.08, 'classifier__l2_regularization': 3.0}`.

## 8. Final held-out test results

| Metric | Value |
|---|---:|
| Accuracy | 0.8659 |
| Majority-class accuracy baseline | 0.9118 |
| Balanced accuracy | 0.6812 |
| Precision | 0.3187 |
| Recall / sensitivity | 0.4570 |
| Specificity | 0.9055 |
| F1 | 0.3755 |
| F2 | 0.4205 |
| ROC-AUC | 0.8266 |
| PR-AUC / Average Precision | 0.3191 |
| Training-selected F1 threshold | 0.7392 |

Confusion matrix: TN=15,877, FP=1,657, FN=921, TP=775.

### Secondary accuracy-oriented operating point

A second threshold was selected from out-of-fold TRAIN predictions only to maximize accuracy subject to a minimum 30% recall constraint. Its held-out test results are:

| Metric | Accuracy-oriented test result |
|---|---:|
| Threshold | 0.8000 |
| Accuracy | 0.8944 |
| Precision | 0.3787 |
| Recall | 0.3084 |
| F1 | 0.3399 |

This is a threshold trade-off, not a better underlying model. It may raise accuracy by reducing positive predictions, while missing more diabetic cases. The primary reporting point remains the F1-selected threshold. Both thresholds were selected without using the held-out test labels.

Accuracy is compared with the majority-class baseline because predicting all cases as non-diabetic can produce high accuracy while missing every diabetic case. Accuracy must always be read with recall, F1 and PR-AUC.

## 9. Overfitting / generalization check

| Metric | Train in-sample | Held-out test | Absolute gap |
|---|---:|---:|---:|
| accuracy | 0.8711 | 0.8659 | 0.0052 |
| balanced_accuracy | 0.6965 | 0.6812 | 0.0152 |
| precision | 0.3389 | 0.3187 | 0.0202 |
| recall_sensitivity | 0.4844 | 0.4570 | 0.0274 |
| f1 | 0.3988 | 0.3755 | 0.0233 |
| roc_auc | 0.8485 | 0.8266 | 0.0219 |
| pr_auc_average_precision | 0.3480 | 0.3191 | 0.0289 |

These gaps are diagnostic, not proof of no overfitting. In-sample training metrics are expected to be optimistic. External validation is required to establish generalization. Moderate precision/recall also mean underfitting or limited signal cannot be ruled out solely from this split.

Permutation importance:

| Feature | Mean ROC-AUC decrease when shuffled |
|---|---:|
| age | 0.21706 |
| bmi | 0.05429 |
| hypertension | 0.00819 |
| heart_disease | 0.00770 |
| gender | 0.00300 |
| smoking_history | 0.00241 |

## 10. Outputs

- `results/model.joblib` — trained end-to-end preprocessing/model pipeline.
- `results/metrics.json` — metrics, selected model, threshold and audit details.
- `results/*.csv` — EDA, cleaning, leakage, comparison and evaluation tables.
- `graphs/*.png` — EDA and evaluation plots.
- `FINAL_PROJECT_REPORT.md` — this report.

## 11. Limitations

The CSV has no patient identifier or visit timestamp. Patient-level split and true longitudinal tracking therefore cannot be verified. The public file does not establish a verified clinical cohort. This model is not clinically validated or calibrated and must not be used for diagnosis or treatment. External validation, fairness analysis, privacy controls and clinician oversight would be necessary before any real clinical use.
