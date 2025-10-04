# Random Forest Classification on TESS Data


## Overview

This project trains a **Random Forest Classifier** to classify TESS (Transiting Exoplanet Survey Satellite) light-curve derived features into disposition classes. The workflow covers data loading, feature engineering, class-imbalance handling, model selection via **Optuna** hyperparameter optimization, evaluation, and artifact export.

## What the notebook does 

* Loads a CSV (default: `tess.csv`) containing a target column **`tfopwg_disp`**.
* Maps target labels via a helper (e.g., `map_labels`) and filters rows accordingly.
* Keeps **numeric** feature columns after basic feature engineering (`engineer_features`).
* Splits data with `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
* Uses an **imblearn Pipeline**: `SimpleImputer` → `StandardScaler` → **SMOTE** → `RandomForestClassifier`.
* Performs **hyperparameter optimization with Optuna** using an `objective(trial, X_train, y_train)` and:

  * `optuna.create_study(direction='maximize')`
  * `study.optimize(..., n_trials=n_trials)` with **default `n_trials=70`**.
  * [Unverified] The exact **sampler** is not set in the cells I can see. If no sampler is provided, **Optuna defaults to TPE (Bayesian optimization)**. Please update this line if you explicitly set a different sampler in your local copy.
* Cross‑validation during optimization: **StratifiedKFold(n_splits=5)**.
* Metrics used in the notebook include: `roc_auc_score`, `average_precision_score`, `f1_score`, `accuracy_score`, `confusion_matrix`, and a **classification report** (scikit‑learn).
* Exports the trained model with **joblib** (`dump`).
* Plots ROC/PR curves and **feature importances** (via RandomForest `feature_importances_`).

## Project Structure

```
randomforest_TESS.ipynb   # Main notebook (training & evaluation)
README.md                 # This document
```

## Data

* **Input**: CSV with a target column `tfopwg_disp` and numeric feature columns.
* **Target processing**: Labels are mapped (via `map_labels`) to form the binary/multi-class target `y`.
* **Feature processing**: `engineer_features` is applied, then only numeric columns are retained.

## Dependencies

Install with pip (Python ≥ 3.9 recommended):

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib optuna joblib jupyter
```

If you prefer a file, create `requirements.txt` like:

```txt
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
optuna
joblib
jupyter
```

## Quickstart

1. Ensure your data file exists (default expected path: `tess.csv`).
2. Launch the notebook:

   ```bash
   jupyter notebook randomforest_TESS.ipynb
   ```
3. Run cells top to bottom. By default, **70 Optuna trials** will run. You can raise this in the `main(..., n_trials=70)` call.

## Hyperparameter Tuning (Optuna)

* **Algorithm**: [Unverified] Sampler not explicitly shown in notebook cells provided. If none is set, Optuna uses **TPE** (a Bayesian optimization method). If you used a different sampler (e.g., `CmaEsSampler`, `RandomSampler`), please adjust this section.
* **Search Objective**: `direction='maximize'`. [Unverified] The exact scoring used inside `objective(...)` is not visible; common choices here are **ROC AUC** or **F1** with 5‑fold Stratified CV.
* **Trials**: Default `n_trials=70` (you can increase for better search coverage).
* **Cross‑Validation**: `StratifiedKFold(n_splits=5)` during objective evaluation.
* **Typical Search Space** (example — edit to match your notebook if different):

  ```python
  # Inside objective(trial, ...):
  params = {
      "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
      "max_depth": trial.suggest_int("max_depth", 3, 30),
      "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
      "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
      "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
      "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
      # Optionally:
      # "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
      # "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
  }
  ```
* **Why Optuna?** It explores the parameter space efficiently (TPE), supports pruning and logging, and integrates cleanly with scikit‑learn and CV.

## Evaluation & Outputs

* **Printed**: confusion matrix and scikit‑learn `classification_report` on the hold‑out test set.
* **Curves**: ROC and Precision‑Recall plots.
* **Feature Importances**: bar plot of `feature_importances_`.
* **Model Artifact**: a serialized `.joblib` model saved to disk (path printed in the notebook).

## Reproducibility

* Uses `random_state=42` where applicable.
* The pipeline fixes preprocessing and SMOTE within CV to avoid leakage.
* For deterministic runs across machines, pin package versions in `requirements.txt`.

## Customization

* **Change number of trials**: set `n_trials` in the `main()` call.
* **Switch metric**: edit the objective to return your preferred CV score (e.g., `roc_auc`, `f1`, `average_precision`).
* **Add parameters**: extend the Optuna search space for `criterion`, `class_weight`, etc.

## Troubleshooting

* **Imbalance issues**: Ensure SMOTE is applied inside the pipeline *before* the classifier.
* **Convergence/time**: Reduce `n_trials` or constrain the search ranges. Increase if results vary.
* **Data errors**: Confirm `tfopwg_disp` exists and that your CSV contains the expected numeric features.



