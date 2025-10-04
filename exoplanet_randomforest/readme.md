# Exoplanet Detection Pipeline — Full Documentation (Markdown)

> This file documents the full pipeline code (Optuna + RandomForest + feature engineering + visualization).  
> It explains **every design decision**, the rationale, trade-offs, and how to run and extend the pipeline.

---

## Table of contents

1. [Purpose & Scope](#purpose--scope)  
2. [Quick start (commands)](#quick-start-commands)  
3. [Files / Outputs produced](#files--outputs-produced)  
4. [Environment & installation](#environment--installation)  
5. [High-level pipeline flow](#high-level-pipeline-flow)  
6. [Labeling strategy (target mapping)](#labeling-strategy-target-mapping)  
7. [Data expectations & prechecks](#data-expectations--prechecks)  
8. [Feature engineering — full details](#feature-engineering---full-details)  
9. [Preprocessing pipeline steps & why](#preprocessing-pipeline-steps--why)  
10. [Model choice & hyperparameters](#model-choice--hyperparameters)  
11. [Hyperparameter optimization with Optuna](#hyperparameter-optimization-with-optuna)  
12. [Evaluation metrics & interpretation](#evaluation-metrics--interpretation)  
13. [Visualization (inline + saved)](#visualization-inline--saved)  
14. [Implementation details & code hooks to change](#implementation-details--code-hooks-to-change)  
15. [Limitations, risks & cautions](#limitations-risks--cautions)  
16. [Possible improvements & next steps](#possible-improvements--next-steps)  
17. [Reproducibility & logging recommendations](#reproducibility--logging-recommendations)  
18. [Contact / authorship note](#contact--authorship-note)

---

## Purpose & Scope

This pipeline classifies *TESS-like* catalog entries as **exoplanet** (`1`) or **not an exoplanet** (`0`). It is engineered specifically for tabular catalogs (TOI/TESS) that include astrophysical parameters, their uncertainties, and limit flags. The code:

- constructs domain-informed features,
- imputes and scales features,
- corrects class imbalance (SMOTE),
- optimizes RandomForest hyperparameters with **Optuna** (Bayesian optimization),
- evaluates on a hold-out test set, and
- optionally plots ROC, PR, and feature importance charts.

This document explains why each step exists and how to modify or extend it.

---

## Quick start (commands)

Assuming the pipeline script is `pipeline_optuna.py` and your CSV is `tess.csv` in the working directory:

```bash
# install optuna (choose one)
pip install optuna
# or (if pip causes conflicts)
# conda install -c conda-forge optuna

# run (defaults: csv_path='tess.csv', outdir=~/Desktop/space_apps_hackathon_optuna, n_trials=30)
python pipeline_optuna.py

# or explicit
python pipeline_optuna.py --csv_path /path/to/tess.csv --outdir ./results --n_trials 50 --visualize True
```

(If run inside a Jupyter notebook cell, prepend `!` before pip/conda commands to run them in the shell.)

---

## Files / Outputs produced

When the pipeline finishes, the default outputs are placed inside `outdir`:

- `rf_pipeline_optuna.joblib` — saved trained pipeline (imputer, scaler, smote, RF).  
- `metrics.json` — dictionary of evaluation metrics (ROC AUC, PR AUC, accuracy, F1, confusion matrix, classification report, best_params).  
- `roc_curve.png`, `precision_recall_curve.png`, `feature_importance.png` — if `visualize=True`.  
- (Optional) `feature_importance.csv` — ranked feature importances (if code writes CSV).

---

## Environment & installation

### Required Python packages (core)
- `numpy`, `pandas`  
- `scikit-learn`  
- `imblearn` (imbalanced-learn)  
- `joblib`  
- `matplotlib` (for plots)  
- `optuna` (for Bayesian hyperparameter optimization)

### Install Optuna (RAPIDS / conda-safe)
If you are using a RAPIDS 24.10 conda environment, prefer conda-forge to avoid numpy/scikit-learn conflicts:
```bash
# recommended if there are dependency issues
conda install -c conda-forge optuna

# or try pip (usually works)
pip install optuna --upgrade
```

**Verification**:
```python
import optuna
print(optuna.__version__)
```

> [Inference] If you cannot install `optuna` because of locked system packages, use `RandomizedSearchCV` or `HalvingRandomSearchCV` as fallbacks (see code hooks).

---

## High-level pipeline flow

1. **Load CSV** → expects TESS-like columns including `tfopwg_disp`.  
2. **Map labels** using the TFOPWG disposition mapping (explained below). Ambiguous rows are dropped.  
3. **Engineer features** (uncertainties, limits, astrophysical ratios).  
4. **Keep numeric columns**, drop columns with all `NaN`.  
5. **Train/test split** (stratified).  
6. **Run Optuna** on training set (CV inside objective) to select RF hyperparameters.  
7. **Build final pipeline** with best params and fit on entire training set.  
8. **Evaluate** on hold-out test set and save metrics + model.  
9. **Visualize** results (plots shown inline in notebooks and saved to disk).

---

## Labeling strategy (target mapping)

Function:
```py
def map_labels(series):
    mapping = {"CP": 1, "KP": 1, "FP": 0, "FA": 0, "PC": np.nan, "APC": np.nan}
    mapped = series.astype(str).str.upper().map(mapping)
    return mapped.dropna()
```

**Why this mapping?**
- `CP` = Confirmed Planet → **positive (1)**  
- `KP` = Known Planet → **positive (1)**  
- `FP` = False Positive → **negative (0)**  
- `FA` = False Alarm → **negative (0)**  
- `PC`, `APC` = Planet Candidate / Ambiguous → **dropped** to avoid label noise

**Trade-offs and alternatives:**
- [Unverified] If you have verified labels for `PC` entries or want to include them as weak positives, you can map them to `1` and use sample weighting or label smoothing. If you do, explicitly state that label noise may increase.

---

## Data expectations & prechecks

**Expected columns (examples from TESS TOI):**
- Planet features: `pl_trandep`, `pl_trandurh`, `pl_orbper`, `pl_rade`, `pl_insol`, `pl_eqt`, and their `*err1`, `*err2`, `*lim` variants.  
- Stellar features: `st_teff`, `st_rad`, `st_logg`, `st_tmag`, `st_dist`, etc.  
- TFOPWG disposition: `tfopwg_disp`.

**Prechecks performed by pipeline:**
- Confirms `tfopwg_disp` existence or raises an explanatory error.  
- Converts strings to numeric where appropriate during feature engineering.  
- Drops numeric columns with all `NaN` (these make median imputer raise warnings).

**User action:** If your file uses different column names, either rename them to match the pipeline or update the code where features are referenced.

---

## Feature engineering — full details

All engineered features are deterministic transformations of available input columns. Each feature is only created if the required input columns exist.

### 1) Relative uncertainty features

For any base feature `X` with columns `Xerr1`, `Xerr2`:

```
relative_uncertainty_X = (|Xerr1| + |Xerr2|) / (2 * |X|)
```

**Handling & rationale:**
- Use absolute errors and absolute value of the measurement in denominator.
- Replace zeros in `X` with `NaN` before division to avoid infinite values: `val = X.replace(0, np.nan).abs()`.
- If any constituent is missing, the resulting `relative_uncertainty_X` will be `NaN` and later imputed.
- Purpose: Provide the model with a measure of measurement reliability; higher values → lower confidence.

### 2) Limit flags

For any `Xlim` column (where `Xlim` indicates limit flags in TESS schema):

```
is_limited_X = 1 if Xlim != 0 else 0
```

**Rationale:** Flags are necessary because limit-derived measurements should be treated differently (less trust).

### 3) Derived astrophysical features (heuristic but useful)

All use `safe_div` which returns `NaN` for infinite/invalid results:

- `transit_snr = pl_trandep / sqrt(pl_trandurh)`  
  - Heuristic proxy for signal-to-noise (depth scaled by sqrt(duration)). Units are mixed (ppm and sqrt(hours)) — treat as heuristic.

- `planet_star_radius_ratio = pl_rade / st_rad`  
  - Planet radius in Earth radii vs stellar radius in solar radii — ratio is a geometric indicator.

- `flux_temp_ratio = pl_insol / st_teff`  
  - Insulation vs star temperature — heuristic for energy-related trends.

- `period_over_duration = pl_orbper / pl_trandurh`  
  - Timescale ratio; can help separate grazing vs central transits.

- `depth_over_tmag = pl_trandep / st_tmag`  
  - Transit depth adjusted for star brightness (magnitude scale). Heuristic.

**Important:** these are *engineered signals* — they are not physically exact conserved quantities; they are intended to improve classification performance by capturing relationships commonly informative in transit analyses.

---

## Preprocessing pipeline steps & why

Pipeline sequence:

1. **`SimpleImputer(strategy='median')`**  
   - Median is robust to extreme values and skewed distributions common in astrophysical data.  
   - Imputation only runs on training folds (because the pipeline is inside the CV / training process).

2. **`StandardScaler()`**  
   - Z-score normalization is chosen because RandomForest is less sensitive to scaling, but scaling ensures SMOTE interpolation happens in a standardized feature space (important because SMOTE creates synthetic points between neighbors).  
   - StandardScaler is applied after imputation to avoid `NaN` propagation.

3. **`SMOTE(random_state=RANDOM_STATE)`**  
   - Synthetic Minority Oversampling Technique: balances the class distribution by synthesizing new minority-class examples.  
   - *Why in the pipeline?* Putting SMOTE in the pipeline ensures SMOTE is applied only to training folds during cross-validation; avoids leaking synthetic examples into validation/test data.  
   - *Caution:* SMOTE assumes local linearity — it can produce physically unrealistic combinations; always inspect generalization.

4. **`RandomForestClassifier(...)`**  
   - Robust to noisy features and doesn't require heavy feature engineering. Good baseline and often strong performer on tabular data.

---

## Model choice & hyperparameters

**Model**: `RandomForestClassifier`  
**Key hyperparameters tuned:**
- `n_estimators` — number of trees. More trees → lower variance but longer training.
- `max_depth` — maximum depth of each tree. Controls overfitting vs underfitting.
- `min_samples_split` — minimum samples to split a node. Higher values → simpler trees.
- `max_features` — number of features considered at each split (e.g. `'sqrt'`, `'log2'`).

**Rationale for RandomForest:**
- Interpretable via feature importances.
- Robust to outliers and irrelevant features.
- Handles mixed numeric features without much preprocessing.
- Fast enough on CPU and parallelizable (`n_jobs=-1`) for moderate dataset sizes.

**When to switch**: for large data / higher performance needs, try XGBoost or LightGBM (often faster / higher accuracy). Those can be integrated into the same pipeline but are not included to keep dependencies minimal.

---

## Hyperparameter optimization with Optuna

**Why Optuna?**
- Bayesian optimization with adaptive sampling and optional pruning → more sample-efficient than grid/random search.
- Provides visualization utilities (if you want to explore parameter importance, optimization history).
- Easy to integrate: objective function returns the CV performance metric to maximize.

**Objective function design:**
- Runs `cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(5), scoring='average_precision')`.
- Returns mean PR AUC (Average Precision) across folds. PR AUC is preferred in imbalanced settings.
- The pipeline used for CV includes imputer → scaler → SMOTE → RF, ensuring hyperparameters are evaluated realistically.

**Optuna parameters used in code (example):**
```py
trial.suggest_int('rf__n_estimators', 100, 600)
trial.suggest_categorical('rf__max_depth', [None, 10, 20, 30])
trial.suggest_int('rf__min_samples_split', 2, 10)
trial.suggest_categorical('rf__max_features', ['sqrt', 'log2'])
```

**n_trials**:
- Default `n_trials=30` is a reasonable compromise between compute and search thoroughness. Increase for better tuning if you have more CPU time.

**Pruning (optional improvement):**
- Optuna supports trial pruning to terminate poor trials early. For expensive trials you can set up a `sklearn`-compatible pruner or use intermediate `cross_val_score` callbacks to report progress. (Not enabled by default in the given code).

---

## Evaluation metrics & interpretation

The pipeline computes (and saves) the following metrics on the **hold-out test set**:

- **ROC AUC (Area Under ROC Curve)** — measures ranking quality across all thresholds. Useful general-purpose metric.
- **PR AUC (Average Precision)** — area under precision-recall curve; preferred in imbalanced settings since it focuses on the positive class performance.
- **Accuracy** — fraction of correct predictions. May be misleading for imbalanced data.
- **F1 Score** — harmonic mean of precision & recall; useful when both false positives and false negatives matter.
- **Confusion Matrix** — counts of true positives, false positives, true negatives, false negatives.
- **Classification report** — detailed precision, recall, F1 by class.

**Interpretation guidance:**
- With imbalanced classes, prioritize **PR AUC**, **precision**, and **recall** over accuracy.
- If downstream process costs (human vetting, telescope time) are expensive, prefer higher precision; if missing planets is riskier, prefer higher recall.
- Use confusion matrix to choose an operating threshold (not necessarily 0.5) that matches operational priorities.

---

## Visualization (inline + saved)

`plot_results(y_test, y_prob, feature_importances, outdir, show_inline=True)` does:

1. **ROC Curve** — saved to `roc_curve.png` and shown inline if `show_inline=True`.  
2. **Precision-Recall Curve** — saved to `precision_recall_curve.png`.  
3. **Top 20 Feature Importances** — saved to `feature_importance.png`.  
4. Plots are created with `matplotlib` and `plt.show()` is used to render inline within notebooks.

**Display in Notebook**:
- Ensure the notebook uses `%matplotlib inline` (or Jupyter default) so `plt.show()` renders inline.
- If you run the script from the shell (non-interactive), `show_inline=True` is harmless but won't display; files are saved for later viewing.

---

## Implementation details & code hooks to change

### Where to change input/output paths:
- Default arguments in `main(...)`: `csv_path="tess.csv"`, `outdir=os.path.expanduser("~/Desktop/space_apps_hackathon_optuna")`.
- You can override these by passing arguments to the script or editing defaults.

### Where to change Optuna search space / trials:
- `objective` function contains `trial.suggest_*` calls — modify ranges or add/remove hyperparameters.  
- `study.optimize(..., n_trials=n)` controls the number of trials.

### Where to change CV strategy:
- The objective uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)`. Increase `n_splits` for more robust CV (but slower).

### Where to change SMOTE params:
- `SMOTE(random_state=RANDOM_STATE, k_neighbors=5)` — adjust `k_neighbors` for different oversampling behavior.

### Switching to another search method:
- Replace Optuna block with `RandomizedSearchCV`, `HalvingRandomSearchCV`, or `GridSearchCV` as desired (code comments provide example replacements).

---

## Limitations, risks & cautions

- **SMOTE may synthesize unrealistic astrophysical feature combinations** — inspect synthetic data behavior or consider alternative imbalance strategies (class weighting, undersampling, hybrid sampling).
- **Engineered features are heuristics** — they help classification but should not be interpreted as precise physical measures without domain validation.
- **Optuna tuning uses CV inside the training set** — although robust, this still risks overfitting hyperparameters to cross-validation idiosyncrasies. Use nested CV if you want to estimate generalization of the entire tuning procedure.
- **Data leakage risk** — placing any preprocessing that uses full dataset statistics outside the pipeline could leak information. The current design avoids that by keeping imputer/scaler inside the pipeline used for CV.
- **Runtime** — Optuna trials each run CV on the pipeline; expect nontrivial compute time for `n_trials >= 30`. Use fewer trials or halving/pruning for resource constraints.

---

## Possible improvements & next steps

- **Enable Optuna pruning** to shorten poor trials.  
- **Add Optuna visualizations**: `optuna.visualization.plot_optimization_history(study)` and `plot_param_importances(study)`.  
- **Test alternative models**: LightGBM/XGBoost (often faster and more accurate on tabular data).  
- **Add nested cross-validation** for hyperparameter selection validation.  
- **Add domain-specific features** from light curves (e.g., transit shape metrics, Fourier features) if raw time-series are available.  
- **Ensemble**: stack RF with gradient-boosted trees and small neural nets for improved performance.  
- **Calibrate probabilities** (e.g., isotonic / Platt scaling) if you need reliable probability estimates for decision thresholds.

---

## Reproducibility & logging recommendations

- Set `RANDOM_STATE` globally (already done). To fully control randomness, also set `numpy.random.seed(RANDOM_STATE)` at top-level in a notebook session if needed.
- Save the Optuna `study` object to disk: `joblib.dump(study, 'optuna_study.joblib')` to reproduce/inspect results later.
- Enable verbose logging during long runs; consider writing to a log file using Python’s `logging` module.

---

## Contact / authorship note

- This documentation was produced programmatically to describe the provided Optuna + RandomForest pipeline.  
- If you want the documentation embedded as docstrings inside the source code, or exported to a `README.md` file in the repository, let me know and I will generate that version.

---

### End of documentation

If you want, I can now:
- produce this as a `README.md` file (write to disk),
- insert these contents as top-of-file docstring in your pipeline script, or
- generate a condensed one-page summary for a presentation. Which would you like next?

