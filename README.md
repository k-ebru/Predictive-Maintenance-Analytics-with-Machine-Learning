# Turbofan Engine Remaining Useful Life and Health Classification

Predicting how many cycles an aircraft engine has left, and flagging engines
that need maintenance soon, from multivariate sensor readings.

I originally built this project in MATLAB; the Python version in this repo
follows the same pipeline and reaches almost identical numerical results.
The MATLAB scripts are kept under `matlab/` as the starting point.

## What the project does

The dataset records one month of run-to-failure histories for 100 training
engines (20,631 cycles total) plus a snapshot of 100 test engines whose
true Time-to-Failure is held out. For each cycle there is a cycle counter
and four sensor channels (`s1` to `s4`).

Two questions:

1. How many cycles until each engine fails? Regression on TTF.
2. Should an engine be flagged now? Binary classification with the rule
   "faulty if remaining TTF is 30 cycles or less".

## Why it matters

For a maintenance team, the useful output is not just an accurate model. The
model also has to support a decision: which engines should be inspected soon,
and how much useful life is likely to remain.

That is why I treated this as both a regression problem and a classification
problem. The regression estimates remaining cycles. The classifier turns the
same sensor history into a maintenance flag.

## Pipeline

Preprocessing (same for both tasks):

* Trailing moving-average smoothing on each sensor, window size 3, applied
  per engine to keep degradation trends but remove short-term noise.
* For regression, the training TTF is capped at 125 cycles (piecewise-linear
  RUL, Heimes 2008) so the model focuses on the informative end-of-life part.
* Z-score normalisation using training statistics for the scale-sensitive
  models.
* For the test set, only the last available cycle of each engine is used.
  That row represents the current state at decision time.

Regression models:

* Random forest, 500 trees, minimum leaf size 120, original feature scale.
* Stepwise quadratic regression with forward/backward selection on a
  candidate set of main effects, cycle-sensor interactions and a few
  squared terms.

Classification models, both on z-scored features:

* Logistic regression. The decision threshold is averaged across 10
  engine-level cross-validation folds, picking the threshold that keeps
  recall above 0.95 in each fold.
* KNN with K chosen in the range 35-55 by mean F1 over the same 10 folds.
  The final threshold comes from Youden's J on the training set.

The 10-fold split is by engine id, not by row, so consecutive cycles of the
same engine never appear in both train and validation.

## Results

Regression on the test set:

| Model              | MAE   | RMSE  | R^2  |
|--------------------|-------|-------|------|
| Stepwise quadratic | 16.63 | 20.65 | 0.73 |
| Random forest      | 15.89 | 20.50 | 0.74 |

Classification on the test set:

| Model               | Accuracy | Precision | Recall | F1   | AUC  |
|---------------------|----------|-----------|--------|------|------|
| Logistic regression | 0.91     | 0.81      | 0.84   | 0.82 | 0.96 |
| KNN (K = 41)        | 0.92     | 0.76      | 1.00   | 0.86 | 0.97 |

The stepwise output reproduces the MATLAB formula exactly:

```
TTF = 84.4956 - 17.0118*cycle - 7.4903*s1 + 10.4856*cycle^2 - 9.2043*s3
      - 5.7366*cycle:s3 - 4.6750*cycle:s1 + 3.3432*s2 + 2.3127*s4
      + 2.3581*cycle:s2 - 0.9063*s1:s2 - 0.5367*s4^2
```

A one-page methodology summary is in [`report.md`](report.md).

## How to run

```bash
pip install -r requirements.txt

# quick run from the command line
python run_pipeline.py

# or open the notebooks
cd notebooks
jupyter notebook
```

The notebooks should be opened in order: `01_eda` then `02_regression_ttf`
then `03_classification`. All the data files are inside `data/` so nothing
extra needs to be downloaded.

## Tests

A small pytest suite covers the preprocessing helpers, the engine level
cross validation split and the regression utilities.

```bash
pip install pytest
pytest tests/ -v
```

## Reproducibility

The random forest and the KFold split both use `random_state=42`, so
`python run_pipeline.py` on the files in `data/` gives the numbers in the
results tables above.

## Repository layout

```
Predictive-Maintenance-Analytics-with-Machine-Learning/
  README.md
  report.md
  run_pipeline.py          quick command-line reproduction
  requirements.txt
  .gitignore
  notebooks/
    01_eda.ipynb
    02_regression_ttf.ipynb
    03_classification.ipynb
  src/
    preprocessing.py       smoothing, clipping, normalisation
    regression.py          random forest + stepwise quadratic
    classification.py      logistic + KNN + threshold tuning
    evaluation.py          metrics and plotting helpers
  tests/                   pytest suite
  sql/
    01_create_tables.sql
    02_engine_lifecycle_summary.sql
    03_faulty_engines_at_risk.sql
    04_sensor_anomalies.sql
  matlab/                  original MATLAB prototypes
    regression.m
    classification.m
  data/
    train_selected.csv
    test_selected.csv
    PM_truth.txt
```

## Limitations and what I would try next

* The 100 training engines all run under similar operating conditions,
  so generalisation to other regimes is not tested here. Multi-regime data
  would be the natural next step.
* The classifiers look at a single cycle in isolation. A sequence model
  like an LSTM or a 1D CNN would be able to use the time-series structure.
* Class imbalance is handled at the threshold, not in the loss. Sample
  weighting or focal loss are reasonable alternatives.

## Stack

Python, pandas, NumPy, scikit-learn, statsmodels, SciPy, Matplotlib, SQL.
Originally prototyped in MATLAB.

## License

The project code is released under the MIT License.
