# Methodology summary — one page

## Data

A pre-selected subset of the NASA C-MAPSS turbofan engine dataset is used:
20,631 training cycles across 100 engines, plus a single snapshot for each of
100 test engines. Each row carries the engine id, cycle counter, four sensors
(`s1`–`s4`) and (for the training set) the remaining Time-to-Failure (TTF).
True test TTFs are held out in `PM_truth.txt`.

EDA highlights:
- no missing sensor values
- strong class imbalance once `TTF ≤ 30` is treated as "faulty"
- train and test sensor distributions overlap closely
- monotone degradation visible in all four sensors

## Preprocessing

The same pipeline feeds both tasks:

1. **Trailing moving-average smoothing** (window = 3) applied per engine to
   suppress short-term noise while preserving the slow degradation trend.
2. **TTF cap at 125 cycles** for regression — a piecewise-linear RUL
   formulation (Heimes, 2008) that focuses the loss on the informative
   end-of-life region instead of the long healthy plateau.
3. **Z-score normalisation** using training statistics, applied to the test
   set with the same parameters, for the scale-sensitive models.
4. **Last-cycle slicing** for the test set — only the final observation per
   test engine is used, reflecting the operational moment of decision.

## Regression — Time-to-Failure

Two models trained on the smoothed, capped data:

- **Random forest** (500 bagged trees, min leaf size 120) — operates on the
  original feature scale. Test MAE = 15.34 cycles, RMSE = 19.77, R² = 0.76.
- **Stepwise quadratic regression** on normalised inputs — forward/backward
  selection over main effects, cycle-sensor interactions and selected squared
  terms; final test MAE = 16.63, RMSE = 20.65, R² = 0.73.

Random forest captures non-linear degradation slightly better, but stepwise
gives an explicit equation that is easy to inspect.

## Classification — engine health

Binary label: `1` if remaining TTF ≤ 30 cycles. Two models on z-scored
features:

- **Logistic regression** with a probability threshold tuned by 10-fold
  engine-level cross-validation, targeting recall ≥ 0.95. The per-fold
  thresholds are averaged into a single operating threshold.
- **KNN** with K chosen in `[35, 55]` by the mean F1 across the same
  engine-level folds, then a final threshold from Youden's J on the training
  set.

Test results: KNN reaches recall = 0.96 / F1 = 0.87 against logistic
regression's 0.84 / 0.82, at comparable precision and AUC. The recall-first
threshold tuning is what keeps undetected failures low — the safety-critical
behaviour for aviation maintenance.

## Limitations

- Single operating regime and a small test fleet — generalisation untested
- Single-cycle classification ignores temporal context (LSTM / 1D-CNN next)
- Imbalance handled at threshold level, not in the loss function
