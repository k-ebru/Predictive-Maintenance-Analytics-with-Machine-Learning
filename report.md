# Methodology summary

## Data

A pre-selected subset of the NASA C-MAPSS turbofan engine dataset is used.
There are 20,631 training cycles across 100 engines and a single snapshot
per test engine for 100 test engines. Each row carries the engine id, the
cycle counter, four sensors (`s1` to `s4`) and, for the training set, the
remaining Time-to-Failure (TTF). The true test TTFs are kept separately in
`PM_truth.txt`.

Quick EDA findings:

* no missing sensor values, so no imputation is needed
* strong class imbalance once `TTF <= 30` is treated as faulty
* train and test sensor distributions overlap well, so the test data is
  representative
* monotone degradation trends are visible in all four sensors

## Preprocessing

The same pipeline feeds both tasks. First, a trailing moving-average filter
with window size 3 is applied to each sensor separately per engine. This
removes short-term noise while keeping the slow degradation pattern.

For regression, the training TTF is capped at 125 cycles. This is the
piecewise-linear RUL formulation (Heimes 2008) and pushes the model to
focus on the informative end-of-life region instead of the long healthy
plateau.

Z-score normalisation is fitted on the training data only and then applied
to the test set with the same mean and standard deviation. Only the
scale-sensitive models use the normalised inputs; random forest uses the
original scale because trees are scale-invariant.

For the test set, only the final observation of each engine is kept. That
row represents the operational moment when a maintenance decision would be
made.

## Regression for Time-to-Failure

Random forest is fitted with 500 bagged trees and a minimum leaf size of
120. On the test engines it gives MAE = 15.89 cycles, RMSE = 20.50,
R^2 = 0.74.

Stepwise quadratic regression starts from a constant model and iteratively
adds or drops terms based on their p-values. The candidate set includes
main effects, cycle-sensor interactions and a couple of squared terms. The
final formula has 11 selected terms and gives MAE = 16.63, RMSE = 20.65,
R^2 = 0.73 on the test set.

Random forest captures the non-linear pattern slightly better; stepwise
trades a small amount of accuracy for an explicit, inspectable equation.

## Classification for engine health

Labels are `1` if remaining TTF is at most 30 cycles, else `0`. Both
models use z-scored features.

Logistic regression is fit on the full training set. Its decision
threshold is chosen by 10-fold engine-level cross-validation: for each
fold, the threshold that gives recall above 0.95 is recorded, then the 10
fold thresholds are averaged into a single operating threshold. The
intuition is that missing a real failure is much more expensive than
issuing a false alarm.

KNN is tuned by trying K values from 35 to 55, again using the same 10
engine-level folds, and picking the K with the best mean F1. The final
threshold comes from Youden's J on the training set. The chosen K was 41.

Test set performance is:

| Model               | Accuracy | Precision | Recall | F1   | AUC  |
|---------------------|----------|-----------|--------|------|------|
| Logistic regression | 0.91     | 0.81      | 0.84   | 0.82 | 0.96 |
| KNN                 | 0.92     | 0.76      | 1.00   | 0.86 | 0.97 |

KNN catches more failures than logistic regression at a small precision
cost, which is the right trade-off in an aviation maintenance setting.

## Limitations

* The 100 training engines all share a similar operating regime; results
  may not transfer to other conditions.
* The classifiers look at a single cycle at a time; sequence models
  (LSTM, 1D CNN) would use the time-series structure directly.
* Class imbalance is addressed through threshold tuning; sample weighting
  or focal loss would be alternatives worth comparing.
