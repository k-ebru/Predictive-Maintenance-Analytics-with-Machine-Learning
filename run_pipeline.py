"""Reproduce the full analysis from the command line.

Runs the same pipeline used in the notebooks: preprocessing, both regression
models and both classifiers. Useful for a quick sanity check without opening
Jupyter.

    python run_pipeline.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.preprocessing import (
    load_data, smooth_sensors_per_engine, cap_ttf,
    fit_zscore, apply_zscore, last_cycle_per_engine,
    FEATURE_COLS, FAULT_THRESHOLD, TTF_CAP,
)
from src.regression import (
    train_random_forest, predict_clip,
    stepwise_quadratic, stepwise_predict, format_stepwise_formula,
)
from src.classification import (
    tune_logistic_threshold, fit_logistic,
    tune_knn_k, youden_threshold,
)
from src.evaluation import regression_metrics, classification_metrics
from sklearn.neighbors import KNeighborsClassifier

print("=" * 70)
print("LOADING DATA")
print("=" * 70)
train, test, y_true_test = load_data("data")
print(f"train: {train.shape},  test: {test.shape},  truth: {y_true_test.shape}")

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)
train_s = smooth_sensors_per_engine(train)
test_s = smooth_sensors_per_engine(test)
X_train = train_s[FEATURE_COLS]
y_train_reg = cap_ttf(train_s["ttf"].values, cap=TTF_CAP)
X_test_last = last_cycle_per_engine(test_s)[FEATURE_COLS]
y_true_test_capped = cap_ttf(y_true_test, cap=TTF_CAP)
print(f"X_train: {X_train.shape},  X_test (last cycle): {X_test_last.shape}")

print("\n" + "=" * 70)
print("REGRESSION: RANDOM FOREST")
print("=" * 70)
rf = train_random_forest(X_train, y_train_reg, n_estimators=500,
                         min_samples_leaf=120, random_state=42)
pred_rf_train = predict_clip(rf, X_train)
pred_rf_test = predict_clip(rf, X_test_last)
print("Train:", {k: round(v, 4) for k, v in
                 regression_metrics(y_train_reg, pred_rf_train).items()})
print("Test :", {k: round(v, 4) for k, v in
                 regression_metrics(y_true_test_capped, pred_rf_test).items()})

print("\n" + "=" * 70)
print("REGRESSION: STEPWISE QUADRATIC")
print("=" * 70)
mu, sigma = fit_zscore(X_train)
X_train_n = pd.DataFrame(apply_zscore(X_train.values, mu, sigma),
                         columns=FEATURE_COLS)
X_test_n = pd.DataFrame(apply_zscore(X_test_last.values, mu, sigma),
                        columns=FEATURE_COLS)
sw_model, sw_terms = stepwise_quadratic(X_train_n, y_train_reg,
                                        feature_names=FEATURE_COLS)
print(f"Selected terms ({len(sw_terms)}):", sw_terms)
print(format_stepwise_formula(sw_model, sw_terms))
pred_sw_train = stepwise_predict(sw_model, sw_terms, X_train_n)
pred_sw_test = stepwise_predict(sw_model, sw_terms, X_test_n)
print("Train:", {k: round(v, 4) for k, v in
                 regression_metrics(y_train_reg, pred_sw_train).items()})
print("Test :", {k: round(v, 4) for k, v in
                 regression_metrics(y_true_test_capped, pred_sw_test).items()})

print("\n" + "=" * 70)
print("CLASSIFICATION: LOGISTIC REGRESSION")
print("=" * 70)
y_train_clf = (train_s["ttf"] <= FAULT_THRESHOLD).astype(int).values
engine_ids_train = train_s["id"].values
y_test_clf = (y_true_test <= FAULT_THRESHOLD).astype(int)
X_train_n_arr = apply_zscore(X_train.values, mu, sigma)
X_test_n_arr = apply_zscore(X_test_last.values, mu, sigma)

log_thr = tune_logistic_threshold(X_train_n_arr, y_train_clf,
                                   engine_ids_train,
                                   n_splits=10, target_recall=0.95)
print(f"Threshold (recall>=0.95): {log_thr:.4f}")
log_model = fit_logistic(X_train_n_arr, y_train_clf)
prob_log_tr = log_model.predict_proba(X_train_n_arr)[:, 1]
prob_log_te = log_model.predict_proba(X_test_n_arr)[:, 1]
pred_log_tr = (prob_log_tr >= log_thr).astype(int)
pred_log_te = (prob_log_te >= log_thr).astype(int)
print("Train:", {k: round(v, 4) for k, v in
                 classification_metrics(y_train_clf, pred_log_tr,
                                        prob_log_tr).items()})
print("Test :", {k: round(v, 4) for k, v in
                 classification_metrics(y_test_clf, pred_log_te,
                                        prob_log_te).items()})

print("\n" + "=" * 70)
print("CLASSIFICATION: KNN")
print("=" * 70)
best_k, cv_scores = tune_knn_k(X_train_n_arr, y_train_clf,
                                engine_ids_train,
                                k_range=range(35, 56), n_splits=10)
print(f"Best K = {best_k}  (CV F1 = {max(cv_scores):.4f})")
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_n_arr, y_train_clf)
prob_knn_tr = knn.predict_proba(X_train_n_arr)[:, 1]
prob_knn_te = knn.predict_proba(X_test_n_arr)[:, 1]
knn_thr = youden_threshold(y_train_clf, prob_knn_tr)
print(f"KNN threshold (Youden): {knn_thr:.4f}")
pred_knn_tr = (prob_knn_tr >= knn_thr).astype(int)
pred_knn_te = (prob_knn_te >= knn_thr).astype(int)
print("Train:", {k: round(v, 4) for k, v in
                 classification_metrics(y_train_clf, pred_knn_tr,
                                        prob_knn_tr).items()})
print("Test :", {k: round(v, 4) for k, v in
                 classification_metrics(y_test_clf, pred_knn_te,
                                        prob_knn_te).items()})

print("\n" + "=" * 70)
print("OK: all models trained and evaluated end-to-end.")
print("=" * 70)
