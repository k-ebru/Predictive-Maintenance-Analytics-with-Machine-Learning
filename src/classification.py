"""Classification of engines into healthy / faulty states.

Two models are implemented (matching the MATLAB study):
    * Logistic regression with a recall-driven decision threshold
    * KNN with K tuned by 10-fold engine-level cross-validation and a
      threshold chosen using Youden's J statistic
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, f1_score
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Cross-validation helper (engine-level splits)
# ---------------------------------------------------------------------------

def engine_level_kfold(engine_ids, n_splits=10, random_state=42):
    """Yield (train_mask, val_mask) pairs split by engine id, not by row.

    This avoids leakage between cycles of the same engine.
    """
    unique_ids = np.array(sorted(set(engine_ids)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    engine_ids = np.asarray(engine_ids)
    for tr_idx, va_idx in kf.split(unique_ids):
        tr_ids = set(unique_ids[tr_idx])
        va_ids = set(unique_ids[va_idx])
        train_mask = np.array([i in tr_ids for i in engine_ids])
        val_mask = np.array([i in va_ids for i in engine_ids])
        yield train_mask, val_mask


# ---------------------------------------------------------------------------
# Logistic regression with recall-targeted threshold
# ---------------------------------------------------------------------------

def _threshold_for_recall(y_true, scores, target_recall=0.95):
    """Pick the largest threshold whose recall is still >= target_recall.

    If no threshold reaches the target, fall back to the threshold with the
    highest recall (i.e. Youden's J).
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    # tpr == recall for the positive class
    feasible = np.where(tpr >= target_recall)[0]
    if len(feasible) > 0:
        # roc_curve returns thresholds in decreasing order
        return thr[feasible[0]]
    j = tpr - fpr
    return thr[int(np.argmax(j))]


def tune_logistic_threshold(X, y, engine_ids,
                            n_splits=10, target_recall=0.95,
                            random_state=42):
    """Average the per-fold logistic regression thresholds chosen for the
    target recall.
    """
    thresholds = []
    for tr_mask, va_mask in engine_level_kfold(engine_ids, n_splits,
                                               random_state):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[tr_mask], y[tr_mask])
        probs = model.predict_proba(X[va_mask])[:, 1]
        thr = _threshold_for_recall(y[va_mask], probs,
                                    target_recall=target_recall)
        if np.isinf(thr):
            thr = 0.0
        thresholds.append(thr)
    return float(np.mean(thresholds))


def fit_logistic(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# KNN with K tuned by F1, then a Youden-J threshold
# ---------------------------------------------------------------------------

def tune_knn_k(X, y, engine_ids, k_range=range(35, 56),
               n_splits=10, random_state=42):
    """Pick the K value that maximises the mean F1-score across folds."""
    cv_scores = []
    for k in k_range:
        fold_f1 = []
        for tr_mask, va_mask in engine_level_kfold(engine_ids, n_splits,
                                                   random_state):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X[tr_mask], y[tr_mask])
            preds = knn.predict(X[va_mask])
            fold_f1.append(f1_score(y[va_mask], preds, zero_division=0))
        cv_scores.append(np.mean(fold_f1))
    best_k = list(k_range)[int(np.argmax(cv_scores))]
    return best_k, cv_scores


def youden_threshold(y_true, scores):
    """Threshold that maximises (TPR - FPR)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])
