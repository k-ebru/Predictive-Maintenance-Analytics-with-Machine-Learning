"""Tests for the classification helpers.

The engine-level CV split is the key piece of correctness here: if a single
engine's rows ended up split across train and validation we would silently
leak time-series structure into the model evaluation.
"""

import numpy as np

from src.classification import (
    engine_level_kfold,
    _threshold_for_recall,
    youden_threshold,
)


def test_engine_kfold_no_overlap():
    engine_ids = np.repeat(np.arange(20), 5)  # 20 engines, 5 cycles each
    for train_mask, val_mask in engine_level_kfold(engine_ids, n_splits=5):
        train_engines = set(engine_ids[train_mask])
        val_engines = set(engine_ids[val_mask])
        # No engine appears in both halves of any fold.
        assert train_engines.isdisjoint(val_engines)
        # Every row is assigned to exactly one side.
        assert (train_mask | val_mask).all()
        assert not (train_mask & val_mask).any()


def test_engine_kfold_reproducible():
    ids = np.repeat(np.arange(20), 3)
    splits_a = [tuple(m.tolist() for m in pair)
                for pair in engine_level_kfold(ids, n_splits=4, random_state=42)]
    splits_b = [tuple(m.tolist() for m in pair)
                for pair in engine_level_kfold(ids, n_splits=4, random_state=42)]
    assert splits_a == splits_b


def test_threshold_for_recall_returns_high_recall_threshold():
    # Perfectly separable: positives have higher scores than negatives.
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = _threshold_for_recall(y, scores, target_recall=0.95)
    # All three positives must be classified positive at this threshold.
    preds = (scores >= thr).astype(int)
    recall = preds[y == 1].sum() / 3
    assert recall >= 0.95


def test_youden_threshold_on_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = youden_threshold(y, scores)
    # On separable data the optimal Youden threshold sits between the two
    # clusters and gives perfect classification.
    preds = (scores >= thr).astype(int)
    assert (preds == y).all()
