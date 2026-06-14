"""Tests for the regression helpers.

These cover the deterministic pieces: candidate term construction, design
matrix layout, and the prediction-clipping rule. Full model fitting is left
to the pipeline run.
"""

import numpy as np
import pandas as pd

from src.regression import (
    _build_candidate_terms,
    _design_matrix,
    train_random_forest,
    predict_clip,
)


def test_candidate_terms_include_expected_pieces():
    terms = _build_candidate_terms(["cycle", "s1", "s2", "s3", "s4"])
    assert "cycle" in terms
    assert "s1" in terms
    assert "cycle:s1" in terms
    assert "cycle^2" in terms
    assert "s4^2" in terms


def test_design_matrix_handles_squared_and_interactions():
    df = pd.DataFrame({"cycle": [1, 2, 3], "s1": [4, 5, 6], "s2": [7, 8, 9]})
    X = _design_matrix(df, ["cycle", "cycle^2", "cycle:s1"])
    # statsmodels adds an intercept column.
    assert "const" in X.columns
    np.testing.assert_array_equal(X["cycle^2"].values, [1, 4, 9])
    np.testing.assert_array_equal(X["cycle:s1"].values, [4, 10, 18])


def test_random_forest_is_reproducible():
    # The same seed should give bit-identical predictions on the same data.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    y = X.sum(axis=1) + rng.standard_normal(200) * 0.1

    rf_a = train_random_forest(X, y, n_estimators=20, min_samples_leaf=5,
                               random_state=42)
    rf_b = train_random_forest(X, y, n_estimators=20, min_samples_leaf=5,
                               random_state=42)
    # Floating-point summation order in the parallel forest is not bit
    # identical across runs, so allow a tiny tolerance.
    np.testing.assert_allclose(rf_a.predict(X), rf_b.predict(X), atol=1e-12)


def test_predict_clip_never_returns_negative():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 3))
    # Use a model that may produce negative outputs on this data.
    rf = train_random_forest(X, rng.standard_normal(50),
                             n_estimators=10, min_samples_leaf=2,
                             random_state=42)
    preds = predict_clip(rf, X)
    assert (preds >= 0).all()
