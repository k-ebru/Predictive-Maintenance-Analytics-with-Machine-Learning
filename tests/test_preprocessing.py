"""Tests for the preprocessing helpers."""

import numpy as np
import pandas as pd

from src.preprocessing import (
    trailing_moving_average,
    smooth_sensors_per_engine,
    cap_ttf,
    fit_zscore,
    apply_zscore,
    last_cycle_per_engine,
    TTF_CAP,
)


def test_trailing_moving_average_window3():
    # With window 3, the first value stays the same, the second is the mean of
    # the first two, then it becomes a true 3-point trailing average.
    out = trailing_moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
    np.testing.assert_allclose(out, [1.0, 1.5, 2.0, 3.0, 4.0])


def test_smoothing_is_per_engine():
    # Two engines with very different signals; smoothing on engine 1 must not
    # leak into engine 2.
    df = pd.DataFrame({
        "id":    [1, 1, 1, 2, 2, 2],
        "cycle": [1, 2, 3, 1, 2, 3],
        "s1":    [10, 10, 10, 100, 100, 100],
        "s2":    [0, 0, 0, 0, 0, 0],
        "s3":    [0, 0, 0, 0, 0, 0],
        "s4":    [0, 0, 0, 0, 0, 0],
    })
    out = smooth_sensors_per_engine(df)
    assert list(out.loc[out["id"] == 1, "s1"]) == [10, 10, 10]
    assert list(out.loc[out["id"] == 2, "s1"]) == [100, 100, 100]


def test_cap_ttf():
    y = np.array([50, 125, 200, 300])
    capped = cap_ttf(y, cap=TTF_CAP)
    assert capped.max() == TTF_CAP
    # Values already below the cap are untouched.
    assert capped[0] == 50


def test_zscore_round_trip():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0],
                      "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    mu, sigma = fit_zscore(X)
    Z = apply_zscore(X, mu, sigma)
    # z-scored columns have mean ~0 and sample std 1.
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(Z.std(axis=0, ddof=1), 1, atol=1e-9)


def test_last_cycle_per_engine_picks_max_cycle():
    df = pd.DataFrame({
        "id":    [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "s1": [0.0] * 5, "s2": [0.0] * 5, "s3": [0.0] * 5, "s4": [0.0] * 5,
    })
    last = last_cycle_per_engine(df)
    assert len(last) == 2
    assert sorted(last["cycle"]) == [2, 3]
