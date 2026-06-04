"""Data loading and preprocessing for the turbofan engine dataset."""

import numpy as np
import pandas as pd

SENSOR_COLS = ["s1", "s2", "s3", "s4"]
FEATURE_COLS = ["cycle"] + SENSOR_COLS

TTF_CAP = 125          # piecewise-linear RUL cap (Heimes, 2008)
FAULT_THRESHOLD = 30   # engine is "faulty" when remaining cycles <= 30
SMOOTH_WINDOW = 3


def load_data(data_dir="../data"):
    """Load the three input files into pandas objects."""
    train = pd.read_csv(f"{data_dir}/train_selected.csv")
    test = pd.read_csv(f"{data_dir}/test_selected.csv")
    y_true_test = np.loadtxt(f"{data_dir}/PM_truth.txt")
    return train, test, y_true_test


def trailing_moving_average(values, window=SMOOTH_WINDOW):
    """Trailing moving-average filter (causal): the smoothed value at time t
    uses samples t-(w-1) .. t.

    Mirrors MATLAB ``movmean(x, [w-1, 0])``.
    """
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).mean().to_numpy()


def smooth_sensors_per_engine(df, sensor_cols=SENSOR_COLS, window=SMOOTH_WINDOW):
    """Apply the trailing moving average to each sensor, separately per
    engine id. Returns a new DataFrame."""
    df = df.copy()
    for col in sensor_cols:
        df[col] = (
            df.groupby("id")[col]
              .transform(lambda x: trailing_moving_average(x.values, window))
        )
    return df


def cap_ttf(y, cap=TTF_CAP):
    """Clip TTF at the piecewise-linear cap."""
    y = np.asarray(y, dtype=float).copy()
    y[y > cap] = cap
    return y


def fit_zscore(X):
    """Fit mean/std on training data."""
    mu = X.mean(axis=0).values
    sigma = X.std(axis=0, ddof=1).values   # MATLAB std uses N-1
    return mu, sigma


def apply_zscore(X, mu, sigma):
    """Apply z-score transform using pre-computed statistics."""
    return (X - mu) / sigma


def last_cycle_per_engine(test_df, feature_cols=FEATURE_COLS):
    """For each engine in the test set, keep only the final observation.

    This represents the engine's current state at prediction time.
    """
    return (
        test_df.sort_values(["id", "cycle"])
               .groupby("id", as_index=False)
               .tail(1)[["id"] + feature_cols]
               .reset_index(drop=True)
    )
