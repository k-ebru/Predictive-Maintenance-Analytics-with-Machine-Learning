"""Regression models for Time-to-Failure prediction.

Two models are provided, matching the MATLAB study:
    * Random Forest regression (500 bagged trees, min leaf size 120)
    * Stepwise quadratic regression using a forward-selection routine

The stepwise routine is a Python equivalent of MATLAB's ``stepwiselm`` with
an upper model that allows main effects (cycle, s1-s4), pairwise
interactions involving ``cycle`` and a few squared terms.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train,
                        n_estimators=500, min_samples_leaf=120,
                        random_state=42):
    """Fit a bagged regression forest. Original features (not normalised)."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def predict_clip(model, X):
    """Predict and clip negative values to zero (TTF cannot be negative)."""
    preds = model.predict(X)
    preds[preds < 0] = 0
    return preds


# ---------------------------------------------------------------------------
# Stepwise quadratic regression
# ---------------------------------------------------------------------------

def _build_candidate_terms(feature_names):
    """Build the set of candidate terms used by the stepwise search.

    Upper model:  main effects + cycle*sensor interactions + cycle^2 + s1^2
    (matches the MATLAB configuration used in the original study).
    """
    main = list(feature_names)
    interactions = [f"cycle:{s}" for s in feature_names if s != "cycle"]
    # also include sensor-sensor interaction s1*s2 (showed up in the
    # MATLAB-selected formula)
    interactions += [f"{a}:{b}" for a, b in combinations(
        [s for s in feature_names if s != "cycle"], 2)]
    squared = [f"{n}^2" for n in ["cycle", "s4"]]
    return main + interactions + squared


def _term_to_design(df, term):
    if "^2" in term:
        base = term[:-2]
        return df[base] ** 2
    if ":" in term:
        a, b = term.split(":")
        return df[a] * df[b]
    return df[term]


def _design_matrix(df, terms):
    cols = {t: _term_to_design(df, t) for t in terms}
    X = pd.DataFrame(cols, index=df.index)
    return sm.add_constant(X, has_constant="add")


def stepwise_quadratic(X_train_norm, y_train, p_enter=0.05, p_leave=0.10,
                       feature_names=None, max_steps=40):
    """Forward-backward stepwise selection on quadratic candidate terms.

    Returns the fitted statsmodels OLS results and the list of selected
    terms (excluding the intercept).
    """
    if feature_names is None:
        feature_names = list(X_train_norm.columns)

    candidates = _build_candidate_terms(feature_names)
    selected = []

    for _ in range(max_steps):
        improved = False

        # Forward step: try each remaining candidate, pick the one with the
        # smallest p-value if it is below p_enter.
        best_p = 1.0
        best_term = None
        for term in candidates:
            if term in selected:
                continue
            trial = selected + [term]
            X = _design_matrix(X_train_norm, trial)
            try:
                res = sm.OLS(y_train, X).fit()
            except Exception:
                continue
            p = res.pvalues.get(term, 1.0)
            if p < best_p:
                best_p, best_term = p, term
        if best_term is not None and best_p < p_enter:
            selected.append(best_term)
            improved = True

        # Backward step: drop the worst term if its p-value exceeds p_leave.
        if selected:
            X = _design_matrix(X_train_norm, selected)
            res = sm.OLS(y_train, X).fit()
            ps = res.pvalues.drop("const", errors="ignore")
            if len(ps) > 0 and ps.max() > p_leave:
                drop = ps.idxmax()
                selected.remove(drop)
                improved = True

        if not improved:
            break

    X_final = _design_matrix(X_train_norm, selected)
    model = sm.OLS(y_train, X_final).fit()
    return model, selected


def stepwise_predict(model, terms, X_norm):
    """Predict TTF using the fitted stepwise model."""
    X = _design_matrix(X_norm, terms)
    preds = np.asarray(model.predict(X), dtype=float).copy()
    preds[preds < 0] = 0
    return preds


def format_stepwise_formula(model, terms):
    """Return the selected regression equation as a readable string."""
    parts = [f"TTF = {model.params['const']:.4f}"]
    for t in terms:
        c = model.params[t]
        sign = "+" if c >= 0 else "-"
        parts.append(f" {sign} ({abs(c):.4f} * {t})")
    return "".join(parts)
