"""Metric calculations and shared plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    """Return MAE, RMSE and R^2 as a dict."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, y_proba=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["AUC"] = roc_auc_score(y_true, y_proba)
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_prediction_band(y_pred, y_true, rmse, title, color="b", ax=None):
    """Predicted curve with a +/-1 RMSE band, overlaid with true values."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(y_pred))
    ax.fill_between(x, y_pred - rmse, y_pred + rmse,
                    color=color, alpha=0.2, label="+/-1 RMSE band")
    ax.plot(x, y_pred, color=color, linewidth=1.5, label="Prediction")
    ax.scatter(x, y_true, c="k", s=18, alpha=0.6, label="True TTF")
    ax.set_xlabel("Test engine index")
    ax.set_ylabel("TTF (cycles)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_residual_density(residuals_dict, ax=None):
    """Kernel-density overlay of residuals for several models."""
    from scipy.stats import gaussian_kde
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for label, (res, color) in residuals_dict.items():
        kde = gaussian_kde(res)
        xs = np.linspace(res.min() - 5, res.max() + 5, 400)
        ax.plot(xs, kde(xs), color=color, linewidth=2, label=label)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (Predicted - True)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_roc(curves, ax=None):
    """Plot ROC curves for several models. ``curves`` is a dict of
    ``{label: (y_true, scores)}``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for label, (y_true, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_confusion(y_true, y_pred, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Faulty"])
    ax.set_yticklabels(["Healthy", "Faulty"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return ax
