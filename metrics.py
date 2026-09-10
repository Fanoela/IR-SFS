"""
metrics.py — Evaluation metrics for photometric redshift estimation.

Includes standard regression metrics (RMSE, R²) and the two
astronomy-specific quality indicators from Fotopoulou & Paltani (2018)
used in the thesis experiments.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    return float(r2_score(y_true, y_pred))


def sigma_nmad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalised Median Absolute Deviation (σ_NMAD).

    Standard accuracy metric for photometric redshift estimation.
    Robust to outliers — preferred over standard deviation.

    σ_NMAD = 1.48 × median( |z_phot - z_spec| / (1 + z_spec) )

    Parameters
    ----------
    y_true : array-like
        Spectroscopic redshifts (z_spec).
    y_pred : array-like
        Photometric redshift estimates (z_phot).

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    delta = np.abs(y_pred - y_true) / (1.0 + y_true)
    return float(1.48 * np.median(delta))


def catastrophic_outlier_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of catastrophic outliers (f_out).

    A prediction is a catastrophic outlier when:
        |z_phot - z_spec| / (1 + z_spec) > 0.15

    Parameters
    ----------
    y_true : array-like
        Spectroscopic redshifts.
    y_pred : array-like
        Photometric redshift estimates.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    delta = np.abs(y_pred - y_true) / (1.0 + y_true)
    return float(np.mean(delta > 0.15))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
    print_report: bool = True,
) -> dict[str, float]:
    """
    Compute all four metrics and optionally print a summary.

    Parameters
    ----------
    y_true : array-like
        Ground-truth redshifts.
    y_pred : array-like
        Predicted redshifts.
    label : str
        Optional label printed in the summary header.
    print_report : bool
        Whether to print the summary table.

    Returns
    -------
    dict with keys: ``rmse``, ``r2``, ``sigma_nmad``, ``f_out``
    """
    scores = {
        "rmse":        rmse(y_true, y_pred),
        "r2":          r2(y_true, y_pred),
        "sigma_nmad":  sigma_nmad(y_true, y_pred),
        "f_out":       catastrophic_outlier_rate(y_true, y_pred),
    }

    if print_report:
        header = f" {label} " if label else ""
        print(f"\n{'─' * 40}")
        print(f"  Evaluation report{header}")
        print(f"{'─' * 40}")
        print(f"  RMSE        : {scores['rmse']:.4f}")
        print(f"  R²          : {scores['r2']:.4f}")
        print(f"  σ_NMAD      : {scores['sigma_nmad']:.4f}")
        print(f"  f_out       : {scores['f_out']:.4f}")
        print(f"{'─' * 40}\n")

    return scores
