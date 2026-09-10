"""
ir_sfs — Iterative Replacement Sequential Forward Selection
===========================================================
A feature selection method that refines the greedy SFS algorithm
via an iterative replacement phase with adaptive stopping criteria.

Usage
-----
>>> from ir_sfs.algorithm import fit
>>> result = fit(X_train, y_train, X_val, y_val)
>>> print(result["ir_features"])
"""

from .algorithm import sfs, ir_sfs, fit
from .metrics import evaluate, rmse, r2, sigma_nmad, catastrophic_outlier_rate

__version__ = "1.0.0"
__author__  = "RABEMANOTRONA Fanoela"
__all__ = [
    "sfs", "ir_sfs", "fit",
    "evaluate", "rmse", "r2", "sigma_nmad", "catastrophic_outlier_rate",
]
