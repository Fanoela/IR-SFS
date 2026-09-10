"""
IR-SFS: Iterative Replacement Sequential Forward Selection
==========================================================
Core algorithm implementation.

Reference:
    RABEMANOTRONA Fanoela (2025). Amélioration de l'algorithme de sélection
    séquentielle avant par remplacement itératif en apprentissage automatique :
    Application à l'estimation du redshift photométrique.
    Master thesis, ESPA, Université d'Antananarivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from typing import Optional


# ---------------------------------------------------------------------------
# Phase 1 — Sequential Forward Selection (SFS)
# ---------------------------------------------------------------------------

def sfs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    estimator=None,
    max_features: Optional[int] = None,
    verbose: bool = False,
) -> tuple[list[str], dict[str, float]]:
    """
    Sequential Forward Selection (SFS).

    Greedily adds one feature at a time, choosing the feature that
    minimises the validation MSE at each step.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (rows = samples, columns = feature names).
    y_train : array-like
        Training targets.
    X_val : pd.DataFrame
        Validation features.
    y_val : array-like
        Validation targets.
    estimator : sklearn estimator, optional
        Must implement fit/predict. Defaults to KNeighborsRegressor(n_neighbors=5).
    max_features : int, optional
        Stop after selecting this many features. Defaults to all.
    verbose : bool
        Print progress.

    Returns
    -------
    selected : list[str]
        Ordered list of selected feature names.
    history : dict[str, float]
        Mapping feature_name -> best MSE after it was added.
    """
    if estimator is None:
        estimator = KNeighborsRegressor(n_neighbors=5)

    selected: list[str] = []
    remaining = list(X_train.columns)
    best_mse = np.inf
    history: dict[str, float] = {}

    while remaining:
        if max_features and len(selected) >= max_features:
            break

        step_best_feature = None
        step_best_mse = np.inf

        for feature in remaining:
            candidate = selected + [feature]
            model = clone(estimator)
            model.fit(X_train[candidate], y_train)
            mse = mean_squared_error(y_val, model.predict(X_val[candidate]))

            if mse < step_best_mse:
                step_best_mse = mse
                step_best_feature = feature

        # Stop if no feature improves the score
        if step_best_feature is None or step_best_mse >= best_mse:
            break

        best_mse = step_best_mse
        selected.append(step_best_feature)
        remaining.remove(step_best_feature)
        history[step_best_feature] = best_mse

        if verbose:
            print(
                f"  [SFS] step {len(selected):2d} | "
                f"added '{step_best_feature}' | MSE={best_mse:.6f} | "
                f"RMSE={np.sqrt(best_mse):.6f}"
            )

    return selected, history


# ---------------------------------------------------------------------------
# Phase 2 — Iterative Replacement (IR)
# ---------------------------------------------------------------------------

def _moving_average_criterion(
    deltas: deque,
    window: int,
    threshold: float,
) -> bool:
    """
    Return True (stop) when the moving average of recent performance
    variations falls below `threshold`.

    Parameters
    ----------
    deltas : deque
        Recent absolute MSE improvements (must have at least `window` entries).
    window : int
        Size of the moving-average window.
    threshold : float
        Tolerance below which gains are considered negligible.
    """
    if len(deltas) < window:
        return False
    avg = sum(list(deltas)[-window:]) / window
    return avg < threshold


def ir_sfs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    selected_features: list[str],
    all_features: list[str],
    estimator=None,
    max_no_improvement: Optional[int] = 5,
    moving_avg_window: int = 3,
    moving_avg_threshold: float = 1e-4,
    verbose: bool = False,
) -> tuple[list[str], float]:
    """
    Iterative Replacement phase of IR-SFS.

    For each feature in the current selection, tries every unselected
    feature as a replacement. Keeps the swap if it reduces the
    validation MSE. Stops early via two complementary criteria:

    1. ``max_no_improvement``: consecutive positions with no improvement.
    2. Moving-average criterion: the average absolute MSE gain over the
       last ``moving_avg_window`` steps falls below ``moving_avg_threshold``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : array-like
        Training targets.
    X_val : pd.DataFrame
        Validation features.
    y_val : array-like
        Validation targets.
    selected_features : list[str]
        Initial subset produced by SFS (will NOT be mutated).
    all_features : list[str]
        Full list of available feature names.
    estimator : sklearn estimator, optional
        Defaults to KNeighborsRegressor(n_neighbors=5).
    max_no_improvement : int, optional
        Stop after this many consecutive positions with no replacement.
        Set to None to disable this criterion.
    moving_avg_window : int
        Window size for the moving-average stopping criterion.
    moving_avg_threshold : float
        Tolerance for the moving-average criterion (δ in the thesis).
    verbose : bool
        Print replacement decisions.

    Returns
    -------
    optimized : list[str]
        Refined feature subset.
    best_mse : float
        Best MSE achieved on the validation set.
    """
    if estimator is None:
        estimator = KNeighborsRegressor(n_neighbors=5)

    S = selected_features.copy()
    remaining = [f for f in all_features if f not in S]

    # Compute initial MSE
    model = clone(estimator)
    model.fit(X_train[S], y_train)
    best_mse = mean_squared_error(y_val, model.predict(X_val[S]))

    no_improvement_count = 0
    deltas: deque = deque()

    for i in range(len(S)):
        best_feature_for_position = S[i]
        mse_before = best_mse

        for f in remaining:
            temp = S.copy()
            temp[i] = f

            m = clone(estimator)
            m.fit(X_train[temp], y_train)
            mse_temp = mean_squared_error(y_val, m.predict(X_val[temp]))

            if mse_temp < best_mse:
                best_mse = mse_temp
                best_feature_for_position = f

        # ---- Evaluate the outcome for position i ----
        improvement = mse_before - best_mse

        if best_feature_for_position != S[i]:
            if verbose:
                print(
                    f"  [IR ] pos {i:2d} | '{S[i]}' → '{best_feature_for_position}' "
                    f"| ΔMSE={improvement:.6f} | MSE={best_mse:.6f}"
                )
            remaining.remove(best_feature_for_position)
            remaining.append(S[i])   # freed feature goes back to pool
            S[i] = best_feature_for_position
            no_improvement_count = 0
        else:
            if verbose:
                print(f"  [IR ] pos {i:2d} | no improvement (count={no_improvement_count + 1})")
            no_improvement_count += 1

        deltas.append(abs(improvement))

        # ---- Stopping criteria ----
        if max_no_improvement is not None and no_improvement_count >= max_no_improvement:
            if verbose:
                print(f"  [IR ] early stop — {max_no_improvement} consecutive positions with no gain.")
            break

        if _moving_average_criterion(deltas, moving_avg_window, moving_avg_threshold):
            if verbose:
                print(
                    f"  [IR ] early stop — moving average of gains "
                    f"({sum(list(deltas)[-moving_avg_window:]) / moving_avg_window:.2e}) "
                    f"< threshold ({moving_avg_threshold:.2e})."
                )
            break

    return S, best_mse


# ---------------------------------------------------------------------------
# Combined pipeline: SFS → IR-SFS
# ---------------------------------------------------------------------------

def fit(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    estimator=None,
    max_features: Optional[int] = None,
    max_no_improvement: Optional[int] = 5,
    moving_avg_window: int = 3,
    moving_avg_threshold: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Full IR-SFS pipeline: SFS initialisation followed by iterative replacement.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : as above.
    estimator : sklearn estimator, optional.
    max_features : int, optional
        Cap on features selected during SFS phase.
    max_no_improvement : int, optional
        IR stopping criterion 1 (consecutive positions with no gain).
    moving_avg_window : int
        IR stopping criterion 2 — window size.
    moving_avg_threshold : float
        IR stopping criterion 2 — tolerance δ.
    verbose : bool

    Returns
    -------
    result : dict with keys
        ``sfs_features``    — subset after SFS phase
        ``sfs_mse``         — MSE after SFS phase
        ``ir_features``     — subset after IR phase
        ``ir_mse``          — MSE after IR phase
        ``sfs_history``     — step-by-step MSE during SFS
    """
    if estimator is None:
        estimator = KNeighborsRegressor(n_neighbors=5)

    all_features = list(X_train.columns)

    # --- Phase 1 ---
    if verbose:
        print("=" * 60)
        print("Phase 1 — Sequential Forward Selection (SFS)")
        print("=" * 60)

    sfs_features, sfs_history = sfs(
        X_train, y_train, X_val, y_val,
        estimator=estimator,
        max_features=max_features,
        verbose=verbose,
    )
    sfs_mse = sfs_history[sfs_features[-1]] if sfs_features else np.inf

    if verbose:
        print(f"\nSFS result  : {sfs_features}")
        print(f"SFS MSE     : {sfs_mse:.6f}  |  RMSE: {np.sqrt(sfs_mse):.6f}\n")

    # --- Phase 2 ---
    if verbose:
        print("=" * 60)
        print("Phase 2 — Iterative Replacement (IR)")
        print("=" * 60)

    ir_features, ir_mse = ir_sfs(
        X_train, y_train, X_val, y_val,
        selected_features=sfs_features,
        all_features=all_features,
        estimator=estimator,
        max_no_improvement=max_no_improvement,
        moving_avg_window=moving_avg_window,
        moving_avg_threshold=moving_avg_threshold,
        verbose=verbose,
    )

    if verbose:
        print(f"\nIR-SFS result : {ir_features}")
        print(f"IR-SFS MSE    : {ir_mse:.6f}  |  RMSE: {np.sqrt(ir_mse):.6f}")
        gain = np.sqrt(sfs_mse) - np.sqrt(ir_mse)
        print(f"RMSE gain vs SFS : {gain:+.6f}")
        print("=" * 60)

    return {
        "sfs_features": sfs_features,
        "sfs_mse": sfs_mse,
        "ir_features": ir_features,
        "ir_mse": ir_mse,
        "sfs_history": sfs_history,
    }
