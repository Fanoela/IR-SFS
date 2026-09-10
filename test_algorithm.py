"""
tests/test_algorithm.py
=======================
Unit tests for the IR-SFS algorithm.
Run with:  pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ir_sfs.algorithm import sfs, ir_sfs, fit
from ir_sfs.metrics   import rmse, sigma_nmad, catastrophic_outlier_rate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """
    Simple synthetic regression dataset:
    y = 2*x0 + x1 + noise
    Features x2..x4 are noise — SFS should prefer x0 and x1.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        rng.standard_normal((n, 5)),
        columns=["x0", "x1", "x2", "x3", "x4"],
    )
    y = 2 * X["x0"] + X["x1"] + 0.1 * rng.standard_normal(n)

    train_idx = range(0, 120)
    val_idx   = range(120, 160)
    test_idx  = range(160, 200)

    return (
        X.iloc[train_idx], y.iloc[train_idx],
        X.iloc[val_idx],   y.iloc[val_idx],
        X.iloc[test_idx],  y.iloc[test_idx],
    )


# ---------------------------------------------------------------------------
# SFS tests
# ---------------------------------------------------------------------------

class TestSFS:

    def test_returns_non_empty_subset(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        features, history = sfs(X_tr, y_tr, X_v, y_v)
        assert len(features) > 0

    def test_selects_informative_features_first(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        features, _ = sfs(X_tr, y_tr, X_v, y_v, max_features=2)
        # x0 and x1 carry the signal — both should appear in top-2
        assert "x0" in features
        assert "x1" in features

    def test_history_length_matches_features(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        features, history = sfs(X_tr, y_tr, X_v, y_v)
        assert len(history) == len(features)

    def test_mse_decreases_monotonically(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        _, history = sfs(X_tr, y_tr, X_v, y_v)
        mse_values = list(history.values())
        assert all(mse_values[i] >= mse_values[i + 1] for i in range(len(mse_values) - 1))

    def test_max_features_respected(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        features, _ = sfs(X_tr, y_tr, X_v, y_v, max_features=3)
        assert len(features) <= 3


# ---------------------------------------------------------------------------
# IR-SFS tests
# ---------------------------------------------------------------------------

class TestIRSFS:

    def test_output_same_length_as_input(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        initial, _ = sfs(X_tr, y_tr, X_v, y_v, max_features=3)
        optimized, _ = ir_sfs(
            X_tr, y_tr, X_v, y_v,
            selected_features=initial,
            all_features=list(X_tr.columns),
        )
        assert len(optimized) == len(initial)

    def test_mse_does_not_increase(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        initial, history = sfs(X_tr, y_tr, X_v, y_v, max_features=3)
        sfs_mse = history[initial[-1]]

        _, ir_mse = ir_sfs(
            X_tr, y_tr, X_v, y_v,
            selected_features=initial,
            all_features=list(X_tr.columns),
        )
        assert ir_mse <= sfs_mse + 1e-9

    def test_no_duplicate_features(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        initial, _ = sfs(X_tr, y_tr, X_v, y_v, max_features=3)
        optimized, _ = ir_sfs(
            X_tr, y_tr, X_v, y_v,
            selected_features=initial,
            all_features=list(X_tr.columns),
        )
        assert len(optimized) == len(set(optimized))

    def test_early_stopping_max_no_improvement(self, synthetic_data):
        """Algorithm must terminate even with max_no_improvement=1."""
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        initial, _ = sfs(X_tr, y_tr, X_v, y_v, max_features=3)
        optimized, _ = ir_sfs(
            X_tr, y_tr, X_v, y_v,
            selected_features=initial,
            all_features=list(X_tr.columns),
            max_no_improvement=1,
        )
        assert len(optimized) == len(initial)

    def test_does_not_mutate_input(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        initial, _ = sfs(X_tr, y_tr, X_v, y_v, max_features=3)
        original_copy = initial.copy()
        ir_sfs(
            X_tr, y_tr, X_v, y_v,
            selected_features=initial,
            all_features=list(X_tr.columns),
        )
        assert initial == original_copy


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestFit:

    def test_pipeline_returns_expected_keys(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        result = fit(X_tr, y_tr, X_v, y_v, verbose=False)
        for key in ["sfs_features", "sfs_mse", "ir_features", "ir_mse", "sfs_history"]:
            assert key in result

    def test_ir_mse_le_sfs_mse(self, synthetic_data):
        X_tr, y_tr, X_v, y_v, *_ = synthetic_data
        result = fit(X_tr, y_tr, X_v, y_v, verbose=False)
        assert result["ir_mse"] <= result["sfs_mse"] + 1e-9


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_rmse_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_sigma_nmad_perfect(self):
        y = np.array([0.5, 1.0, 2.0])
        assert sigma_nmad(y, y) == pytest.approx(0.0)

    def test_catastrophic_outlier_rate_none(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert catastrophic_outlier_rate(y_true, y_pred) == pytest.approx(0.0)

    def test_catastrophic_outlier_rate_all(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([5.0, 5.0, 5.0])   # delta >> 0.15
        assert catastrophic_outlier_rate(y_true, y_pred) == pytest.approx(1.0)
