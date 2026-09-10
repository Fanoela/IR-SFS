"""
run_experiment.py
=================
Reproduces the photometric redshift experiment from the thesis:

    RABEMANOTRONA Fanoela (2025). Amélioration de l'algorithme de sélection
    séquentielle avant par remplacement itératif en apprentissage automatique.
    ESPA, Université d'Antananarivo.

Dataset: QSO from SDSS DR12 cross-matched with WISE, 2MASS, GALEX.
         Provided by Dr. Stephen Curran (Victoria University of Wellington).

Usage
-----
    python scripts/run_experiment.py --data /path/to/all_mags.dat
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# ── make imports work whether run from root or scripts/ ──────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ir_sfs.data      import load_qso_data, split_and_normalise
from ir_sfs.algorithm import fit as ir_sfs_fit, sfs
from ir_sfs.metrics   import evaluate


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="IR-SFS photometric redshift experiment")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the QSO data file (e.g. all_mags.dat)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of neighbours for kNN (default: 5)",
    )
    parser.add_argument(
        "--max_no_improvement", type=int, default=5,
        help="IR stopping criterion: consecutive positions with no gain (default: 5)",
    )
    parser.add_argument(
        "--moving_avg_window", type=int, default=3,
        help="IR stopping criterion: moving-average window size (default: 3)",
    )
    parser.add_argument(
        "--moving_avg_threshold", type=float, default=1e-4,
        help="IR stopping criterion: moving-average tolerance δ (default: 1e-4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Load & prepare data ─────────────────────────────────────────────────
    print(f"\nLoading data from: {args.data}")
    df = load_qso_data(args.data)
    print(f"Dataset shape: {df.shape}  ({df.shape[0]} QSOs, {df.shape[1]} columns)")

    y = df["redshift"]
    X = df.drop(columns=["redshift"])

    splits = split_and_normalise(X, y, random_state=args.seed)
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]
    y_test  = splits["y_test"]

    print(
        f"Split sizes — train: {len(y_train)} | "
        f"val: {len(y_val)} | test: {len(y_test)}"
    )

    estimator = KNeighborsRegressor(n_neighbors=args.k)

    # ── IR-SFS ──────────────────────────────────────────────────────────────
    result = ir_sfs_fit(
        X_train, y_train, X_val, y_val,
        estimator=estimator,
        max_no_improvement=args.max_no_improvement,
        moving_avg_window=args.moving_avg_window,
        moving_avg_threshold=args.moving_avg_threshold,
        verbose=True,
    )

    sfs_features = result["sfs_features"]
    ir_features  = result["ir_features"]

    # ── Validation evaluation ────────────────────────────────────────────────
    print("\n── Validation set ──────────────────────────────────────────")

    knn_sfs = KNeighborsRegressor(n_neighbors=args.k)
    knn_sfs.fit(X_train[sfs_features], y_train)
    evaluate(y_val, knn_sfs.predict(X_val[sfs_features]), label="SFS  (val)")

    knn_ir = KNeighborsRegressor(n_neighbors=args.k)
    knn_ir.fit(X_train[ir_features], y_train)
    evaluate(y_val, knn_ir.predict(X_val[ir_features]),  label="IR-SFS (val)")

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("── Test set ────────────────────────────────────────────────")

    evaluate(y_test, knn_sfs.predict(X_test[sfs_features]), label="SFS  (test)")
    evaluate(y_test, knn_ir.predict(X_test[ir_features]),   label="IR-SFS (test)")

    # ── Feature summary ──────────────────────────────────────────────────────
    print("\n── Feature sets ────────────────────────────────────────────")
    print(f"SFS    ({len(sfs_features)} features): {sfs_features}")
    print(f"IR-SFS ({len(ir_features)} features): {ir_features}")

    replaced = set(sfs_features) - set(ir_features)
    added    = set(ir_features)  - set(sfs_features)
    if replaced:
        print(f"\nReplaced by IR phase : {replaced} → {added}")
    else:
        print("\nIR phase made no replacements.")


if __name__ == "__main__":
    main()
