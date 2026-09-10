"""
data.py — Data loading, colour index computation, and preprocessing.

Designed for the QSO photometric redshift dataset used in the thesis
(SDSS DR12 cross-matched with WISE, 2MASS, GALEX).
Can be adapted to any multi-band photometric catalogue.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Band and colour definitions
# ---------------------------------------------------------------------------

BANDS = ["FUV", "NUV", "u", "g", "r", "i", "z", "W1", "W2"]

COLOUR_PAIRS = [
    ("FUV", "NUV"),
    ("NUV", "u"),
    ("u",   "g"),
    ("g",   "r"),
    ("r",   "i"),
    ("i",   "z"),
    ("z",   "W1"),
    ("W1",  "W2"),
]


def add_colour_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute colour indices (magnitude differences between adjacent bands)
    and append them as new columns.

    Colour index name convention: '<band1>-<band2>'

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns listed in BANDS.

    Returns
    -------
    pd.DataFrame
        Original dataframe with colour columns appended (in-place copy).
    """
    df = df.copy()
    for b1, b2 in COLOUR_PAIRS:
        col_name = f"{b1}-{b2}"
        df[col_name] = df[b1] - df[b2]
    return df


def load_qso_data(filepath: str) -> pd.DataFrame:
    """
    Load the QSO photometric dataset from a space-separated text file.

    Expected column order (no header in file):
        redshift  u  g  r  i  z  W1  W2  NUV  FUV

    Colour indices are computed and appended automatically.

    Parameters
    ----------
    filepath : str
        Path to the data file (e.g. 'all_mags.dat').

    Returns
    -------
    pd.DataFrame
        DataFrame with bands, colours, and redshift column.
    """
    columns = ["redshift"] + BANDS[2:][::-1] + BANDS[:2][::-1]
    # actual order in the Curran dataset file:
    columns = ["redshift", "u", "g", "r", "i", "z", "W1", "W2", "NUV", "FUV"]

    df = pd.read_csv(filepath, delim_whitespace=True, names=columns)
    df = add_colour_indices(df)
    return df


def split_and_normalise(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Split data into train / validation / test sets and apply
    StandardScaler (fit on train only, applied to val and test).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or array-like
        Target values.
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.
    random_state : int

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test  — normalised DataFrames
        y_train, y_val, y_test  — Series / arrays
        scaler                  — fitted StandardScaler
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "train_ratio + val_ratio + test_ratio must equal 1."

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
    )

    feature_names = list(X.columns)
    scaler = StandardScaler()

    X_train_norm = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_val_norm = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_names,
        index=X_val.index,
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

    return {
        "X_train": X_train_norm,
        "X_val":   X_val_norm,
        "X_test":  X_test_norm,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "scaler":  scaler,
    }
