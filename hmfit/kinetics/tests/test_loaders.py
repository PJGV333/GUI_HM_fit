"""Tests for kinetics data loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hmfit.kinetics.data.loaders import load_matrix_file, load_xlsx

DATA_DIR = Path(__file__).parent / "data"


def test_load_csv_matrix_numeric_header() -> None:
    path = DATA_DIR / "matrix_numeric.csv"
    t, D, x, labels = load_matrix_file(path)

    assert np.allclose(t, [0.0, 1.0, 2.0])
    assert D.shape == (3, 3)
    assert labels == ["400", "410", "420"]
    assert x is not None
    assert np.allclose(x, [400.0, 410.0, 420.0])


def test_load_csv_channels_named() -> None:
    path = DATA_DIR / "channels.csv"
    t, D, x, labels = load_matrix_file(path)

    assert np.allclose(t, [0.0, 1.0, 2.0])
    assert D.shape == (3, 2)
    assert labels == ["peak1", "peak2"]
    assert x is None


def test_load_xlsx_matrix_numeric_header() -> None:
    path = DATA_DIR / "matrix_numeric.xlsx"
    t, D, x, labels = load_xlsx(path, sheet="data")

    assert np.allclose(t, [0.0, 1.0, 2.0])
    assert D.shape == (3, 3)
    assert labels == ["400", "410", "420"]
    assert x is not None
    assert np.allclose(x, [400.0, 410.0, 420.0])


def test_load_transpose_matrix() -> None:
    path = DATA_DIR / "transpose.csv"
    t, D, x, labels = load_matrix_file(path, transpose=True)

    assert np.allclose(t, [0.0, 1.0, 2.0, 3.0])
    assert D.shape == (4, 2)
    assert labels == ["0", "1"]
    assert x is not None
    assert np.allclose(x, [0.0, 1.0])


def test_load_nan_fails() -> None:
    path = DATA_DIR / "nan.csv"
    with pytest.raises(ValueError, match="NaN values"):
        load_matrix_file(path)


def test_load_non_monotonic_time_fails() -> None:
    path = DATA_DIR / "nonmonotonic.csv"
    with pytest.raises(ValueError, match="strictly increasing"):
        load_matrix_file(path)
