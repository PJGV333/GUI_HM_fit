"""File loaders for kinetics datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def load_matrix_file(
    path: str | Path,
    *,
    delimiter: str | None = None,
    transpose: bool = False,
    time_col: int | str = 0,
    header: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    """Load a matrix file (CSV/TSV) into t, D, x, labels."""
    df = _load_dataframe(path, delimiter=delimiter, header=header)

    if transpose:
        df = df.transpose().reset_index()

    if not header:
        df.columns = [f"col_{idx}" for idx in range(df.shape[1])]

    time_idx = _resolve_time_column(df.columns, time_col)
    t = _coerce_numeric(df.iloc[:, time_idx].to_numpy(), "time column")
    data_df = df.drop(df.columns[time_idx], axis=1)
    D = _coerce_numeric(data_df.to_numpy(), "data matrix")

    _validate_time(t)
    labels = [str(col) for col in data_df.columns]
    x = _parse_axis(labels)
    return t, D, x, labels


def load_xlsx(
    path: str | Path,
    *,
    sheet: str = "data",
    transpose: bool = False,
    time_col: int | str = 0,
    header: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    """Load an Excel file (XLSX) into t, D, x, labels."""
    df = _load_dataframe(path, sheet=sheet, header=header)

    if transpose:
        df = df.transpose().reset_index()

    if not header:
        df.columns = [f"col_{idx}" for idx in range(df.shape[1])]

    time_idx = _resolve_time_column(df.columns, time_col)
    t = _coerce_numeric(df.iloc[:, time_idx].to_numpy(), "time column")
    data_df = df.drop(df.columns[time_idx], axis=1)
    D = _coerce_numeric(data_df.to_numpy(), "data matrix")

    _validate_time(t)
    labels = [str(col) for col in data_df.columns]
    x = _parse_axis(labels)
    return t, D, x, labels


def _load_dataframe(
    path: str | Path,
    *,
    delimiter: str | None = None,
    sheet: str | None = None,
    header: bool = True,
):
    import pandas as pd

    path = Path(path)
    if sheet is None:
        sep = delimiter
        engine = "python" if sep is None else "c"
        return pd.read_csv(path, sep=sep, header=0 if header else None, engine=engine)
    return pd.read_excel(path, sheet_name=sheet, header=0 if header else None)


def _resolve_time_column(columns: Iterable[object], time_col: int | str) -> int:
    if isinstance(time_col, int):
        return time_col
    columns_list = [str(col) for col in columns]
    if time_col not in columns_list:
        raise ValueError(f"Time column '{time_col}' not found.")
    return columns_list.index(time_col)


def _coerce_numeric(values: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"NaN values found in {label}.")
    return arr


def _validate_time(t: np.ndarray) -> None:
    if t.ndim != 1:
        t = t.reshape(-1)
    if t.size < 2:
        return
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time values must be strictly increasing.")


def _parse_axis(labels: Sequence[str]) -> np.ndarray | None:
    axis: list[float] = []
    for label in labels:
        try:
            axis.append(float(label))
        except ValueError:
            return None
    return np.asarray(axis, dtype=float)
