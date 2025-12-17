from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class WorkbookInfo:
    path: Path
    sheets: List[str]


def list_sheets(file_path: str | Path) -> List[str]:
    xl = pd.ExcelFile(str(file_path))
    return list(xl.sheet_names)


def list_columns(file_path: str | Path, sheet_name: str) -> List[str]:
    df = pd.read_excel(str(file_path), sheet_name=sheet_name, nrows=0)
    return [str(c) for c in df.columns]

