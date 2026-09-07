"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlguardian.exceptions import DatasetLoadError, UnsupportedFormatError

SUPPORTED_FORMATS = {".csv", ".parquet"}


def load_dataset(path: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise DatasetLoadError(f"Dataset not found: {path}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(f"Unsupported dataset format: {suffix}")

    try:
        if suffix == ".csv":
            return pd.read_csv(source)
        return pd.read_parquet(source)
    except Exception as exc:  # pragma: no cover - pandas exceptions vary
        raise DatasetLoadError(f"Failed to load dataset at {path}: {exc}") from exc
