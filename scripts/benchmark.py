"""Lightweight benchmark methodology runner.

Generates synthetic tabular data and records runtime for 10K, 100K, 1M rows.
This script reports timings only and does not claim benchmark universality.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlguardian import DatasetAuditor


def make_data(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": np.arange(rows),
            "churn": rng.choice(["yes", "no"], size=rows, p=[0.15, 0.85]),
            "feature_a": rng.normal(0, 1, size=rows),
            "feature_b": rng.normal(10, 5, size=rows),
            "feature_c": rng.choice(["a", "b", "c"], size=rows),
        }
    )


if __name__ == "__main__":
    auditor = DatasetAuditor(target="churn")
    for size in (10_000, 100_000, 1_000_000):
        train = make_data(size)
        test = make_data(max(1_000, size // 5))
        start = time.perf_counter()
        auditor.audit(train=train, test=test)
        duration = time.perf_counter() - start
        print(f"rows={size} duration_seconds={duration:.3f}")
