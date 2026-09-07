"""Detector protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from mlguardian.config import AuditConfig
from mlguardian.models import Finding


@dataclass(slots=True)
class AuditContext:
    target: str
    config: AuditConfig


class Detector(Protocol):
    name: str

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        """Run detector and return findings."""
