"""Core domain models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class FindingCategory(StrEnum):
    DATASET_QUALITY = "dataset_quality"
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    CONSTANT_FEATURES = "constant_features"
    TARGET_IMBALANCE = "target_imbalance"
    OUTLIERS = "outliers"
    CORRELATION = "correlation"
    LEAKAGE = "potential_leakage"
    CONTAMINATION = "contamination"
    DISTRIBUTION_SHIFT = "distribution_shift"
    SUSPICIOUS_FEATURES = "suspicious_features"
    SCHEMA = "schema"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class Finding:
    detector: str
    category: FindingCategory
    severity: FindingSeverity
    title: str
    description: str
    affected_features: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        data["severity"] = self.severity.value
        return data


@dataclass(slots=True)
class DatasetSummary:
    rows: int
    columns: int
    numeric_columns: int
    categorical_columns: int
    missing_cells: int


@dataclass(slots=True)
class RiskScore:
    total_score: float
    risk_level: RiskLevel
    severity_counts: dict[str, int]


@dataclass(slots=True)
class AuditReport:
    schema_version: str
    summary: dict[str, Any]
    findings: list[Finding]
    risk: RiskScore
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_level(self) -> RiskLevel:
        return self.risk.risk_level

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": self.summary,
            "findings": [finding.to_dict() for finding in self.findings],
            "risk": {
                "total_score": self.risk.total_score,
                "risk_level": self.risk.risk_level.value,
                "severity_counts": self.risk.severity_counts,
            },
            "metadata": self.metadata,
        }
