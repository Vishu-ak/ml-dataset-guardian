"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mlguardian.exceptions import ConfigurationError


@dataclass(slots=True)
class MissingConfig:
    info_threshold: float = 0.05
    warning_threshold: float = 0.20
    high_threshold: float = 0.40
    high_missing_row_threshold: float = 0.50


@dataclass(slots=True)
class CorrelationConfig:
    method: str = "pearson"
    warning_threshold: float = 0.95


@dataclass(slots=True)
class OutlierConfig:
    method: str = "iqr"
    zscore_threshold: float = 3.5


@dataclass(slots=True)
class ShiftConfig:
    psi_moderate: float = 0.10
    psi_high: float = 0.25


@dataclass(slots=True)
class LeakageConfig:
    association_threshold: float = 0.90
    enable_probe: bool = False


@dataclass(slots=True)
class AuditConfig:
    missing: MissingConfig = field(default_factory=MissingConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    max_rows_for_expensive_checks: int = 200_000
    sample_size: int = 50_000
    random_seed: int = 42
    fail_on: str = "critical"
    task_type: str | None = None
    key: str | None = None


def _merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None, overrides: dict[str, Any] | None = None) -> AuditConfig:
    data: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Config file does not exist: {config_path}")
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text()) or {}
        elif path.suffix == ".toml":
            data = tomllib.loads(path.read_text())
        else:
            raise ConfigurationError("Config must be YAML or TOML")
    if overrides:
        data = _merge_dict(data, overrides)

    try:
        missing = MissingConfig(**data.get("missing", {}))
        correlation = CorrelationConfig(**data.get("correlation", {}))
        outlier = OutlierConfig(**data.get("outlier", {}))
        shift = ShiftConfig(**data.get("shift", {}))
        leakage = LeakageConfig(**data.get("leakage", {}))
        config = AuditConfig(
            missing=missing,
            correlation=correlation,
            outlier=outlier,
            shift=shift,
            leakage=leakage,
            max_rows_for_expensive_checks=data.get("max_rows_for_expensive_checks", 200_000),
            sample_size=data.get("sample_size", 50_000),
            random_seed=data.get("random_seed", 42),
            fail_on=data.get("fail_on", "critical"),
            task_type=data.get("task_type"),
            key=data.get("key"),
        )
    except TypeError as exc:
        raise ConfigurationError(f"Invalid configuration values: {exc}") from exc

    if config.fail_on not in {"info", "warning", "high", "critical"}:
        raise ConfigurationError("fail_on must be one of: info, warning, high, critical")
    if config.correlation.method not in {"pearson", "spearman"}:
        raise ConfigurationError("correlation.method must be pearson or spearman")
    if config.outlier.method not in {"iqr", "mad_zscore"}:
        raise ConfigurationError("outlier.method must be iqr or mad_zscore")
    return config
