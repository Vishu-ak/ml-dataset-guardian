"""Main audit orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mlguardian.config import AuditConfig, load_config
from mlguardian.detectors import AuditContext, build_detectors, summarize_dataset
from mlguardian.exceptions import AuditError, ConfigurationError, SchemaMismatchError
from mlguardian.loaders import load_dataset
from mlguardian.models import AuditReport
from mlguardian.risk import compute_risk
from mlguardian.schema import validate_schema

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetAuditor:
    target: str
    config: AuditConfig | None = None

    def __post_init__(self) -> None:
        self.config = self.config or load_config(None)

    def audit(
        self,
        train: str | pd.DataFrame,
        test: str | pd.DataFrame | None = None,
        config_path: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AuditReport:
        try:
            base_config = self.config
            if base_config is None:  # pragma: no cover - guaranteed by __post_init__
                base_config = load_config(None)
            runtime_config: AuditConfig = (
                base_config if config_path is None and not overrides else load_config(config_path, overrides)
            )
            train_df = load_dataset(train) if isinstance(train, str) else train.copy()
            test_df = load_dataset(test) if isinstance(test, str) else test.copy() if test is not None else None

            context = AuditContext(target=self.target, config=runtime_config)
            findings = validate_schema(train_df, test_df, self.target)
            if any(f.severity.value == "critical" for f in findings):
                raise SchemaMismatchError("Critical schema validation error")

            for detector in build_detectors():
                detector_findings = detector.run(train_df, test_df, context)
                findings.extend(detector_findings)

            summary = {
                "train": summarize_dataset(train_df),
                "test": summarize_dataset(test_df) if test_df is not None else None,
                "finding_count": len(findings),
            }
            risk = compute_risk(findings)
            report = AuditReport(
                schema_version="1.0.0",
                summary=summary,
                findings=findings,
                risk=risk,
                metadata={
                    "target": self.target,
                    "detectors_executed": [detector.name for detector in build_detectors()],
                },
            )
            return report
        except (ConfigurationError, SchemaMismatchError):
            raise
        except Exception as exc:  # pragma: no cover - safety net for CLI mapping
            logger.exception("Unexpected audit error")
            raise AuditError(str(exc)) from exc


def should_fail(report: AuditReport, threshold: str) -> bool:
    ordering = {"info": 0, "warning": 1, "high": 2, "critical": 3}
    threshold_value = ordering[threshold]
    for finding in report.findings:
        if ordering[finding.severity.value] >= threshold_value:
            return True
    return False


def save_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output)
