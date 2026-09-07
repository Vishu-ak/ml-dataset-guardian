"""Schema validation routines."""

from __future__ import annotations

import pandas as pd

from mlguardian.models import Finding, FindingCategory, FindingSeverity


def validate_schema(train_df: pd.DataFrame, test_df: pd.DataFrame | None, target: str) -> list[Finding]:

    findings: list[Finding] = []

    if target not in train_df.columns:
        findings.append(
            Finding(
                detector="schema_validator",
                category=FindingCategory.SCHEMA,
                severity=FindingSeverity.CRITICAL,
                title="Target missing in training dataset",
                description=f"Target column '{target}' is not present in train dataset.",
                recommendation="Provide a valid target column.",
            )
        )
        return findings

    if test_df is not None:
        missing_in_test = sorted(set(train_df.columns) - set(test_df.columns))
        extra_in_test = sorted(set(test_df.columns) - set(train_df.columns))
        if missing_in_test or extra_in_test:
            findings.append(
                Finding(
                    detector="schema_validator",
                    category=FindingCategory.SCHEMA,
                    severity=FindingSeverity.HIGH,
                    title="Train/test schema mismatch",
                    description="Train and test columns differ.",
                    affected_features=missing_in_test + extra_in_test,
                    metrics={"missing_in_test": missing_in_test, "extra_in_test": extra_in_test},
                    recommendation="Align feature columns between splits before training.",
                )
            )

        common = [c for c in train_df.columns if c in test_df.columns and c != target]
        mismatches: dict[str, str] = {}
        for column in common:
            if str(train_df[column].dtype) != str(test_df[column].dtype):
                mismatches[column] = f"train={train_df[column].dtype}, test={test_df[column].dtype}"
        if mismatches:
            findings.append(
                Finding(
                    detector="schema_validator",
                    category=FindingCategory.SCHEMA,
                    severity=FindingSeverity.WARNING,
                    title="Train/test dtype mismatch",
                    description="Some shared columns have different dtypes.",
                    affected_features=list(mismatches),
                    metrics={"dtype_mismatches": mismatches},
                    recommendation="Cast columns to consistent dtypes before training.",
                )
            )

    return findings
