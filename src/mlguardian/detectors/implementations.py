"""Detector implementations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlguardian.detectors.base import AuditContext
from mlguardian.detectors.registry import register_detector
from mlguardian.models import Finding, FindingCategory, FindingSeverity


@dataclass(slots=True)
class _BaseDetector:
    name: str


def _severity_from_fraction(value: float, info: float, warning: float, high: float) -> FindingSeverity | None:
    if value >= high:
        return FindingSeverity.HIGH
    if value >= warning:
        return FindingSeverity.WARNING
    if value >= info:
        return FindingSeverity.INFO
    return None


@register_detector("dataset_quality")
def dataset_quality_detector() -> DatasetQualityDetector:
    return DatasetQualityDetector(name="dataset_quality")


@dataclass(slots=True)
class DatasetQualityDetector(_BaseDetector):
    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        findings: list[Finding] = []
        for split_name, df in (("train", train_df), ("test", test_df)):
            if df is None:
                continue
            if df.empty:
                findings.append(
                    Finding(
                        detector=self.name,
                        category=FindingCategory.DATASET_QUALITY,
                        severity=FindingSeverity.CRITICAL,
                        title=f"Empty {split_name} dataset",
                        description=f"The {split_name} dataset contains no rows.",
                        recommendation="Provide a non-empty dataset split.",
                    )
                )
        return findings


@register_detector("missing_values")
def missing_values_detector() -> MissingValuesDetector:
    return MissingValuesDetector(name="missing_values")


@dataclass(slots=True)
class MissingValuesDetector(_BaseDetector):
    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        df = train_df
        findings: list[Finding] = []
        missing_pct = df.isna().mean()
        high_missing_cols = []
        for col, ratio in missing_pct.items():
            sev = _severity_from_fraction(
                float(ratio),
                context.config.missing.info_threshold,
                context.config.missing.warning_threshold,
                context.config.missing.high_threshold,
            )
            if sev:
                if ratio >= context.config.missing.high_threshold:
                    high_missing_cols.append(col)
                findings.append(
                    Finding(
                        detector=self.name,
                        category=FindingCategory.MISSING_VALUES,
                        severity=sev,
                        title=f"Missing values in '{col}'",
                        description=f"Feature '{col}' has {ratio:.1%} missing values.",
                        affected_features=[col],
                        metrics={"missing_fraction": round(float(ratio), 4)},
                        recommendation="Assess imputation strategy and root cause of missingness.",
                    )
                )
        if context.target in df.columns and df[context.target].isna().any():
            findings.append(
                Finding(
                    detector=self.name,
                    category=FindingCategory.MISSING_VALUES,
                    severity=FindingSeverity.HIGH,
                    title="Missing values in target",
                    description="Target column contains missing values.",
                    affected_features=[context.target],
                    metrics={"missing_fraction": round(float(df[context.target].isna().mean()), 4)},
                    recommendation="Remove or label-target-impute rows with missing target values.",
                )
            )
        row_missingness = df.isna().mean(axis=1)
        high_rows = int((row_missingness >= context.config.missing.high_missing_row_threshold).sum())
        if high_rows:
            findings.append(
                Finding(
                    detector=self.name,
                    category=FindingCategory.MISSING_VALUES,
                    severity=FindingSeverity.WARNING,
                    title="Rows with high missingness",
                    description="Some rows have high missing-value ratios.",
                    metrics={"rows_above_threshold": high_rows},
                    recommendation="Review data collection for sparse records.",
                )
            )
        if len(high_missing_cols) >= max(1, df.shape[1] // 3):
            findings.append(
                Finding(
                    detector=self.name,
                    category=FindingCategory.MISSING_VALUES,
                    severity=FindingSeverity.HIGH,
                    title="Many features have excessive missingness",
                    description="A large subset of features exceeds high missingness threshold.",
                    affected_features=high_missing_cols,
                    metrics={"count": len(high_missing_cols)},
                    recommendation="Consider dropping or improving highly sparse features.",
                )
            )
        return findings


@register_detector("duplicates")
def duplicates_detector() -> DuplicatesDetector:
    return DuplicatesDetector(name="duplicates")


@dataclass(slots=True)
class DuplicatesDetector(_BaseDetector):
    def _hash_rows(self, df: pd.DataFrame) -> pd.Series:
        normalized = df.copy()
        for col in normalized.columns:
            series = normalized[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                normalized[col] = numeric.map(lambda v: "<NA>" if pd.isna(v) else f"{float(v):.12g}")
            else:
                normalized[col] = series.astype("string").str.strip().str.lower()
        packed = normalized.fillna("<NA>").astype(str).agg("|".join, axis=1)
        return packed.apply(lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest())

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        findings: list[Finding] = []
        train_dupes = int(train_df.duplicated().sum())
        if train_dupes:
            findings.append(
                Finding(
                    detector=self.name,
                    category=FindingCategory.DUPLICATES,
                    severity=FindingSeverity.WARNING,
                    title="Duplicate rows in training data",
                    description="Exact duplicate rows detected in train split.",
                    metrics={"duplicate_rows": train_dupes},
                    recommendation="Deduplicate train split and inspect data ingestion.",
                )
            )
        if test_df is not None:
            test_dupes = int(test_df.duplicated().sum())
            if test_dupes:
                findings.append(
                    Finding(
                        detector=self.name,
                        category=FindingCategory.DUPLICATES,
                        severity=FindingSeverity.INFO,
                        title="Duplicate rows in test data",
                        description="Exact duplicate rows detected in test split.",
                        metrics={"duplicate_rows": test_dupes},
                        recommendation="Confirm whether duplicate test rows are expected.",
                    )
                )
            train_hashes = set(self._hash_rows(train_df).tolist())
            overlap = int(self._hash_rows(test_df).isin(train_hashes).sum())
            if overlap:
                findings.append(
                    Finding(
                        detector="contamination",
                        category=FindingCategory.CONTAMINATION,
                        severity=FindingSeverity.HIGH,
                        title="Potential train/test contamination",
                        description="Rows overlap between train and test splits.",
                        metrics={"overlapping_rows": overlap},
                        recommendation="Rebuild split to eliminate duplicated rows across splits.",
                    )
                )
            if context.config.key and context.config.key in train_df.columns and context.config.key in test_df.columns:
                overlap_keys = int(
                    test_df[context.config.key].astype("string").isin(
                        train_df[context.config.key].astype("string")
                    ).sum()
                )
                if overlap_keys:
                    findings.append(
                        Finding(
                            detector="contamination",
                            category=FindingCategory.CONTAMINATION,
                            severity=FindingSeverity.HIGH,
                            title="Potential train/test key contamination",
                            description="Key-based overlap detected between train and test splits.",
                            affected_features=[context.config.key],
                            metrics={"overlapping_keys": overlap_keys},
                            recommendation="Re-split data ensuring key uniqueness across splits.",
                        )
                    )
        return findings


@register_detector("constant_features")
def constant_features_detector() -> ConstantFeaturesDetector:
    return ConstantFeaturesDetector(name="constant_features")


@dataclass(slots=True)
class ConstantFeaturesDetector(_BaseDetector):
    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        findings: list[Finding] = []
        for col in train_df.columns:
            unique = train_df[col].nunique(dropna=False)
            if unique <= 1:
                findings.append(
                    Finding(
                        detector=self.name,
                        category=FindingCategory.CONSTANT_FEATURES,
                        severity=FindingSeverity.WARNING,
                        title=f"Constant feature '{col}'",
                        description="Feature has a single value in train data.",
                        affected_features=[col],
                        recommendation="Drop this feature unless required for business logic.",
                    )
                )
                continue
            top = train_df[col].value_counts(dropna=False, normalize=True).iloc[0]
            if float(top) >= 0.98:
                findings.append(
                    Finding(
                        detector=self.name,
                        category=FindingCategory.CONSTANT_FEATURES,
                        severity=FindingSeverity.INFO,
                        title=f"Near-constant feature '{col}'",
                        description="A single value dominates this feature.",
                        affected_features=[col],
                        metrics={"dominant_ratio": round(float(top), 4)},
                        recommendation="Review predictive utility and potential sampling artifacts.",
                    )
                )
        return findings


@register_detector("target_imbalance")
def target_imbalance_detector() -> TargetImbalanceDetector:
    return TargetImbalanceDetector(name="target_imbalance")


@dataclass(slots=True)
class TargetImbalanceDetector(_BaseDetector):
    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        if context.target not in train_df.columns:
            return []
        target = train_df[context.target]
        inferred_classification = context.config.task_type == "classification" or (
            context.config.task_type is None
            and target.nunique(dropna=True) <= 20
            and (
                target.dtype == "object"
                or pd.api.types.is_integer_dtype(target)
                or pd.api.types.is_bool_dtype(target)
            )
        )
        if not inferred_classification:
            return []

        counts = target.value_counts(dropna=False)
        if counts.empty:
            return []
        ratio = float(counts.max() / max(1, counts.min()))
        severity = FindingSeverity.WARNING if ratio >= 4 else FindingSeverity.INFO if ratio >= 2 else None
        if severity is None:
            return []
        return [
            Finding(
                detector=self.name,
                category=FindingCategory.TARGET_IMBALANCE,
                severity=severity,
                title="Target class imbalance",
                description="Class distribution is imbalanced.",
                affected_features=[context.target],
                metrics={
                    "class_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
                    "imbalance_ratio": round(ratio, 3),
                },
                recommendation="Use stratified splits and consider class-weighting or resampling.",
            )
        ]


@register_detector("outliers")
def outliers_detector() -> OutlierDetector:
    return OutlierDetector(name="outliers")


@dataclass(slots=True)
class OutlierDetector(_BaseDetector):
    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        findings: list[Finding] = []
        numeric_cols = [c for c in train_df.select_dtypes(include=np.number).columns if c != context.target]
        for col in numeric_cols:
            series = train_df[col].dropna()
            if series.size < 5:
                continue
            if context.config.outlier.method == "iqr":
                q1, q3 = np.percentile(series, [25, 75])
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_count = int(((series < lower) | (series > upper)).sum())
            else:
                median = float(np.median(series))
                mad = float(np.median(np.abs(series - median)))
                if mad == 0:
                    continue
                robust_z = 0.6745 * (series - median) / mad
                outlier_count = int((np.abs(robust_z) > context.config.outlier.zscore_threshold).sum())
            frac = outlier_count / max(1, len(series))
            if frac >= 0.05:
                findings.append(
                    Finding(
                        detector=self.name,
                        category=FindingCategory.OUTLIERS,
                        severity=FindingSeverity.INFO,
                        title=f"Potential outliers in '{col}'",
                        description="Outliers detected with configured robust heuristic.",
                        affected_features=[col],
                        metrics={"outlier_fraction": round(frac, 4), "outlier_count": outlier_count},
                        recommendation="Investigate domain validity; do not auto-delete without review.",
                    )
                )
        return findings


@register_detector("correlation")
def correlation_detector() -> CorrelationDetector:
    return CorrelationDetector(name="correlation")


@dataclass(slots=True)
class CorrelationDetector(_BaseDetector):
    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        numeric = train_df.select_dtypes(include=np.number)
        if numeric.shape[1] < 2:
            return []
        corr = numeric.corr(method=context.config.correlation.method)
        threshold = context.config.correlation.warning_threshold
        flagged: list[tuple[str, str, float]] = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                value = float(corr.loc[a, b])
                if abs(value) >= threshold:
                    flagged.append((a, b, value))
        if not flagged:
            return []
        return [
            Finding(
                detector=self.name,
                category=FindingCategory.CORRELATION,
                severity=FindingSeverity.WARNING,
                title="Highly correlated numeric features",
                description="High absolute pairwise correlation detected (heuristic signal only).",
                affected_features=sorted({f for a, b, _ in flagged for f in (a, b)}),
                metrics={"pairs": [{"a": a, "b": b, "corr": round(v, 4)} for a, b, v in flagged[:50]]},
                recommendation="Review multicollinearity; correlation alone is not proof of redundancy.",
            )
        ]


@register_detector("potential_leakage")
def leakage_detector() -> PotentialLeakageDetector:
    return PotentialLeakageDetector(name="potential_leakage")


@dataclass(slots=True)
class PotentialLeakageDetector(_BaseDetector):
    LEAKY_NAME_HINTS = ("target", "label", "outcome", "leak", "future", "post")

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        if context.target not in train_df.columns:
            return []
        findings: list[Finding] = []
        target = train_df[context.target]
        for col in train_df.columns:
            if col == context.target:
                continue
            low_col = col.lower()
            if any(h in low_col for h in self.LEAKY_NAME_HINTS):
                findings.append(
                    Finding(
                        detector=self.name,
                        category=FindingCategory.LEAKAGE,
                        severity=FindingSeverity.WARNING,
                        title=f"Potential leakage-like feature name: '{col}'",
                        description="Feature name suggests post-outcome or label leakage possibility.",
                        affected_features=[col],
                        recommendation="Manually verify feature generation timing and semantics.",
                    )
                )

            if train_df[col].dtype == "object":
                token_hit_ratio = float(train_df[col].astype(str).str.contains(str(context.target), case=False, regex=False).mean())
                if token_hit_ratio >= 0.2:
                    findings.append(
                        Finding(
                            detector=self.name,
                            category=FindingCategory.LEAKAGE,
                            severity=FindingSeverity.WARNING,
                            title=f"Potential target-token leakage in '{col}'",
                            description="Feature values frequently contain target-like tokens.",
                            affected_features=[col],
                            metrics={"token_hit_ratio": round(token_hit_ratio, 4)},
                            recommendation="Inspect feature extraction to ensure target text did not leak into inputs.",
                        )
                    )
            if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(target):
                aligned = pd.DataFrame({"x": train_df[col], "y": target}).dropna()
                if len(aligned) > 3:
                    score = abs(float(aligned["x"].corr(aligned["y"])))
                    if score >= context.config.leakage.association_threshold:
                        findings.append(
                            Finding(
                                detector=self.name,
                                category=FindingCategory.LEAKAGE,
                                severity=FindingSeverity.HIGH,
                                title=f"Potential leakage via high target association: '{col}'",
                                description="Very high feature-target association found; this is a heuristic not proof.",
                                affected_features=[col],
                                metrics={"absolute_correlation": round(score, 4)},
                                recommendation="Verify feature availability time and derivation lineage.",
                            )
                        )
        return findings


@register_detector("distribution_shift")
def shift_detector() -> DistributionShiftDetector:
    return DistributionShiftDetector(name="distribution_shift")


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cutoffs = np.unique(np.quantile(expected, quantiles))
    if len(cutoffs) <= 2:
        return 0.0
    expected_bins = pd.cut(expected, bins=cutoffs, include_lowest=True)
    actual_bins = pd.cut(actual, bins=cutoffs, include_lowest=True)
    expected_dist = expected_bins.value_counts(normalize=True, sort=False)
    actual_dist = actual_bins.value_counts(normalize=True, sort=False)
    epsilon = 1e-6
    aligned = pd.DataFrame({"e": expected_dist, "a": actual_dist}).fillna(epsilon)
    return float(((aligned["a"] - aligned["e"]) * np.log((aligned["a"] + epsilon) / (aligned["e"] + epsilon))).sum())


@dataclass(slots=True)
class DistributionShiftDetector(_BaseDetector):
    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        if test_df is None:
            return []
        findings: list[Finding] = []
        shared_cols = [c for c in train_df.columns if c in test_df.columns and c != context.target]
        for col in shared_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(test_df[col]):
                psi = _psi(train_df[col], test_df[col])
            else:
                train_freq = train_df[col].astype("string").fillna("<NA>").value_counts(normalize=True)
                test_freq = test_df[col].astype("string").fillna("<NA>").value_counts(normalize=True)
                all_keys = train_freq.index.union(test_freq.index)
                epsilon = 1e-6
                e = train_freq.reindex(all_keys, fill_value=epsilon)
                a = test_freq.reindex(all_keys, fill_value=epsilon)
                psi = float(((a - e) * np.log((a + epsilon) / (e + epsilon))).sum())
            severity = None
            if psi > context.config.shift.psi_high:
                severity = FindingSeverity.HIGH
            elif psi >= context.config.shift.psi_moderate:
                severity = FindingSeverity.WARNING
            elif psi > 0:
                severity = FindingSeverity.INFO
            if severity:
                findings.append(
                    Finding(
                        detector=self.name,
                        category=FindingCategory.DISTRIBUTION_SHIFT,
                        severity=severity,
                        title=f"Distribution shift detected in '{col}'",
                        description="PSI-based shift signal detected; interpret with domain context.",
                        affected_features=[col],
                        metrics={"psi": round(psi, 4)},
                        recommendation="Review split strategy and monitor model robustness under shift.",
                    )
                )
        return findings


@register_detector("suspicious_features")
def suspicious_features_detector() -> SuspiciousFeaturesDetector:
    return SuspiciousFeaturesDetector(name="suspicious_features")


@dataclass(slots=True)
class SuspiciousFeaturesDetector(_BaseDetector):
    PATTERNS = ("id", "uuid", "ssn", "email", "phone")

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, context: AuditContext) -> list[Finding]:
        findings: list[Finding] = []
        for col in train_df.columns:
            low = col.lower()
            if any(p in low for p in self.PATTERNS):
                distinct_ratio = float(train_df[col].nunique(dropna=False) / max(1, len(train_df)))
                if distinct_ratio >= 0.9:
                    findings.append(
                        Finding(
                            detector=self.name,
                            category=FindingCategory.SUSPICIOUS_FEATURES,
                            severity=FindingSeverity.INFO,
                            title=f"Suspicious high-cardinality identifier-like feature: '{col}'",
                            description="Identifier-like feature may hurt generalization or leak membership signals.",
                            affected_features=[col],
                            metrics={"distinct_ratio": round(distinct_ratio, 4)},
                            recommendation="Validate whether identifier-like columns should be excluded.",
                        )
                    )
        return findings


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = int(df.select_dtypes(include=np.number).shape[1])
    categorical_cols = int(df.shape[1] - numeric_cols)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_cells": int(df.isna().sum().sum()),
    }
