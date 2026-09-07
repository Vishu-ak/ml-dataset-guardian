"""Risk scoring logic."""

from __future__ import annotations

from collections import Counter

from mlguardian.models import Finding, FindingSeverity, RiskLevel, RiskScore

SEVERITY_WEIGHTS: dict[FindingSeverity, int] = {
    FindingSeverity.INFO: 1,
    FindingSeverity.WARNING: 3,
    FindingSeverity.HIGH: 6,
    FindingSeverity.CRITICAL: 10,
}


def compute_risk(findings: list[Finding]) -> RiskScore:
    severity_counts = Counter(f.severity.value for f in findings)
    if not findings:
        return RiskScore(total_score=0.0, risk_level=RiskLevel.LOW, severity_counts=dict(severity_counts))

    raw_score = sum(SEVERITY_WEIGHTS[f.severity] * max(1, len(f.affected_features)) for f in findings)
    normalized = raw_score / max(1, len(findings))

    if normalized >= 8:
        level = RiskLevel.CRITICAL
    elif normalized >= 5:
        level = RiskLevel.HIGH
    elif normalized >= 2:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    return RiskScore(total_score=round(normalized, 2), risk_level=level, severity_counts=dict(severity_counts))
