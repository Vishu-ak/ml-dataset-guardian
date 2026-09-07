from mlguardian.models import Finding, FindingCategory, FindingSeverity, RiskLevel
from mlguardian.risk import compute_risk


def test_risk_scoring_levels() -> None:
    critical = Finding(
        detector="x",
        category=FindingCategory.SCHEMA,
        severity=FindingSeverity.CRITICAL,
        title="t",
        description="d",
    )
    warning = Finding(
        detector="x",
        category=FindingCategory.MISSING_VALUES,
        severity=FindingSeverity.WARNING,
        title="t",
        description="d",
        affected_features=["a", "b"],
    )
    report_score = compute_risk([critical, warning])
    assert report_score.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert report_score.total_score > 0
