from pathlib import Path

from mlguardian import DatasetAuditor
from mlguardian.models import RiskLevel

FIXTURES = Path(__file__).parent / "fixtures"


def test_audit_flow_returns_report_with_findings() -> None:
    auditor = DatasetAuditor(target="churn")
    report = auditor.audit(train=str(FIXTURES / "train.csv"), test=str(FIXTURES / "test.csv"))

    assert report.schema_version == "1.0.0"
    assert report.findings
    assert report.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert any(f.category.value == "missing_values" for f in report.findings)
    assert any(f.category.value == "contamination" for f in report.findings)
    assert any(f.category.value == "distribution_shift" for f in report.findings)
