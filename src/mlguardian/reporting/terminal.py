"""Human-readable terminal report rendering."""

from __future__ import annotations

from collections import Counter

from mlguardian.models import AuditReport


def render_terminal(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append("ML Dataset Guardian Audit Report")
    lines.append("=" * 32)
    lines.append(f"Risk level: {report.risk.risk_level.value} (score={report.risk.total_score})")
    lines.append("")
    lines.append("Summary:")
    for key, value in report.summary.items():
        lines.append(f"  - {key}: {value}")

    severity_counts = Counter(f.severity.value for f in report.findings)
    lines.append("")
    lines.append("Findings by severity:")
    for sev in ["critical", "high", "warning", "info"]:
        lines.append(f"  - {sev}: {severity_counts.get(sev, 0)}")

    lines.append("")
    lines.append("Findings:")
    if not report.findings:
        lines.append("  No findings detected.")
    else:
        for finding in report.findings:
            lines.append(f"  [{finding.severity.value.upper()}] {finding.title}")
            lines.append(f"    detector={finding.detector} category={finding.category.value}")
            lines.append(f"    {finding.description}")
            if finding.affected_features:
                lines.append(f"    affected_features={', '.join(finding.affected_features)}")
            if finding.recommendation:
                lines.append(f"    recommendation={finding.recommendation}")
    return "\n".join(lines)
