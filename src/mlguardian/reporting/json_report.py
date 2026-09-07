"""JSON rendering for machine-readable reports."""

from __future__ import annotations

import json

from mlguardian.models import AuditReport


def render_json(report: AuditReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)
