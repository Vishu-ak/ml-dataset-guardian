from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mlguardian.cli.main import app

FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_audit_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlguardian",
            "audit",
            "--train",
            str(FIXTURES / "train.csv"),
            "--test",
            str(FIXTURES / "test.csv"),
            "--target",
            "churn",
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        app()
    assert exc.value.code in {0, 1}
    assert output.exists()
    assert "schema_version" in output.read_text()


def test_cli_version(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["mlguardian", "version"])
    with pytest.raises(SystemExit) as exc:
        app()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip()
