import pandas as pd

from mlguardian.schema import validate_schema


def test_validate_schema_reports_mismatch() -> None:
    train = pd.DataFrame({"target": [1], "a": [1], "b": ["x"]})
    test = pd.DataFrame({"a": [1], "b": [1.2], "c": [2]})
    findings = validate_schema(train, test, "target")
    assert findings
    assert any(f.category.value == "schema" for f in findings)
