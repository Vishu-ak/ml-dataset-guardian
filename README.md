# ML Dataset Guardian

An open-source ML dataset auditing toolkit for detecting data-quality issues, leakage, contamination, and distribution shift before model training.

## Why
Most model failures start with data problems, not model code.
ML Dataset Guardian gives you a quick pre-training health check for tabular datasets (CSV/Parquet) and highlights risks in plain language and JSON.

## Features
- Dataset quality checks
- Missing-value analysis
- Duplicate detection
- Constant/near-constant feature detection
- Target imbalance checks for classification
- Outlier detection (IQR / MAD z-score)
- Correlation analysis (Pearson/Spearman)
- Potential target leakage heuristics
- Train/test contamination detection
- Distribution-shift detection (PSI)
- Suspicious feature identification
- Terminal + JSON reporting
- CLI + Python API
- YAML/TOML configuration
- CI-friendly exit codes
- Extensible detector registry

## Install
```bash
pip install -e .
```

## CLI
```bash
mlguardian audit --train data/train.csv --test data/test.csv --target churn
mlguardian audit --train data/train.csv --test data/test.csv --target churn --format json --output report.json
mlguardian version
```

Supported options:
- `--format {terminal,json}`
- `--output report.json`
- `--config config.yaml|config.toml`
- `--fail-on {info,warning,high,critical}`
- `--verbose`
- `--key` (key column for contamination checks)

Exit codes:
- `0`: audit completed and findings do not exceed threshold
- `1`: audit completed and findings exceed threshold
- `2`: invalid input/configuration
- `3`: unexpected/internal error

## Python API
```python
from mlguardian import DatasetAuditor

report = DatasetAuditor(target="churn").audit(train="data/train.csv", test="data/test.csv")
print(report.risk_level)
print(report.findings)
```

## Scientific honesty and limitations
- This tool uses heuristics, so both false positives and false negatives are possible.
- Correlation by itself does not prove leakage or feature redundancy.
- Outliers can be valid real-world behavior, not always data errors.
- Drift metrics need domain context and the right time window to be meaningful.
- Contamination checks are only as good as the split strategy and keys you provide.
- Default thresholds are starting points, not universal rules.
- ML Dataset Guardian runs locally and does not transmit your dataset contents.

## Development
```bash
pip install -e .[dev]
ruff check .
mypy
pytest
python -m build
```

## Architecture
- `src/mlguardian/models.py`: typed domain models
- `src/mlguardian/config.py`: YAML/TOML config + defaults
- `src/mlguardian/loaders.py`: CSV/Parquet loaders
- `src/mlguardian/detectors/`: detector interface, registry, implementations
- `src/mlguardian/auditor.py`: orchestration + risk aggregation
- `src/mlguardian/reporting/`: terminal + JSON renderers
- `src/mlguardian/cli/main.py`: CLI entrypoint

## Extending detectors
Add a detector with `@register_detector("name")`, implement `run(train_df, test_df, context)`, and return a list of `Finding` objects.
If you open a PR for a new detector, please include fixture-based tests that show both expected hits and expected non-hits.

## Examples
- `examples/basic_audit.py`
- `examples/config.yaml`
- `examples/data/`
- `scripts/benchmark.py`
