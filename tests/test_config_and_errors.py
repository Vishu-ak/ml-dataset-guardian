from pathlib import Path

import pytest

from mlguardian.config import load_config
from mlguardian.exceptions import ConfigurationError, UnsupportedFormatError
from mlguardian.loaders import load_dataset


def test_yaml_config_load(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("missing:\n  warning_threshold: 0.3\nfail_on: high\n")
    loaded = load_config(str(cfg))
    assert loaded.missing.warning_threshold == 0.3
    assert loaded.fail_on == "high"


def test_invalid_config_suffix(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    with pytest.raises(ConfigurationError):
        load_config(str(cfg))


def test_unsupported_dataset_type(tmp_path: Path) -> None:
    data = tmp_path / "data.xlsx"
    data.write_text("x")
    with pytest.raises(UnsupportedFormatError):
        load_dataset(str(data))
