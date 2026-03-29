from pathlib import Path

import pytest

from finsight.config.settings import get_settings


def test_get_settings_reads_training_tickers_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
training:
  training_tickers:
    - aapl
    - jpm
""".strip(),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.training.training_tickers == ("aapl", "jpm")


def test_get_settings_uses_default_training_tickers_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("stock_data: {}", encoding="utf-8")

    settings = get_settings(config_path)

    assert settings.training.training_tickers == ("AAPL", "JPM", "XOM", "KO", "TSLA")


def test_get_settings_parses_model_catalog_with_default_model_id(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  catalog:
    - id: naive_zero
      label: Naive (Zero)
      supports_training: true
      supports_prediction: true
    - id: ridge
      label: Ridge Regression
      supports_training: false
      supports_prediction: true
  default_model_id: naive_zero
""".strip(),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.model_defaults.model_ids() == ("naive_zero", "ridge")
    assert settings.model_defaults.labels() == ("Naive (Zero)", "Ridge Regression")
    assert settings.model_defaults.default_model_id == "naive_zero"
    assert settings.model_defaults.training_model_ids() == ("naive_zero",)


def test_get_settings_rejects_unknown_default_model_id(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  catalog:
    - id: naive_zero
      label: Naive (Zero)
  default_model_id: does_not_exist
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown model_defaults.default_model_id"):
        get_settings(config_path)


def test_get_settings_uses_default_model_catalog_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("stock_data: {}", encoding="utf-8")

    settings = get_settings(config_path)

    assert settings.model_defaults.model_ids() == ("naive_zero", "naive_mean", "ridge")
    assert settings.model_defaults.default_model_id == "naive_zero"


