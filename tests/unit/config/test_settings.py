from pathlib import Path

import pytest

from finsight.config.settings import get_settings


def test_get_settings_reads_ticker_catalog_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ticker_catalog:
  - symbol: aapl
    company_name: Apple Inc.
  - symbol: jpm
    company_name: JPMorgan Chase & Co.
""".strip(),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.ticker_catalog.symbols() == ("AAPL", "JPM")
    assert settings.ticker_catalog.entries[0].company_name == "Apple Inc."


def test_get_settings_uses_default_ticker_catalog_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("stock_data: {}", encoding="utf-8")

    settings = get_settings(config_path)

    assert settings.ticker_catalog.symbols() == ("AAPL", "JPM", "XOM", "KO", "TSLA")


def test_get_settings_rejects_explicit_empty_ticker_catalog(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ticker_catalog: []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ticker_catalog must contain at least one ticker entry"):
        get_settings(config_path)


def test_get_settings_rejects_explicit_null_ticker_catalog(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ticker_catalog: null
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ticker_catalog must contain at least one ticker entry"):
        get_settings(config_path)


def test_get_settings_rejects_duplicate_ticker_symbols_case_insensitive(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ticker_catalog:
  - symbol: aapl
    company_name: Apple Inc.
  - symbol: AAPL
    company_name: Apple Duplicate
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ticker_catalog contains duplicate symbols"):
        get_settings(config_path)


def test_get_settings_rejects_entries_without_symbol_or_company_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ticker_catalog:
  - symbol: ""
    company_name: Missing Symbol
  - symbol: AAPL
    company_name: ""
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"ticker_catalog\[0\]\.symbol must be a non-empty string"):
        get_settings(config_path)


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


def test_get_settings_rejects_explicit_null_model_catalog(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  catalog: null
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_defaults.catalog must contain at least one model entry"):
        get_settings(config_path)


def test_get_settings_rejects_model_catalog_without_training_enabled_models(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  catalog:
    - id: ridge
      label: Ridge Regression
      supports_training: false
      supports_prediction: true
  default_model_id: ridge
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must enable training for at least one model"):
        get_settings(config_path)


def test_get_settings_uses_default_model_catalog_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("stock_data: {}", encoding="utf-8")

    settings = get_settings(config_path)

    assert settings.model_defaults.model_ids() == ("naive_zero", "naive_mean", "ridge")
    assert settings.model_defaults.default_model_id == "naive_zero"


