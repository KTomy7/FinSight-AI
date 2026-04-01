from pathlib import Path

import pytest

from finsight.config import settings as settings_module
from finsight.config.settings import get_settings, reload_settings


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


def test_model_defaults_mapping_helpers_and_prediction_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  catalog:
    - id: naive_zero
      label: Naive (Zero)
      supports_training: true
      supports_prediction: true
    - id: train_only
      label: Train Only
      supports_training: true
      supports_prediction: false
""".strip(),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.model_defaults.id_to_label() == {
        "naive_zero": "Naive (Zero)",
        "train_only": "Train Only",
    }
    assert settings.model_defaults.label_to_id() == {
        "Naive (Zero)": "naive_zero",
        "Train Only": "train_only",
    }
    assert settings.model_defaults.prediction_model_ids() == ("naive_zero",)


@pytest.mark.parametrize(
    "shuffle_value, expected",
    [
        ("yes", True),
        ("on", True),
        ("false", False),
        ("off", False),
        ("not-a-bool", False),
    ],
)
def test_get_settings_parses_string_booleans_for_shuffle(tmp_path: Path, shuffle_value: str, expected: bool) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
preprocessing:
  shuffle: "{shuffle_value}"
""".strip(),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.preprocessing.shuffle is expected


def test_get_settings_clamps_numeric_ranges(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
stock_data:
  default_lookback_days: 0
preprocessing:
  lag_window: 0
  test_size: 2.5
cache:
  resource_ttl_seconds: -1
  data_ttl_seconds: 0
model_defaults:
  horizon_min: 15
  horizon_max: 10
  default_horizon: 99
""".strip(),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.stock_data.default_lookback_days == 1
    assert settings.preprocessing.lag_window == 1
    assert settings.preprocessing.test_size == 0.99
    assert settings.cache.resource_ttl_seconds == 1
    assert settings.cache.data_ttl_seconds == 1
    assert settings.model_defaults.horizon_min == 15
    assert settings.model_defaults.horizon_max == 15
    assert settings.model_defaults.default_horizon == 15


def test_get_settings_rejects_duplicate_model_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  catalog:
    - id: naive_zero
      label: Naive (Zero)
    - id: naive_zero
      label: Duplicate Id
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate model ids"):
        get_settings(config_path)


def test_get_settings_rejects_duplicate_model_labels(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  catalog:
    - id: naive_zero
      label: Shared Label
    - id: naive_mean
      label: Shared Label
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate model labels"):
        get_settings(config_path)


def test_get_settings_rejects_non_collection_ticker_catalog(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ticker_catalog: 123", encoding="utf-8")

    with pytest.raises(ValueError, match="ticker_catalog must be a list or tuple"):
        get_settings(config_path)


def test_get_settings_rejects_ticker_entry_missing_company_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ticker_catalog:
  - symbol: AAPL
    company_name: ""
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"ticker_catalog\[0\]\.company_name must be a non-empty string"):
        get_settings(config_path)


def test_get_settings_uses_defaults_when_config_is_unreadable_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{not: valid", encoding="utf-8")

    settings = get_settings(config_path)

    assert settings.stock_data.default_symbol == "AAPL"
    assert settings.model_defaults.default_model_id == "naive_zero"


def test_get_settings_ignores_catalog_entries_without_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  catalog:
    - label: Missing Id
    - id: fallback
      label: ""
""".strip(),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.model_defaults.model_ids() == ("fallback",)
    assert settings.model_defaults.labels() == ("fallback",)


def test_reload_settings_clears_cache_and_reloads(monkeypatch) -> None:
    reloaded = object()
    calls: list[str] = []

    def _fake_cache_clear() -> None:
        calls.append("cleared")

    def _fake_get_settings() -> object:
        calls.append("loaded")
        return reloaded

    _fake_get_settings.cache_clear = _fake_cache_clear
    monkeypatch.setattr(settings_module, "get_settings", _fake_get_settings)

    result = reload_settings()

    assert result is reloaded
    assert calls == ["cleared", "loaded"]


def test_settings_helper_parsers_cover_min_max_and_bool_fallbacks() -> None:
    assert settings_module._as_int("7", default=1, minimum=3) == 7
    assert settings_module._as_int("1", default=9, minimum=3) == 3
    assert settings_module._as_int("50", default=9, maximum=10) == 10

    assert settings_module._as_float("0.25", default=0.2, minimum=0.1, maximum=0.9) == 0.25
    assert settings_module._as_float("0.01", default=0.2, minimum=0.1, maximum=0.9) == 0.1
    assert settings_module._as_float("2.5", default=0.2, minimum=0.1, maximum=0.9) == 0.9

    assert settings_module._as_bool(" true ", default=False) is True
    assert settings_module._as_bool("0", default=True) is False
    assert settings_module._as_bool("unknown", default=True) is True


def test_default_config_path_points_to_repo_config() -> None:
    config_path = settings_module._default_config_path()

    assert config_path.name == "config.yaml"
    assert config_path.parent.name == "config"


