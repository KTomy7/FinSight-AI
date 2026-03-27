from pathlib import Path

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

