from __future__ import annotations

import pandas as pd
import pytest

from finsight.domain.value_objects import DateRange, Interval, Ticker
from finsight.infrastructure.market_data.yfinance_provider import YFinanceMarketDataProvider
import finsight.infrastructure.market_data.yfinance_provider as yfinance_provider_module


def test_fetch_ohlcv_flattens_multiindex_normalizes_datetime_and_passes_end_exclusive(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_download(symbol: str, **kwargs):
        captured["symbol"] = symbol
        captured.update(kwargs)

        index = pd.to_datetime(["2026-03-01 09:30:00", "2026-03-02 09:30:00"])
        index.name = "Datetime"
        data = pd.DataFrame(
            {
                ("Open", symbol): [100.0, 101.0],
                ("High", symbol): [101.0, 102.0],
                ("Low", symbol): [99.0, 100.0],
                ("Close", symbol): [100.5, 101.5],
                ("Volume", symbol): [1000, 1100],
            },
            index=index,
        )
        data.iloc[1, data.columns.get_loc(("Volume", symbol))] = pd.NA
        return data

    monkeypatch.setattr(yfinance_provider_module.yf, "download", _fake_download)

    provider = YFinanceMarketDataProvider()
    date_range = DateRange("2026-03-01", "2026-03-02")
    series = provider.fetch_ohlcv(Ticker("aapl"), date_range, Interval("1d"))

    assert captured["symbol"] == "AAPL"
    assert captured["start"] == "2026-03-01"
    assert captured["end"] == "2026-03-03"
    assert captured["interval"] == "1d"
    assert captured["auto_adjust"] is True
    assert captured["progress"] is False

    assert list(series.df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(series.df) == 1
    assert series.df["Date"].iloc[0].isoformat().startswith("2026-03-01")


def test_fetch_ohlcv_raises_runtime_error_when_download_fails(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise Exception("network down")

    monkeypatch.setattr(yfinance_provider_module.yf, "download", _boom)

    provider = YFinanceMarketDataProvider()
    with pytest.raises(RuntimeError, match="Failed to fetch stock data for AAPL: network down"):
        provider.fetch_ohlcv(Ticker("AAPL"), DateRange("2026-01-01", "2026-01-02"), Interval("1d"))


def test_fetch_ohlcv_raises_when_download_returns_empty_frame(monkeypatch) -> None:
    monkeypatch.setattr(yfinance_provider_module.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    provider = YFinanceMarketDataProvider()
    with pytest.raises(RuntimeError, match="No data found for AAPL"):
        provider.fetch_ohlcv(Ticker("AAPL"), DateRange("2026-01-01", "2026-01-02"), Interval("1d"))


def test_fetch_ohlcv_raises_when_required_columns_are_missing(monkeypatch) -> None:
    def _missing_columns(*args, **kwargs):
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-01"]),
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
            }
        )

    monkeypatch.setattr(yfinance_provider_module.yf, "download", _missing_columns)

    provider = YFinanceMarketDataProvider()
    with pytest.raises(RuntimeError, match="Missing required columns in the data: Volume"):
        provider.fetch_ohlcv(Ticker("AAPL"), DateRange("2026-01-01", "2026-01-02"), Interval("1d"))


def test_get_summary_returns_expected_mapping_with_defaults(monkeypatch) -> None:
    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.info = {
                "longName": "Apple Inc.",
                "sector": "Technology",
                "marketCap": 100,
                "currentPrice": 200.5,
            }

    monkeypatch.setattr(yfinance_provider_module.yf, "Ticker", _FakeTicker)

    provider = YFinanceMarketDataProvider()
    summary = provider.get_summary(Ticker("aapl"))

    assert summary.ticker.value == "AAPL"
    assert summary.data["ticker"] == "AAPL"
    assert summary.data["name"] == "Apple Inc."
    assert summary.data["sector"] == "Technology"
    assert summary.data["market_cap"] == 100
    assert summary.data["current_price"] == 200.5
    assert summary.data["industry"] == "N/A"
    assert summary.data["pe_ratio"] == "N/A"


def test_get_summary_raises_runtime_error_when_ticker_lookup_fails(monkeypatch) -> None:
    def _boom(symbol: str):
        raise Exception("bad gateway")

    monkeypatch.setattr(yfinance_provider_module.yf, "Ticker", _boom)

    provider = YFinanceMarketDataProvider()
    with pytest.raises(RuntimeError, match="Failed to fetch stock summary for AAPL: bad gateway"):
        provider.get_summary(Ticker("AAPL"))


def test_get_summary_raises_when_info_is_empty(monkeypatch) -> None:
    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.info = {}

    monkeypatch.setattr(yfinance_provider_module.yf, "Ticker", _FakeTicker)

    provider = YFinanceMarketDataProvider()
    with pytest.raises(RuntimeError, match="No information found for ticker AAPL"):
        provider.get_summary(Ticker("AAPL"))

