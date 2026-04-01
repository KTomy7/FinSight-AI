from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

import finsight.application.use_cases.fetch_market_data as fetch_market_data_module
from finsight.application.dto import FetchMarketDataRequest
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.ports import MarketDataPort
from finsight.domain.value_objects import DateRange, Interval, Ticker


class _StubMarketData:
    def __init__(self) -> None:
        self.fetch_calls: list[SimpleNamespace] = []
        self.summary_calls: list[str] = []

    def fetch_ohlcv(self, ticker: Ticker, date_range: DateRange, interval: Interval) -> OHLCVSeries:
        self.fetch_calls.append(SimpleNamespace(ticker=ticker, date_range=date_range, interval=interval))
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-03-15", "2026-03-16", "2026-03-17"]),
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
            }
        )
        return OHLCVSeries(ticker=ticker, date_range=date_range, interval=interval, df=frame)

    def get_summary(self, ticker: Ticker) -> StockSummary:
        self.summary_calls.append(str(ticker))
        return StockSummary(ticker=ticker, data={"market_cap": 123_000_000})


class _FixedDate(date):
    @classmethod
    def today(cls) -> date:
        return cls(2026, 3, 17)


def test_execute_applies_default_lookback_and_interval(monkeypatch) -> None:
    monkeypatch.setattr(fetch_market_data_module.datetime, "date", _FixedDate)

    stub = _StubMarketData()
    use_case = FetchMarketData(cast(MarketDataPort, cast(object, stub)), default_lookback_days=3, default_interval="1wk")

    result = use_case.execute(FetchMarketDataRequest(ticker=" aapl "))

    assert len(stub.fetch_calls) == 1
    fetch_call = stub.fetch_calls[0]
    assert str(fetch_call.ticker) == "AAPL"
    assert str(fetch_call.interval) == "1wk"
    assert fetch_call.date_range.start.isoformat() == "2026-03-15"
    assert fetch_call.date_range.end.isoformat() == "2026-03-17"
    assert stub.summary_calls == ["AAPL"]
    assert result.summary_dict == {"market_cap": 123_000_000}


def test_execute_with_explicit_end_date_resolves_start_from_parsed_end() -> None:
    stub = _StubMarketData()
    use_case = FetchMarketData(cast(MarketDataPort, cast(object, stub)), default_lookback_days=5, default_interval="1d")

    use_case.execute(FetchMarketDataRequest(ticker="MSFT", end_date="2026-03-20", include_summary=False))

    assert len(stub.fetch_calls) == 1
    fetch_call = stub.fetch_calls[0]
    assert fetch_call.date_range.start.isoformat() == "2026-03-16"
    assert fetch_call.date_range.end.isoformat() == "2026-03-20"
    assert stub.summary_calls == []


def test_execute_with_explicit_start_date_uses_that_start_without_lookback() -> None:
    stub = _StubMarketData()
    use_case = FetchMarketData(cast(MarketDataPort, cast(object, stub)), default_lookback_days=365)

    result = use_case.execute(
        FetchMarketDataRequest(
            ticker="JPM",
            start_date="2026-01-01",
            end_date="2026-01-15",
            interval="1d",
            include_summary=False,
        )
    )

    assert len(stub.fetch_calls) == 1
    fetch_call = stub.fetch_calls[0]
    assert fetch_call.date_range.start.isoformat() == "2026-01-01"
    assert fetch_call.date_range.end.isoformat() == "2026-01-15"
    assert result.summary is None
    assert result.summary_dict is None


def test_execute_invalid_end_date_raises_and_skips_market_calls() -> None:
    stub = _StubMarketData()
    use_case = FetchMarketData(cast(MarketDataPort, cast(object, stub)))

    with pytest.raises(ValueError, match="DateRange.end is not a valid ISO date"):
        use_case.execute(FetchMarketDataRequest(ticker="AAPL", end_date="2026-99-01"))

    assert stub.fetch_calls == []
    assert stub.summary_calls == []

