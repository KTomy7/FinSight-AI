"""Unit tests for FetchMarketData use case."""
from __future__ import annotations

import datetime
from typing import Any, Mapping
from unittest.mock import MagicMock

import pandas as pd
import pytest

from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.value_objects import DateRange, Interval, Ticker
from finsight.application.use_cases.fetch_market_data import (
    FetchMarketData,
    FetchMarketDataRequest,
    FetchMarketDataResult,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv_series(ticker: str = "AAPL") -> OHLCVSeries:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1_000_000] * 5,
        }
    )
    return OHLCVSeries(
        ticker=Ticker(ticker),
        date_range=DateRange(start="2024-01-01", end="2024-01-05"),
        interval=Interval("1d"),
        df=df,
    )


def _make_summary(ticker: str = "AAPL") -> StockSummary:
    return StockSummary(ticker=Ticker(ticker), data={"marketCap": 3_000_000_000_000})


def _make_mock_port(ticker: str = "AAPL") -> MagicMock:
    port = MagicMock()
    port.fetch_ohlcv.return_value = _make_ohlcv_series(ticker)
    port.get_summary.return_value = _make_summary(ticker)
    return port


# Fixed "today" to avoid date-sensitive flakiness
_FIXED_TODAY = datetime.date(2024, 6, 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchMarketDataExecute:
    def test_explicit_start_and_end_passed_to_port(self) -> None:
        port = _make_mock_port()
        uc = FetchMarketData(port)

        uc.execute(
            FetchMarketDataRequest(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-05",
                include_summary=False,
            )
        )

        port.fetch_ohlcv.assert_called_once()
        call_kwargs = port.fetch_ohlcv.call_args.kwargs
        assert call_kwargs["date_range"].start == datetime.date(2024, 1, 1)
        assert call_kwargs["date_range"].end == datetime.date(2024, 1, 5)

    def test_no_start_date_computes_default_lookback_from_explicit_end(self) -> None:
        """When start_date is omitted but end_date is explicit, start = end - (lookback - 1) days."""
        port = _make_mock_port()
        uc = FetchMarketData(port, default_lookback_days=365)

        uc.execute(
            FetchMarketDataRequest(
                ticker="AAPL",
                end_date="2024-06-01",
                include_summary=False,
            )
        )

        call_kwargs = port.fetch_ohlcv.call_args.kwargs
        end = datetime.date(2024, 6, 1)
        expected_start = end - datetime.timedelta(days=364)
        assert call_kwargs["date_range"].end == end
        assert call_kwargs["date_range"].start == expected_start

    def test_include_summary_true_calls_get_summary(self) -> None:
        port = _make_mock_port()
        uc = FetchMarketData(port)

        result = uc.execute(
            FetchMarketDataRequest(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-05",
                include_summary=True,
            )
        )

        port.get_summary.assert_called_once()
        assert result.summary is not None

    def test_include_summary_false_skips_get_summary(self) -> None:
        port = _make_mock_port()
        uc = FetchMarketData(port)

        result = uc.execute(
            FetchMarketDataRequest(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-05",
                include_summary=False,
            )
        )

        port.get_summary.assert_not_called()
        assert result.summary is None

    def test_result_contains_history(self) -> None:
        port = _make_mock_port()
        uc = FetchMarketData(port)

        result = uc.execute(
            FetchMarketDataRequest(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-05",
                include_summary=False,
            )
        )

        assert isinstance(result, FetchMarketDataResult)
        assert isinstance(result.history, OHLCVSeries)

    def test_ticker_normalisation_passed_to_port(self) -> None:
        port = _make_mock_port()
        uc = FetchMarketData(port)

        uc.execute(
            FetchMarketDataRequest(
                ticker=" aapl ",
                start_date="2024-01-01",
                end_date="2024-01-05",
                include_summary=False,
            )
        )

        call_kwargs = port.fetch_ohlcv.call_args.kwargs
        assert call_kwargs["ticker"].value == "AAPL"

    def test_summary_dict_property_returns_data(self) -> None:
        port = _make_mock_port()
        uc = FetchMarketData(port)

        result = uc.execute(
            FetchMarketDataRequest(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-05",
                include_summary=True,
            )
        )

        assert result.summary_dict == {"marketCap": 3_000_000_000_000}

    def test_summary_dict_is_none_when_no_summary(self) -> None:
        port = _make_mock_port()
        uc = FetchMarketData(port)

        result = uc.execute(
            FetchMarketDataRequest(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-05",
                include_summary=False,
            )
        )

        assert result.summary_dict is None
