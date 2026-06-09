"""
Unit tests for MarketDataPresenter.
"""
import pandas as pd
import pytest

from finsight.adapters.web_streamlit.presenters import MarketDataPresenter
from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.value_objects import DateRange, Interval, Ticker
from datetime import date


class TestMarketDataPresenterFormatSummaryMetrics:
    """Tests for format_summary_metrics method."""

    def test_format_summary_metrics_with_valid_summary(self):
        """Test formatting with complete summary data."""
        summary_data = {
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Computer Hardware",
            "current_price": 150.25,
            "previous_close": 149.80,
            "market_cap": 2_400_000_000_000,
            "pe_ratio": 25.5,
            "fifty_two_week_high": 199.9,
            "fifty_two_week_low": 124.17,
            "volume": 52_000_000,
            "avg_volume": 50_000_000,
            "dividend_yield": 0.0045,
        }
        summary = StockSummary(ticker=Ticker("AAPL"), data=summary_data)

        result = MarketDataPresenter.format_summary_metrics(summary)

        assert result["ticker"] == "AAPL"
        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["industry"] == "Computer Hardware"
        assert result["current_price"] == 150.25
        assert result["pe_ratio"] == 25.5
        assert isinstance(result["market_cap"], int)

    def test_format_summary_metrics_with_na_values(self):
        """Test formatting with N/A values in summary."""
        summary_data = {
            "name": "Unknown Corp",
            "sector": "N/A",
            "industry": "N/A",
            "current_price": "N/A",
            "previous_close": 100.0,
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "fifty_two_week_high": 110.0,
            "fifty_two_week_low": 90.0,
            "volume": "N/A",
            "avg_volume": "N/A",
            "dividend_yield": None,
        }
        summary = StockSummary(ticker=Ticker("TEST"), data=summary_data)

        result = MarketDataPresenter.format_summary_metrics(summary)

        assert result["current_price"] == "N/A"
        assert result["pe_ratio"] == "N/A"
        assert result["previous_close"] == 100.0  # Valid float values are converted
        assert result["dividend_yield"] == "N/A"  # None values become "N/A"

    def test_format_summary_metrics_with_none_summary(self):
        """Test formatting with None summary."""
        result = MarketDataPresenter.format_summary_metrics(None)

        assert result == {}

    def test_format_summary_metrics_with_empty_data(self):
        """Test formatting with empty data dict."""
        summary = StockSummary(ticker=Ticker("AAPL"), data={})

        result = MarketDataPresenter.format_summary_metrics(summary)

        assert result["ticker"] == "AAPL"
        assert result["name"] == "N/A"
        assert result["sector"] == "N/A"
        assert result["current_price"] == "N/A"
        assert result["market_cap"] == "N/A"


class TestMarketDataPresenterFormatChartData:
    """Tests for format_chart_data method."""

    def test_format_chart_data_with_valid_ohlcv(self):
        """Test formatting with valid OHLCV data."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5),
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [101.5, 102.5, 103.5, 104.5, 105.5],
            "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        })

        ohlcv = OHLCVSeries(
            ticker=Ticker("AAPL"),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 5)),
            interval=Interval("1d"),
            df=df,
        )

        result = MarketDataPresenter.format_chart_data(ohlcv)

        assert not result.empty
        assert "Close" in result.columns
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert len(result) == 5
        assert result.index.name == "Date"

    def test_format_chart_data_with_lowercase_columns(self):
        """Test formatting with lowercase column names."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.5, 102.5, 103.5],
        })

        ohlcv = OHLCVSeries(
            ticker=Ticker("AAPL"),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 3)),
            interval=Interval("1d"),
            df=df,
        )

        result = MarketDataPresenter.format_chart_data(ohlcv)

        assert not result.empty
        assert "close" in result.columns

    def test_format_chart_data_with_none_ohlcv(self):
        """Test formatting with None OHLCV."""
        result = MarketDataPresenter.format_chart_data(None)

        assert result.empty

    def test_format_chart_data_with_empty_dataframe(self):
        """Test formatting with empty DataFrame."""
        df = pd.DataFrame({"Date": [], "Open": [], "High": [], "Low": [], "Close": [], "Volume": []})
        ohlcv = OHLCVSeries(
            ticker=Ticker("AAPL"),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 5)),
            interval=Interval("1d"),
            df=df,
        )

        result = MarketDataPresenter.format_chart_data(ohlcv)

        assert result.empty

    def test_format_chart_data_with_missing_date_column(self):
        """Test formatting with missing date column."""
        df = pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.5, 102.5],
        })

        ohlcv = OHLCVSeries(
            ticker=Ticker("AAPL"),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 2)),
            interval=Interval("1d"),
            df=df,
        )

        result = MarketDataPresenter.format_chart_data(ohlcv)

        assert result.empty

    def test_format_chart_data_with_missing_ohlc_columns(self):
        """Test formatting with missing OHLC columns."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=2),
            "Volume": [1000000, 1100000],
        })

        ohlcv = OHLCVSeries(
            ticker=Ticker("AAPL"),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 2)),
            interval=Interval("1d"),
            df=df,
        )

        result = MarketDataPresenter.format_chart_data(ohlcv)

        assert result.empty

