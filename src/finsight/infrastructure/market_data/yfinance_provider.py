from __future__ import annotations

from typing import Any, Dict, Mapping

import yfinance as yf

from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.ports import MarketDataPort
from finsight.domain.value_objects import Interval, Period, Ticker

class YFinanceMarketDataProvider(MarketDataPort):
    def get_price_history(self, ticker: Ticker, period: Period, interval: Interval) -> OHLCVSeries:
        try:
            df = yf.download(
                str(ticker),
                period=str(period),
                interval=str(interval),
                auto_adjust=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch stock data for {ticker}: {e}") from e

        if df is None or df.empty:
            raise RuntimeError(f"No data found for ticker {ticker}")

        # Ensure consistent DataFrame structure
        df = df.dropna().reset_index()

        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in the data: {', '.join(missing)}")

        return OHLCVSeries(ticker=ticker, period=period, interval=interval, df=df)

    def get_summary(self, ticker: Ticker) -> StockSummary:
        try:
            stock = yf.Ticker(str(ticker))
            info: Mapping[str, Any] = stock.info or {}
        except Exception as e:
            raise RuntimeError(f"Failed to fetch stock summary for {ticker}: {e}") from e

        if not info:
            raise RuntimeError(f"No information found for ticker {ticker}")

        summary: Dict[str, Any] = {
            "ticker": str(ticker),
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "current_price": info.get("currentPrice", "N/A"),
            "previous_close": info.get("regularMarketPreviousClose", "N/A"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "volume": info.get("volume", "N/A"),
            "avg_volume": info.get("averageVolume", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
        }

        return StockSummary(ticker=ticker, data=summary)

