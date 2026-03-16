from __future__ import annotations

from typing import Any, Dict, Mapping

import pandas as pd
import yfinance as yf

from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.ports import MarketDataPort
from finsight.domain.value_objects import DateRange, Interval, Ticker

class YFinanceMarketDataProvider(MarketDataPort):
    def fetch_ohlcv(
        self,
        ticker: Ticker,
        date_range: DateRange,
        interval: Interval,
    ) -> OHLCVSeries:
        """Download OHLCV bars for *ticker* over *date_range* (inclusive).

        yfinance treats its ``end`` argument as *exclusive*, so we pass
        ``date_range.end_exclusive`` (end + 1 day) to ensure the requested
        end date is included in the result.
        """
        try:
            df = yf.download(
                str(ticker),
                start=date_range.start.isoformat(),
                end=date_range.end_exclusive.isoformat(),
                interval=str(interval),
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch stock data for {ticker}: {e}") from e

        if df is None or df.empty:
            raise RuntimeError(
                f"No data found for {ticker} in range {date_range}"
            )

        # Flatten MultiIndex columns produced by some yfinance versions
        # (e.g. ("Open", "AAPL") → "Open")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Ensure consistent structure
        df = df.dropna().reset_index()

        # Normalise intraday time-index column name (Datetime → Date)
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise RuntimeError(
                f"Missing required columns in the data: {', '.join(missing)}"
            )

        return OHLCVSeries(
            ticker=ticker,
            date_range=date_range,
            interval=interval,
            df=df,
        )

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

