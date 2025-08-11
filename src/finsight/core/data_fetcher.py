import yfinance as yf
import pandas as pd
from typing import Any, Dict

def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL" for Apple Inc.).
    period : str
        The period for which to fetch data (e.g., "1y" for one year)
    interval : str
        The interval for the data (e.g., "1d" for daily data).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the historical stock data with columns for Date, Open, High, Low, Close, Volume.
    """
    ticker = ticker.upper().strip()

    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

        if df.empty:
            raise Exception(f"No data found for ticker {ticker}")

        # Ensure consistent DataFrame structure
        df = df.dropna().reset_index()

        # Validate required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the data: {', '.join(missing_columns)}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch stock data for {ticker}: {e}")

def fetch_stock_summary(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive stock summary information.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL" for Apple Inc.).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing key stock information such as name, sector, industry, market cap, etc.
    """
    ticker = ticker.upper().strip()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            raise ValueError(f"No information found for ticker {ticker}")

        summary = {
            "ticker": ticker,
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
            "dividend_yield": info.get("dividendYield", "N/A")
        }

        return summary

    except Exception as e:
        raise RuntimeError(f"Failed to fetch stock summary for {ticker}: {e}")
