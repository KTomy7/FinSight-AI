import yfinance as yf
import pandas as pd

def fetch_stock_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download historical stock data for a given ticker from Yahoo Finance.
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df.reset_index(inplace=True)
    return df


def fetch_stock_summary(ticker: str) -> dict:
    """
    Fetch basic stock info summary.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        summary = {
            "Name": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A")
        }

        return summary

    except Exception as e:
        raise RuntimeError(f"Failed to fetch stock summary: {e}")
