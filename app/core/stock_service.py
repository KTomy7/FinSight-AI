from core.data import fetch_stock_history, fetch_stock_summary
import pandas as pd

class StockDataService:
    def __init__(self,  ticker: str, period: str = "5y", interval: str = "1d"):
        self.ticker = ticker
        self.period = period
        self.interval = interval

    def get_stock_history_data(self):
        """
        Retrieves stock data for a given stock symbol.
        """
        return fetch_stock_history(self.ticker, self.period, self.interval)

    def get_summary_info(self):
        """
        Updates the stock data for a given stock symbol.
        """
        return fetch_stock_summary(self.ticker)

    def get_closing_prices(self):
        """
        Get only the date and closing price from the historical data.
        """
        df = self.get_stock_data()
        return df[['Date', 'Close']]