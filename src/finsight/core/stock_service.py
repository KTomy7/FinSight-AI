from finsight.core.data import fetch_stock_data, fetch_stock_summary

class StockDataService:
    """
    Service for retrieving stock data and summary information.

    Provides methods to fetch historical stock data, summary information,
    and closing prices for a given stock ticker.
    """
    def __init__(self,  ticker: str, period: str = "5y", interval: str = "1d"):
        """
        Initializes the StockDataService with a stock ticker, period, and interval.

        Parameters
        ----------
        ticker : str
            The stock ticker symbol to retrieve data for.
        period : str
            The period for which to retrieve stock data (default is "5y").
        interval : str
            The interval for the stock data (default is "1d").
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval

    def get_stock_data(self) -> pd.DataFrame:
        """
        Fetches historical stock data for the specified ticker, period, and interval.
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the historical stock data with columns for date, open, high, low, close, volume, and adjusted close.
        """
        return fetch_stock_data(self.ticker, self.period, self.interval)

    def get_summary_info(self) -> Dict[str, Any]:
        """
        Fetches summary information for the specified stock ticker.
        Returns
        -------
        Dict[str, Any]
            A dictionary containing summary information such as market cap, PE ratio, dividend yield, etc.
        """
        return fetch_stock_summary(self.ticker)

    def get_closing_prices(self) -> pd.DataFrame:
        """
        Fetches the closing prices for the specified stock ticker.
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the date and closing prices for the stock.
        """
        df = fetch_stock_data(self.ticker, self.period, self.interval)
        return df[['Date', 'Close']]
