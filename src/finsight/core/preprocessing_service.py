from finsight.core.data import get_stock_data
from finsight.core.features import create_lag_features
from finsight.core.utils import scale_data, split_data

def prepare_stock_data_for_prediction(ticker: str, window: int = 5, period: str = "1y", interval: str = "1d"):
    """
    Prepares stock data for prediction by fetching, creating features, scaling, and splitting the data.
    Args:
        ticker (str): Stock ticker symbol.
        window (int): Number of lag features to create.
        period (str): Period for which to fetch stock data (default is "1y").
        interval (str): Interval for the stock data (default is "1d").
    Returns:
        tuple: Contains training and testing data, scalers for features and target.
    """
    df = get_stock_data(ticker, period=period, interval=interval)
    X, y = create_lag_features(df, window)
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler
