import numpy.typing as npt
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from finsight.core.stock_service import fetch_stock_data
from finsight.core.features import create_lag_features
from finsight.core.data_preprocessing import scale_data, split_data
from finsight.helpers.helper import get_preprocessing_settings

def prepare_data_for_prediction(
    ticker: str,
    period: str = None,
    interval: str = None,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, MinMaxScaler, MinMaxScaler]:
    """
    Prepare stock data for prediction.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol to fetch data for.
    period : str
        Time period for which to fetch stock data (e.g., "1y", "6mo").
    interval : str
        Time interval for the stock data (e.g., "1d", "1h").

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, MinMaxScaler, MinMaxScaler]
    Contains:
        - X_train: Training feature data.
        - X_test: Testing feature data.
        - y_train: Training target variable.
        - y_test: Testing target variable.
        - X_scaler: Scaler for feature data.
        - y_scaler: Scaler for target variable.
    """
    try:
        # Load default settings from config
        preprocessing_settings = get_preprocessing_settings()
        window = preprocessing_settings.get("lag_window")
        test_size = preprocessing_settings.get("test_size")
        shuffle = preprocessing_settings.get("shuffle")

        df = fetch_stock_data(ticker, period=period, interval=interval)
        X, y = create_lag_features(df, window)
        X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size, shuffle)

        return X_train, X_test, y_train, y_test, X_scaler, y_scaler

    except Exception as e:
        raise ValueError(f"Error preparing data for prediction: {e}")
