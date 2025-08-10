import numpy.typing as npt
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from finsight.core.stock_service import fetch_stock_data
from finsight.core.features import create_lag_features
from finsight.core.data_preprocessing import scale_data, split_data

def prepare_data_for_prediction(
    ticker: str,
    window: int = 5,
    period: str = "1y",
    interval: str = "1d",
    test_size: float = 0.2,
    shuffle: bool = False
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, MinMaxScaler, MinMaxScaler]:
    """
    Prepare stock data for prediction.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol to fetch data for.
    window : int
        Number of previous time steps to use as features for the current time step.
        Must be between 1 and len(df)-1.
    period : str
        Time period for which to fetch stock data (e.g., "1y", "6mo").
    interval : str
        Time interval for the stock data (e.g., "1d", "1h").
    test_size : float
        Proportion of the dataset to include in the test split.
    shuffle : bool
        Whether to shuffle the data before splitting.

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

        df = fetch_stock_data(ticker, period=period, interval=interval)
        X, y = create_lag_features(df, window)
        X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size, shuffle)

        return X_train, X_test, y_train, y_test, X_scaler, y_scaler

    except Exception as e:
        raise ValueError(f"Error preparing data for prediction: {e}")
