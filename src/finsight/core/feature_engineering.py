import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple

def create_lag_features(df: pd.DataFrame, window: int) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Creates lag features for stock price prediction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stock data with a 'Close' column.
    window : int
        Number of previous time steps to use as features for the current time step.
        Must be between 1 and len(df)-1.

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray] containing:
    X : np.ndarray
        Array of lagged features with shape (n_samples, window).
    y : np.ndarray
        Array of target values (current time step).

    """
    df = df.copy()
    close_values = df['Close'].values

    close_min = close_values.min()
    close_max = close_values.max()

    if close_min == close_max:
        df['Close_scaled'] = np.zeros_like(close_values)
    else:
        df['Close_scaled'] = (close_values - close_min) / (close_max - close_min)

    X, y = [], []
    scaled_values = df['Close_scaled'].values

    for i in range(window, len(df)):
        X.append(scaled_values[i - window:i])
        y.append(scaled_values[i])

    return np.array(X), np.array(y)
