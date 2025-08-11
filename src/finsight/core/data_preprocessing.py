import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def scale_data(
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list]
) -> Tuple[npt.NDArray, npt.NDArray, MinMaxScaler, MinMaxScaler]:
    """
    Scales the features and target variable using Min-Max scaling.

    Parameters
    ----------
    X : Union[npt.NDArray, list]
        The feature data to be scaled.
    y : Union[npt.NDArray, list]
        The target variable to be scaled.

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray, MinMaxScaler, MinMaxScaler]
        Scaled feature data, scaled target variable, scaler for features, and scaler for target.

    """
    try:
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

    except Exception as e:
        raise ValueError(f"Error converting input data to numpy arrays: {e}")

    if X.size == 0 or y.size == 0:
        raise ValueError("Input arrays cannot be empty.")
    if len(X) != len(y):
        raise ValueError("Feature and target arrays must have the same number of samples.")

    # Initialize scalers
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    try:
        # Reshape X to 2D for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])
        # Scale the reshaped X
        X_scaled_reshaped = X_scaler.fit_transform(X_reshaped)
        # Reshape back to original shape
        X_scaled = X_scaled_reshaped.reshape(X.shape)
        y_scaled = y_scaler.fit_transform(y)

    except Exception as e:
        raise ValueError(f"Scaling failed: {e}")

    return X_scaled, y_scaled, X_scaler, y_scaler

def split_data(
    X: npt.NDArray,
    y: npt.NDArray,
    test_size: float,
    shuffle: bool
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Splits the data into training and testing sets.

    Parameters
    ----------
    X : npt.NDArray
        The feature data to be split.
    y : npt.NDArray
        The target variable to be split.
    test_size : float, optional
        The proportion of the dataset to include in the test split (default is 0.2).
    shuffle : bool, optional
        Whether to shuffle the data before splitting (default is False).

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        Training features, testing features, training target, testing target.
    """
    try:
        return train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    except Exception as e:
        raise ValueError(f"Data splitting failed: {e}")
