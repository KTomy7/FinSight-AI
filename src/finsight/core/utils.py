import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def scale_data(X, y):
    """
    Scales the features and target variable using MinMaxScaler.
    X should be a 2D array-like structure and y should be a 1D array-like structure.
    """
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = y_scaler.fit_transform(y)

    return X_scaled, y_scaled, X_scaler, y_scaler

def split_data(X, y, test_size: float = 0.2):
    """
    Splits the data into training and testing sets. Uses a fixed split without shuffling.
    """
    return train_test_split(X, y, test_size=test_size, shuffle=False)
