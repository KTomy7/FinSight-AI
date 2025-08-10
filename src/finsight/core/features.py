import pandas as pd

def create_lag_features(df: pd.DataFrame, window: int = 5):
    """
    Create lag features for stock price prediction. Lag features are used to predict future values based on past values.
    This function scales the 'Close' prices to a range of 0 to 1, then creates sequences of past values (lags) for the specified window size.
    """
    df = df.copy()
    df['Close_scaled'] = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())

    X, y = [], []
    for i in range(window, len(df)):
        X.append(df['Close_scaled'].values[i - window:i])
        y.append(df['Close_scaled'].values[i])

    return X, y
