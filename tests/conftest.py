from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def mock_stock_price_df() -> pd.DataFrame:
    """Deterministic OHLCV frame for fast unit/integration tests."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    close = [100.0, 101.5, 103.0, 102.5, 104.0, 105.5, 106.0, 107.5, 108.0, 109.5]

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [value - 0.4 for value in close],
            "High": [value + 0.8 for value in close],
            "Low": [value - 0.9 for value in close],
            "Close": close,
            "Volume": [1_000_000 + idx * 10_000 for idx in range(len(close))],
        }
    )

