import pandas as pd
import pytest

from finsight.domain.entities import OHLCVSeries
from finsight.domain.value_objects import DateRange, Interval, Ticker
from finsight.infrastructure.features.feature_pipeline import (
    add_target,
    build_feature_dataset,
    to_panel_df,
)


def _make_ohlcv_series(ticker: str, close: list[float]) -> OHLCVSeries:
    """
    Build a minimal OHLCVSeries suitable for unit tests.

    Notes:
    - Uses daily calendar dates (freq="D") for deterministic test data.
    - Sets open/high/low equal to close to keep things simple.
    - Uses non-constant volume so volume_z20 has non-zero rolling std.
    """
    n = len(close)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": list(range(100, 100 + n)),  # non-constant volume
        }
    )

    start = dates[0].date().isoformat()
    end = dates[-1].date().isoformat()

    return OHLCVSeries(
        ticker=Ticker(ticker),
        date_range=DateRange(start, end),
        interval=Interval("1d"),
        df=df,
    )


def test_to_panel_df_normalizes_schema():
    series = _make_ohlcv_series("AAA", [100, 101, 102])
    panel = to_panel_df([series])

    assert set(["date", "ticker", "open", "high", "low", "close", "volume"]).issubset(panel.columns)
    assert panel["ticker"].unique().tolist() == ["AAA"]
    assert pd.api.types.is_datetime64_any_dtype(panel["date"])


def test_target_alignment_single_ticker():
    series = _make_ohlcv_series("AAA", [100, 110, 121])
    panel = to_panel_df([series])
    labeled = add_target(panel)

    assert pytest.approx(labeled.loc[0, "target_ret_1d"], rel=1e-12) == 0.10
    assert pytest.approx(labeled.loc[1, "target_ret_1d"], rel=1e-12) == 0.10
    assert pd.isna(labeled.loc[2, "target_ret_1d"])


def test_no_cross_ticker_leakage():
    """
    Ensure computations are isolated per ticker (groupby('ticker')).

    We need enough rows to survive warmup windows:
    - sma_50 requires 50 rows
    - target shift(-1) drops last row
    So use n >= 51; we choose 70 for safety.
    """
    n = 70

    a_close = [100.0] * n

    b_close = [100.0]
    for _ in range(n - 1):
        b_close.append(b_close[-1] * 1.1)

    a = _make_ohlcv_series("AAA", a_close)
    b = _make_ohlcv_series("BBB", b_close)

    features = build_feature_dataset([a, b])

    aaa = features[features["ticker"] == "AAA"]
    bbb = features[features["ticker"] == "BBB"]

    assert len(aaa) > 0
    assert len(bbb) > 0

    last_aaa = aaa.iloc[-1]
    last_bbb = bbb.iloc[-1]

    assert abs(last_aaa["ret_1d"]) < 1e-12
    assert pytest.approx(last_bbb["ret_1d"], rel=1e-6) == 0.10

    # Constant close => close_sma50_ratio should be ~0 after warmup.
    assert abs(last_aaa["close_sma50_ratio"]) < 1e-12
