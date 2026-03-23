import pandas as pd
import pytest

from finsight.domain.entities import OHLCVSeries
from finsight.domain.value_objects import DateRange, Interval, Ticker
from finsight.infrastructure.features.feature_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    add_features,
    add_target,
    build_feature_dataset,
    finalize_feature_frame,
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


def test_to_panel_df_empty_returns_expected_schema():
    panel = to_panel_df([])

    assert panel.empty
    assert list(panel.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]


def test_to_panel_df_accepts_date_index_and_normalizes():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [100, 101, 102],
            "Low": [100, 101, 102],
            "Close": [100, 101, 102],
            "Volume": [100, 101, 102],
        },
        index=dates,
    )
    df.index.name = "Date"

    series = OHLCVSeries(
        ticker=Ticker("AAA"),
        date_range=DateRange(dates[0].date().isoformat(), dates[-1].date().isoformat()),
        interval=Interval("1d"),
        df=df,
    )

    panel = to_panel_df([series])

    assert set(["date", "ticker", "open", "high", "low", "close", "volume"]).issubset(panel.columns)
    assert panel["ticker"].unique().tolist() == ["AAA"]
    assert pd.api.types.is_datetime64_any_dtype(panel["date"])


def test_to_panel_df_deduplicates_ticker_date_keep_last():
    s1 = _make_ohlcv_series("AAA", [100, 101, 102])
    s2 = _make_ohlcv_series("AAA", [200, 201, 202])

    panel = to_panel_df([s1, s2])

    assert len(panel) == 3
    assert panel["close"].tolist() == [200, 201, 202]


def test_to_panel_df_raises_on_missing_required_columns():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    bad_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [1, 1, 1],
            "High": [1, 1, 1],
            "Low": [1, 1, 1],
            "Volume": [10, 11, 12],
        }
    )

    series = OHLCVSeries(
        ticker=Ticker("AAA"),
        date_range=DateRange(dates[0].date().isoformat(), dates[-1].date().isoformat()),
        interval=Interval("1d"),
        df=bad_df,
    )

    with pytest.raises(ValueError, match="missing columns"):
        to_panel_df([series])


def test_to_panel_df_raises_on_invalid_date_values():
    dates = ["2020-01-01", "not-a-date", "2020-01-03"]
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100, 101, 102],
            "High": [100, 101, 102],
            "Low": [100, 101, 102],
            "Close": [100, 101, 102],
            "Volume": [100, 101, 102],
        }
    )
    series = OHLCVSeries(
        ticker=Ticker("AAA"),
        date_range=DateRange("2020-01-01", "2020-01-03"),
        interval=Interval("1d"),
        df=df,
    )

    with pytest.raises(ValueError, match="invalid date values"):
        to_panel_df([series])


def test_add_features_computes_expected_returns_and_sma():
    series = _make_ohlcv_series("AAA", [100, 110, 121, 133.1, 146.41])
    panel = to_panel_df([series])

    feat = add_features(panel)

    assert pytest.approx(feat.loc[1, "ret_1d"], rel=1e-12) == 0.10
    assert pytest.approx(feat.loc[2, "ret_1d"], rel=1e-12) == 0.10
    assert pd.isna(feat.loc[3, "sma_5"])
    assert not pd.isna(feat.loc[4, "sma_5"])


def test_add_features_volume_z20_nan_when_std_zero():
    n = 25
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * n,
            "open": [100.0 + i for i in range(n)],
            "high": [100.0 + i for i in range(n)],
            "low": [100.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
        }
    )

    feat = add_features(df)

    assert pd.isna(feat.loc[19, "volume_z20"])
    assert pd.isna(feat.loc[24, "volume_z20"])


def test_add_target_last_row_per_ticker_is_nan():
    a = _make_ohlcv_series("AAA", [100, 110, 121])
    b = _make_ohlcv_series("BBB", [200, 180, 198])
    panel = to_panel_df([a, b])

    labeled = add_target(panel)
    last_rows = labeled.groupby("ticker", sort=False).tail(1)

    assert last_rows["target_ret_1d"].isna().all()


def test_finalize_feature_frame_raises_on_missing_columns():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "ticker": ["AAA"]})

    with pytest.raises(ValueError, match="missing expected columns"):
        finalize_feature_frame(df)


def test_build_feature_dataset_empty_input_returns_empty_frame():
    out = build_feature_dataset([])

    assert out.empty
    assert list(out.columns) == ["date", "ticker", *FEATURE_COLUMNS, *TARGET_COLUMNS]


