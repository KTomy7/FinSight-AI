import pandas as pd
import pytest

from finsight.domain.entities import OHLCVSeries
from finsight.domain.value_objects import DateRange, Interval, Ticker
from finsight.infrastructure.features.feature_store import PandasFeatureStore


def _make_series(ticker: str, *, periods: int = 80) -> OHLCVSeries:
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    close = [100.0 + idx for idx in range(periods)]
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": [1000 + idx for idx in range(periods)],
        }
    )
    return OHLCVSeries(
        ticker=Ticker(ticker),
        date_range=DateRange(dates[0].date().isoformat(), dates[-1].date().isoformat()),
        interval=Interval("1d"),
        df=frame,
    )


def test_build_inference_feature_dataset_returns_feature_only_frame() -> None:
    store = PandasFeatureStore()

    out = store.build_inference_feature_dataset([_make_series("AAA")])

    assert not out.empty
    assert "target_ret_1d" not in out.columns
    assert {"date", "ticker"}.issubset(set(out.columns))


def test_frame_date_range_rejects_invalid_dates() -> None:
    store = PandasFeatureStore()
    frame = pd.DataFrame({"date": [None, None]})

    with pytest.raises(ValueError, match="does not contain valid dates"):
        store.frame_date_range(frame)


def test_row_count_rejects_non_dataframe_input() -> None:
    store = PandasFeatureStore()

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        store.row_count(dataset=[{"date": "2024-01-01"}])
