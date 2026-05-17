"""Integration test for walk-forward Backtest use case."""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd

from finsight.application.dto import BacktestRequest, FetchMarketDataRequest
from finsight.application.use_cases.backtest import Backtest
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.domain.metrics import SUPPORTED_METRIC_NAMES
from finsight.domain.value_objects import DateRange, Interval, Ticker
from finsight.domain.entities import OHLCVSeries
from finsight.infrastructure.features.feature_store import PandasFeatureStore
from finsight.infrastructure.ml.sklearn import NaiveBaselineModel, LinearSklearnModel, SklearnModelRouter


class _StubFetchMarketData:
    """Stub market data provider that returns synthetic OHLCV data by ticker."""

    def __init__(self, data_by_ticker: dict[str, OHLCVSeries]) -> None:
        self._data_by_ticker = data_by_ticker
        self.calls: list[FetchMarketDataRequest] = []

    def execute(self, request: FetchMarketDataRequest) -> object:
        self.calls.append(request)
        return SimpleNamespace(history=self._data_by_ticker[request.ticker])


def _make_synthetic_ohlcv_series(ticker: str) -> OHLCVSeries:
    # Keep this deterministic and long enough for all rolling windows in feature_pipeline.
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    close = [100.0 + (idx * 0.20) for idx in range(len(dates))]
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": [1000 + idx for idx in range(len(dates))],
        }
    )

    return OHLCVSeries(
        ticker=Ticker(ticker),
        date_range=DateRange(start=dates[0].date().isoformat(), end=dates[-1].date().isoformat()),
        interval=Interval("1d"),
        df=df,
    )


def test_backtest_walk_forward_end_to_end_multiple_models() -> None:
    tickers = ("AAPL", "JPM")
    data = {ticker: _make_synthetic_ohlcv_series(ticker) for ticker in tickers}
    fetch_stub = _StubFetchMarketData(data)

    use_case = Backtest(
        fetch_market_data=cast(FetchMarketData, cast(object, fetch_stub)),
        feature_store=PandasFeatureStore(),
        model=SklearnModelRouter(adapters=[NaiveBaselineModel(), LinearSklearnModel()]),
        training_tickers=tickers,
        supported_model_ids=("naive_zero", "ridge"),
        default_interval="1d",
    )

    report = use_case.execute(
        BacktestRequest(
            model_ids=["naive_zero", "ridge"],
            years=2,
            end="2024-12-31",
            interval="1d",
            min_train_days=120,
            test_window_days=30,
            step_days=30,
            max_folds=3,
        )
    )

    assert len(fetch_stub.calls) == len(tickers)
    assert report.dataset_spec is not None
    assert report.dataset_spec.tickers == tickers
    assert report.dataset_spec.interval == "1d"

    assert report.split_spec["name"] == "walk_forward"
    assert report.split_spec["fold_count"] == 3

    assert len(report.results) == 2
    by_model = {row.model_id: row for row in report.results}
    assert set(by_model.keys()) == {"naive_zero", "ridge"}

    for model_id in ("naive_zero", "ridge"):
        result = by_model[model_id]
        assert result.metrics["fold_count"] == 3
        assert set(SUPPORTED_METRIC_NAMES).issubset(result.metrics.keys())

        assert len(result.folds) == 3
        for fold in result.folds:
            assert int(fold["fold_index"]) >= 1
            assert int(fold["n_train"]) > 0
            assert int(fold["n_test"]) > 0
            assert isinstance(fold["train_start"], str)
            assert isinstance(fold["train_end"], str)
            assert isinstance(fold["test_start"], str)
            assert isinstance(fold["test_end"], str)
            assert set(SUPPORTED_METRIC_NAMES).issubset(fold["metrics"].keys())

