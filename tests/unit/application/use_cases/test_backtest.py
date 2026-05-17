from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

import finsight.application.dto as application_dto
from finsight.application.use_cases.backtest import Backtest
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.domain.entities import ModelEvaluationResult, OHLCVSeries
from finsight.domain.ports import FeatureStorePort, ModelPort
from finsight.domain.value_objects import DateRange, Interval, Ticker


class _StubFetchMarketData:
    def __init__(self, series_by_ticker: dict[str, OHLCVSeries]) -> None:
        self._series_by_ticker = series_by_ticker
        self.calls: list[application_dto.FetchMarketDataRequest] = []

    def execute(self, request: application_dto.FetchMarketDataRequest) -> SimpleNamespace:
        self.calls.append(request)
        return SimpleNamespace(history=self._series_by_ticker[request.ticker])


class _StubFeatureStore:
    def __init__(self, feature_df: pd.DataFrame) -> None:
        self._feature_df = feature_df
        self.received_series_count = 0

    def build_feature_dataset(self, series_list: list[OHLCVSeries]) -> pd.DataFrame:
        self.received_series_count = len(series_list)
        return self._feature_df.copy()


class _StubModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def supported_model_types(self) -> tuple[str, ...]:
        return ("naive_zero", "naive_mean")

    def evaluate(
        self,
        *,
        train_dataset: object,
        test_dataset: object,
        model_type: str,
        target_column: str,
        id_columns=("date", "ticker"),
    ) -> ModelEvaluationResult:
        train_df = cast(pd.DataFrame, train_dataset)
        test_df = cast(pd.DataFrame, test_dataset)
        self.calls.append((model_type, len(train_df), len(test_df)))

        base = 0.1 if model_type == "naive_zero" else 0.2
        mae = base + (len(test_df) / 100.0)
        rmse = base + (len(train_df) / 100.0)
        direction_accuracy = 0.5 + base

        return ModelEvaluationResult(
            metrics={
                "mae": mae,
                "rmse": rmse,
                "direction_accuracy": direction_accuracy,
            },
            predictions=pd.DataFrame(),
            trained_artifact=object(),
            model_metadata={"model_id": model_type},
        )


def _make_series(ticker: str) -> OHLCVSeries:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    closes = [100.0 + idx for idx in range(12)]
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000 + idx for idx in range(12)],
        }
    )
    return OHLCVSeries(
        ticker=Ticker(ticker),
        date_range=DateRange("2024-01-01", "2024-01-12"),
        interval=Interval("1d"),
        df=df,
    )


def _feature_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    rows: list[dict[str, object]] = []
    for ticker in ("AAA", "BBB"):
        for idx, day in enumerate(dates):
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "ret_1d": float(idx) / 100.0,
                    "mom_20d": float(idx) / 50.0,
                    "target_ret_1d": float(idx) / 200.0,
                }
            )
    return pd.DataFrame(rows)


def test_backtest_executes_multiple_models_and_aggregates_fold_metrics() -> None:
    fetch_stub = _StubFetchMarketData({"AAA": _make_series("AAA"), "BBB": _make_series("BBB")})
    feature_store = _StubFeatureStore(_feature_frame())
    model = _StubModel()

    use_case = Backtest(
        fetch_market_data=cast(FetchMarketData, cast(object, fetch_stub)),
        feature_store=cast(FeatureStorePort, cast(object, feature_store)),
        model=cast(ModelPort, cast(object, model)),
        training_tickers=("AAA", "BBB"),
        default_interval="1d",
    )

    report = use_case.execute(
        application_dto.BacktestRequest(
            model_ids=["naive_zero", "naive_mean"],
            years=1,
            end="2024-01-08",
            interval="1d",
            min_train_days=4,
            test_window_days=2,
            step_days=2,
        )
    )

    assert feature_store.received_series_count == 2
    assert len(fetch_stub.calls) == 2
    assert len(report.results) == 2
    assert report.split_spec["name"] == "walk_forward"
    assert report.split_spec["fold_count"] == 2

    by_model = {row.model_id: row for row in report.results}
    assert set(by_model.keys()) == {"naive_zero", "naive_mean"}

    naive_zero = by_model["naive_zero"]
    assert naive_zero.metrics["fold_count"] == 2
    assert naive_zero.metrics["mae"] == pytest.approx(0.14)
    assert naive_zero.metrics["rmse"] == pytest.approx(0.20)
    assert naive_zero.metrics["direction_accuracy"] == pytest.approx(0.6)

    assert len(naive_zero.folds) == 2
    assert naive_zero.folds[0]["fold_index"] == 1
    assert naive_zero.folds[0]["train_end"] == "2024-01-04"
    assert naive_zero.folds[0]["test_start"] == "2024-01-05"
    assert naive_zero.folds[0]["test_end"] == "2024-01-06"

    naive_mean = by_model["naive_mean"]
    assert naive_mean.metrics["fold_count"] == 2
    assert naive_mean.metrics["mae"] == pytest.approx(0.24)
    assert naive_mean.metrics["rmse"] == pytest.approx(0.30)
    assert naive_mean.metrics["direction_accuracy"] == pytest.approx(0.7)

    assert len(model.calls) == 4


def test_backtest_rejects_unsupported_model_ids() -> None:
    fetch_stub = _StubFetchMarketData({"AAA": _make_series("AAA")})
    feature_store = _StubFeatureStore(_feature_frame())
    use_case = Backtest(
        fetch_market_data=cast(FetchMarketData, cast(object, fetch_stub)),
        feature_store=cast(FeatureStorePort, cast(object, feature_store)),
        model=cast(ModelPort, cast(object, _StubModel())),
        training_tickers=("AAA",),
    )

    with pytest.raises(ValueError, match="Unsupported model id"):
        use_case.execute(
            application_dto.BacktestRequest(
                model_ids=["ridge"],
                years=1,
                end="2024-01-08",
                min_train_days=4,
                test_window_days=2,
                step_days=2,
            )
        )



