import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

import finsight.application.use_cases.train_model as train_model_module
from finsight.application.use_cases.fetch_market_data import FetchMarketData, FetchMarketDataRequest
from finsight.application.use_cases.train_model import TrainModel, TrainModelRequest
from finsight.domain.entities import OHLCVSeries
from finsight.domain.value_objects import DateRange, Interval, Ticker
from finsight.infrastructure.features import PandasFeatureStore
from finsight.infrastructure.ml.sklearn import NaiveBaselineModel
from finsight.infrastructure.persistence import LocalFileModelRegistry


class _StubFetchMarketData:
    def __init__(self, series_by_ticker: dict[str, OHLCVSeries]) -> None:
        self._series_by_ticker = series_by_ticker
        self.calls: list[FetchMarketDataRequest] = []

    def execute(self, request: FetchMarketDataRequest) -> SimpleNamespace:
        self.calls.append(request)
        return SimpleNamespace(history=self._series_by_ticker[request.ticker])


def _make_ohlcv_series(ticker: str, *, start: str = "2024-01-01", periods: int = 900) -> OHLCVSeries:
    dates = pd.date_range(start, periods=periods, freq="D")
    close = [100.0 + (idx * 0.25) for idx in range(periods)]
    df = pd.DataFrame(
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
        df=df,
    )


def test_execute_uses_expected_fetch_date_window_and_default_interval(tmp_path) -> None:
    tickers = ("AAPL", "JPM", "XOM")
    stub = _StubFetchMarketData({ticker: _make_ohlcv_series(ticker) for ticker in tickers})
    train_model = TrainModel(
        fetch_market_data=cast(FetchMarketData, cast(object, stub)),
        feature_store=PandasFeatureStore(),
        model=NaiveBaselineModel(),
        model_registry=LocalFileModelRegistry(),
        training_tickers=tickers,
        default_interval="1wk",
    )

    request = TrainModelRequest(
        cutoff_date="2025-06-01",
        years=2,
        end="2026-03-17",
        model_types=["naive_zero"],
        artifacts_dir=str(tmp_path / "runs"),
    )

    train_model.execute(request)

    assert len(stub.calls) == len(tickers)

    expected_end = date.fromisoformat("2026-03-17")
    expected_start = expected_end - timedelta(days=(2 * 365) - 1)

    for call in stub.calls:
        assert call.start_date == expected_start.isoformat()
        assert call.end_date == expected_end.isoformat()
        assert call.interval == "1wk"
        assert call.include_summary is False


def test_execute_writes_artifacts_and_applies_unique_run_dir_suffix(tmp_path, monkeypatch) -> None:
    fixed_now = datetime(2026, 3, 27, 12, 34, 56, tzinfo=timezone.utc)

    class _FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(train_model_module, "datetime", _FixedDateTime)

    tickers = ("AAPL", "JPM")
    stub = _StubFetchMarketData({ticker: _make_ohlcv_series(ticker) for ticker in tickers})
    train_model = TrainModel(
        fetch_market_data=cast(FetchMarketData, cast(object, stub)),
        feature_store=PandasFeatureStore(),
        model=NaiveBaselineModel(),
        model_registry=LocalFileModelRegistry(),
        training_tickers=tickers,
        default_interval="1d",
    )

    artifacts_root = tmp_path / "runs"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "2026-03-27T123456Z__naive_zero").mkdir()

    response = train_model.execute(
        TrainModelRequest(
            cutoff_date="2025-06-01",
            years=2,
            end="2026-03-17",
            model_types=["naive_zero", "naive_mean"],
            artifacts_dir=str(artifacts_root),
        )
    )

    naive_zero_dir = artifacts_root / "2026-03-27T123456Z__naive_zero_1"
    naive_mean_dir = artifacts_root / "2026-03-27T123456Z__naive_mean"

    assert response.run_dirs["naive_zero"] == str(naive_zero_dir)
    assert response.run_dirs["naive_mean"] == str(naive_mean_dir)

    for model_type, run_dir in (("naive_zero", naive_zero_dir), ("naive_mean", naive_mean_dir)):
        metrics_path = run_dir / "metrics.json"
        metadata_path = run_dir / "metadata.json"
        predictions_path = run_dir / "predictions.csv"

        assert metrics_path.exists()
        assert metadata_path.exists()
        assert predictions_path.exists()

        metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata_json = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictions_df = pd.read_csv(predictions_path)

        assert metadata_json["model_type"] == model_type
        assert metadata_json["run_id"] == run_dir.name
        assert metadata_json["tickers"] == ["AAPL", "JPM"]
        assert metadata_json["interval"] == "1d"

        assert metrics_json["n_train"] > 0
        assert metrics_json["n_test"] > 0
        assert {"date", "ticker", "y_true", "y_pred"}.issubset(predictions_df.columns)

        assert response.metrics[model_type]["n_train"] == metrics_json["n_train"]
        assert response.metrics[model_type]["n_test"] == metrics_json["n_test"]


def test_execute_rejects_unsupported_model_types_from_model_port(tmp_path) -> None:
    tickers = ("AAPL",)
    stub = _StubFetchMarketData({ticker: _make_ohlcv_series(ticker) for ticker in tickers})
    train_model = TrainModel(
        fetch_market_data=cast(FetchMarketData, cast(object, stub)),
        feature_store=PandasFeatureStore(),
        model=NaiveBaselineModel(),
        model_registry=LocalFileModelRegistry(),
        training_tickers=tickers,
    )

    with pytest.raises(ValueError, match=r"Unsupported model type\(s\)"):
        train_model.execute(
            TrainModelRequest(
                cutoff_date="2025-06-01",
                years=2,
                end="2026-03-17",
                model_types=["naive_last"],
                artifacts_dir=str(tmp_path / "runs"),
            )
        )

    assert stub.calls == []



