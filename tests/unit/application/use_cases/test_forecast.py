from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

from finsight.application.dto import FetchMarketDataRequest, ForecastRequest, ModelRunArtifacts
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.application.use_cases.forecast import Forecast
from finsight.domain.entities import OHLCVSeries
from finsight.domain.ports import FeatureStorePort, ModelRegistryPort
from finsight.domain.value_objects import DateRange, Interval, Ticker


@dataclass
class _PredictSpyModel:
    expected: list[float]
    calls: list[pd.DataFrame]

    def predict(self, X: object) -> list[float]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("expected DataFrame input")
        self.calls.append(X.copy())
        return list(self.expected)


class _StubFetchMarketData:
    def __init__(self, history: OHLCVSeries) -> None:
        self._history = history
        self.calls: list[FetchMarketDataRequest] = []

    def execute(self, request: FetchMarketDataRequest) -> SimpleNamespace:
        self.calls.append(request)
        return SimpleNamespace(history=self._history)


class _StubFeatureStore(FeatureStorePort):
    def __init__(self, inference_df: pd.DataFrame) -> None:
        self._inference_df = inference_df

    def build_feature_dataset(self, series_list):
        raise NotImplementedError

    def build_inference_feature_dataset(self, series_list):
        return self._inference_df.copy()

    def split_train_test(self, dataset, *, cutoff_date: str, date_col: str = "date", inclusive_test: bool = True):
        raise NotImplementedError

    def frame_date_range(self, dataset, *, date_col: str = "date"):
        raise NotImplementedError

    def row_count(self, dataset):
        raise NotImplementedError

    def feature_columns(self):
        raise NotImplementedError


class _StubRegistry(ModelRegistryPort):
    def __init__(self, artifacts: ModelRunArtifacts | None = None, *, latest_run_id_error: Exception | None = None) -> None:
        self._artifacts = artifacts
        self._latest_run_id_error = latest_run_id_error
        self.latest_run_calls: list[tuple[str, str]] = []
        self.load_artifact_calls: list[tuple[str, str]] = []

    def create_run_dir(self, *, artifact_root: str, run_id: str) -> str:
        raise NotImplementedError

    def latest_run_id(self, *, artifact_root: str, model_id: str) -> str:
        self.latest_run_calls.append((artifact_root, model_id))
        if self._latest_run_id_error is not None:
            raise self._latest_run_id_error
        return "2026-04-10T120000Z__ridge"

    def save_model(self, *, run_dir: str, model: object) -> None:
        raise NotImplementedError

    def load_model(self, *, artifact_root: str, run_id: str) -> object:
        raise NotImplementedError

    def save_metrics(self, *, run_dir: str, metrics):
        raise NotImplementedError

    def load_metrics(self, *, artifact_root: str, run_id: str):
        raise NotImplementedError

    def save_manifest(self, *, run_dir: str, manifest):
        raise NotImplementedError

    def load_manifest(self, *, artifact_root: str, run_id: str):
        raise NotImplementedError

    def save_predictions(self, *, run_dir: str, predictions):
        raise NotImplementedError

    def load_run_artifacts(self, *, artifact_root: str, run_id: str) -> object:
        self.load_artifact_calls.append((artifact_root, run_id))
        if self._artifacts is None:
            raise RuntimeError("artifacts not configured")
        return self._artifacts


def _make_history(ticker: str) -> OHLCVSeries:
    dates = pd.date_range("2026-04-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": [95.0 + idx for idx in range(10)],
            "High": [96.0 + idx for idx in range(10)],
            "Low": [94.0 + idx for idx in range(10)],
            "Close": [100.0 + idx for idx in range(10)],
            "Volume": [1000 + idx for idx in range(10)],
        }
    )
    return OHLCVSeries(
        ticker=Ticker(ticker),
        date_range=DateRange(dates[0].date().isoformat(), dates[-1].date().isoformat()),
        interval=Interval("1d"),
        df=frame,
    )


def test_execute_returns_price_path_from_predicted_returns() -> None:
    history = _make_history("AAPL")
    inference_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-09", "2026-04-10"]),
            "ticker": ["AAPL", "AAPL"],
            "f1": [1.0, 2.0],
            "f2": [10.0, 20.0],
        }
    )

    model = _PredictSpyModel(expected=[0.10, -0.05], calls=[])
    artifacts = ModelRunArtifacts(
        run_id="2026-04-10T120000Z__ridge",
        run_dir="artifacts/runs/2026-04-10T120000Z__ridge",
        model_path="model.joblib",
        metrics_path="metrics.json",
        manifest_path="manifest.json",
        predictions_path="predictions.csv",
        model=model,
        metrics={},
        manifest={"feature_columns": ["f2", "f1"]},
    )

    forecast = Forecast(
        fetch_market_data=cast(FetchMarketData, cast(object, _StubFetchMarketData(history))),
        feature_store=_StubFeatureStore(inference_df),
        model_registry=_StubRegistry(artifacts),
    )

    result = forecast.execute(
        ForecastRequest(ticker="AAPL", model_id="ridge", horizon_days=2, artifacts_dir="artifacts/runs")
    )

    assert result.model_id == "ridge"
    assert result.ticker == "AAPL"
    assert result.horizon_days == 2
    assert result.predictions == [
        {"date": "2026-04-11", "pred_ret_1d": 0.10, "pred_close": 119.9},
        {"date": "2026-04-12", "pred_ret_1d": -0.05, "pred_close": 113.905},
    ]

    assert len(model.calls) == 1
    infer_call = model.calls[0]
    assert list(infer_call.columns) == ["f2", "f1"]
    assert len(infer_call) == 2


def test_execute_propagates_error_when_no_runs_exist_for_model_id() -> None:
    history = _make_history("AAPL")
    forecast = Forecast(
        fetch_market_data=cast(FetchMarketData, cast(object, _StubFetchMarketData(history))),
        feature_store=_StubFeatureStore(pd.DataFrame({"date": [], "ticker": [], "f1": []})),
        model_registry=_StubRegistry(latest_run_id_error=FileNotFoundError("No runs found for model_id 'ridge'")),
    )

    with pytest.raises(FileNotFoundError, match="No runs found for model_id 'ridge'"):
        forecast.execute(ForecastRequest(ticker="AAPL", model_id="ridge", horizon_days=7))


def test_execute_rejects_invalid_request_inputs() -> None:
    history = _make_history("AAPL")
    forecast = Forecast(
        fetch_market_data=cast(FetchMarketData, cast(object, _StubFetchMarketData(history))),
        feature_store=_StubFeatureStore(pd.DataFrame()),
        model_registry=_StubRegistry(),
    )

    with pytest.raises(ValueError, match="ticker must be a non-empty string"):
        forecast.execute(ForecastRequest(ticker="   ", model_id="ridge", horizon_days=7))

    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        forecast.execute(ForecastRequest(ticker="AAPL", model_id="", horizon_days=7))

    with pytest.raises(ValueError, match="horizon_days must be a positive integer"):
        forecast.execute(ForecastRequest(ticker="AAPL", model_id="ridge", horizon_days=0))

