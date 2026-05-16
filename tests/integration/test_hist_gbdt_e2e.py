"""End-to-end integration tests for HistGradientBoostingModel (hist_gbdt)."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast

import pandas as pd

from finsight.application.dto import FetchMarketDataRequest, ForecastRequest, RunSummary, TrainModelRequest
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.application.use_cases.forecast import Forecast
from finsight.application.use_cases.train_model import TrainModel
from finsight.domain.entities import OHLCVSeries
from finsight.domain.metrics import SUPPORTED_METRIC_NAMES
from finsight.domain.ports import RunRegistryPort
from finsight.domain.value_objects import DateRange, Interval, Ticker
from finsight.infrastructure.features.feature_store import PandasFeatureStore
from finsight.infrastructure.ml.registry import LocalFileModelRegistry
from finsight.infrastructure.ml.sklearn import HistGradientBoostingModel, NaiveBaselineModel, SklearnModelRouter


class _StubFetchMarketData:
    """Stub market data provider that returns synthetic OHLCV data."""

    def __init__(self, data_by_ticker: dict[str, OHLCVSeries]) -> None:
        self.data_by_ticker = data_by_ticker
        self.calls: list[FetchMarketDataRequest] = []

    def execute(self, request: FetchMarketDataRequest) -> object:
        self.calls.append(request)
        ticker = request.ticker
        if ticker not in self.data_by_ticker:
            raise ValueError(f"No data for ticker {ticker}")
        return SimpleNamespace(history=self.data_by_ticker[ticker])


class _StubRunRegistry(RunRegistryPort):
    """Stub implementation of RunRegistryPort for integration tests."""

    def __init__(self) -> None:
        self.recorded_runs: list[RunSummary] = []

    def load_registry(self, *, artifact_root: str):
        return None

    def record_completed_run(self, *, artifact_root: str, run_summary: RunSummary) -> None:
        self.recorded_runs.append(run_summary)


def _make_synthetic_ohlcv_series(ticker: str) -> OHLCVSeries:
    """Create a synthetic OHLCV series with features for model training."""
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    close = [100.0 + (idx * 0.25) for idx in range(len(dates))]
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


def test_hist_gbdt_training_end_to_end() -> None:
    """Verify hist_gbdt can train, evaluate, and save artifacts."""
    with TemporaryDirectory() as tmp_dir:
        # Setup
        tickers = ("AAPL",)
        data = {ticker: _make_synthetic_ohlcv_series(ticker) for ticker in tickers}
        stub_market_data = _StubFetchMarketData(data)

        train_model = TrainModel(
            fetch_market_data=cast(FetchMarketData, stub_market_data),
            feature_store=PandasFeatureStore(),
            model=SklearnModelRouter(adapters=[
                NaiveBaselineModel(),
                HistGradientBoostingModel(),
            ]),
            model_registry=LocalFileModelRegistry(),
            run_registry=_StubRunRegistry(),
            training_tickers=tickers,
            supported_model_types=("hist_gbdt",),
        )

        # Execute
        request = TrainModelRequest(
            cutoff_date="2024-06-01",
            years=1,
            end="2024-12-31",
            model_types=["hist_gbdt"],
            artifacts_dir=tmp_dir,
        )
        result = train_model.execute(request)

        # Verify
        assert "hist_gbdt" in result.run_dirs
        assert isinstance(result.run_dirs["hist_gbdt"], str)

        run_dir = result.run_dirs["hist_gbdt"]
        assert Path(run_dir).exists()

        # Verify metrics
        assert "hist_gbdt" in result.metrics
        metrics = result.metrics["hist_gbdt"]
        assert set(SUPPORTED_METRIC_NAMES).issubset(metrics)

        # Verify artifacts exist
        assert (Path(run_dir) / "metrics.json").exists()
        assert (Path(run_dir) / "manifest.json").exists()
        assert (Path(run_dir) / "predictions.csv").exists()

        # Verify manifest content
        manifest_path = Path(run_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["model_id"] == "hist_gbdt"
        assert manifest["params"]["model_metadata"]["model_id"] == "hist_gbdt"
        assert manifest["params"]["model_metadata"]["base_estimator"] == "HistGradientBoostingRegressor"

        # Verify predictions
        predictions_path = Path(run_dir) / "predictions.csv"
        predictions_df = pd.read_csv(predictions_path)
        assert "date" in predictions_df.columns
        assert "ticker" in predictions_df.columns
        assert "y_true" in predictions_df.columns
        assert "y_pred" in predictions_df.columns
        assert len(predictions_df) > 0


def test_hist_gbdt_training_with_multiple_models() -> None:
    """Verify hist_gbdt trains alongside other models."""
    with TemporaryDirectory() as tmp_dir:
        # Setup
        tickers = ("AAPL",)
        data = {ticker: _make_synthetic_ohlcv_series(ticker) for ticker in tickers}
        stub_market_data = _StubFetchMarketData(data)

        train_model = TrainModel(
            fetch_market_data=cast(FetchMarketData, stub_market_data),
            feature_store=PandasFeatureStore(),
            model=SklearnModelRouter(adapters=[
                NaiveBaselineModel(),
                HistGradientBoostingModel(),
            ]),
            model_registry=LocalFileModelRegistry(),
            run_registry=_StubRunRegistry(),
            training_tickers=tickers,
            supported_model_types=("naive_zero", "hist_gbdt"),
        )

        # Execute: train both naive_zero and hist_gbdt
        request = TrainModelRequest(
            cutoff_date="2024-06-01",
            years=1,
            end="2024-12-31",
            model_types=["naive_zero", "hist_gbdt"],
            artifacts_dir=tmp_dir,
        )
        result = train_model.execute(request)

        # Verify both models trained
        assert "naive_zero" in result.run_dirs
        assert "hist_gbdt" in result.run_dirs

        # Verify both have metrics
        assert "naive_zero" in result.metrics
        assert "hist_gbdt" in result.metrics


def test_hist_gbdt_deterministic_predictions() -> None:
    """Verify hist_gbdt produces identical predictions on same data."""
    with TemporaryDirectory() as tmp_dir_1, TemporaryDirectory() as tmp_dir_2:
        from finsight.config.settings import get_settings
        from finsight.bootstrap.container import build_container

        # Build container to get properly configured training (respecting config.yaml)
        build_container.cache_clear()
        container = build_container()
        settings = get_settings()

        tickers = ("AAPL",)
        data = {ticker: _make_synthetic_ohlcv_series(ticker) for ticker in tickers}

        # Train 1
        stub_market_data_1 = _StubFetchMarketData(data)
        train_model_1 = TrainModel(
            fetch_market_data=cast(FetchMarketData, stub_market_data_1),
            feature_store=PandasFeatureStore(),
            model=SklearnModelRouter(adapters=[
                NaiveBaselineModel(),
                HistGradientBoostingModel(),
            ]),
            model_registry=LocalFileModelRegistry(),
            run_registry=_StubRunRegistry(),
            training_tickers=tickers,
            supported_model_types=("hist_gbdt",),
        )
        request = TrainModelRequest(
            cutoff_date="2024-06-01",
            years=1,
            end="2024-12-31",
            model_types=["hist_gbdt"],
            artifacts_dir=tmp_dir_1,
        )
        result_1 = train_model_1.execute(request)

        # Train 2 (same data, same random_state)
        stub_market_data_2 = _StubFetchMarketData(data)
        train_model_2 = TrainModel(
            fetch_market_data=cast(FetchMarketData, stub_market_data_2),
            feature_store=PandasFeatureStore(),
            model=SklearnModelRouter(adapters=[
                NaiveBaselineModel(),
                HistGradientBoostingModel(),
            ]),
            model_registry=LocalFileModelRegistry(),
            run_registry=_StubRunRegistry(),
            training_tickers=tickers,
            supported_model_types=("hist_gbdt",),
        )
        request = TrainModelRequest(
            cutoff_date="2024-06-01",
            years=1,
            end="2024-12-31",
            model_types=["hist_gbdt"],
            artifacts_dir=tmp_dir_2,
        )
        result_2 = train_model_2.execute(request)

        # Load predictions from both runs
        pred_path_1 = Path(result_1.run_dirs["hist_gbdt"]) / "predictions.csv"
        pred_path_2 = Path(result_2.run_dirs["hist_gbdt"]) / "predictions.csv"

        pred_df_1 = pd.read_csv(pred_path_1)
        pred_df_2 = pd.read_csv(pred_path_2)

        # Verify predictions are identical (deterministic)
        pd.testing.assert_frame_equal(pred_df_1, pred_df_2)


def test_hist_gbdt_config_integration() -> None:
    """Verify hist_gbdt is discoverable and configurable in Settings."""
    from finsight.config.settings import get_settings

    settings = get_settings()

    # Verify hist_gbdt is in training models
    training_ids = settings.model_defaults.training_model_ids()
    assert "hist_gbdt" in training_ids

    # Verify hist_gbdt is in prediction models
    prediction_ids = settings.model_defaults.prediction_model_ids()
    assert "hist_gbdt" in prediction_ids

    # Verify label mapping
    id_to_label = settings.model_defaults.id_to_label()
    assert id_to_label["hist_gbdt"] == "Histogram Gradient Boosting"


def test_hist_gbdt_router_integration() -> None:
    """Verify hist_gbdt is registered in router via container."""
    from finsight.bootstrap.container import build_container

    build_container.cache_clear()
    container = build_container()

    router = container.train_model._model
    supported_types = router.supported_model_types()

    assert "hist_gbdt" in supported_types
    assert ("naive_zero", "naive_mean", "ridge", "hist_gbdt", "xgboost") == supported_types


def test_hist_gbdt_produces_valid_metadata() -> None:
    """Verify hist_gbdt metadata is serializable and complete."""
    with TemporaryDirectory() as tmp_dir:
        tickers = ("AAPL",)
        data = {ticker: _make_synthetic_ohlcv_series(ticker) for ticker in tickers}
        stub_market_data = _StubFetchMarketData(data)

        train_model = TrainModel(
            fetch_market_data=cast(FetchMarketData, stub_market_data),
            feature_store=PandasFeatureStore(),
            model=SklearnModelRouter(adapters=[HistGradientBoostingModel()]),
            model_registry=LocalFileModelRegistry(),
            run_registry=_StubRunRegistry(),
            training_tickers=tickers,
            supported_model_types=("hist_gbdt",),
        )

        request = TrainModelRequest(
            cutoff_date="2024-06-01",
            years=1,
            end="2024-12-31",
            model_types=["hist_gbdt"],
            artifacts_dir=tmp_dir,
        )
        result = train_model.execute(request)

        run_dir = result.run_dirs["hist_gbdt"]
        manifest_path = Path(run_dir) / "manifest.json"

        # Verify JSON is valid and serializable
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Verify key metadata fields
        params = manifest.get("params", {})
        model_metadata = params.get("model_metadata", {})

        assert model_metadata["model_id"] == "hist_gbdt"
        assert model_metadata["base_estimator"] == "HistGradientBoostingRegressor"
        assert "hyperparams" in model_metadata
        assert model_metadata["hyperparams"]["random_state"] == 42  # Deterministic
        assert "feature_importance_ranking" in model_metadata
        assert isinstance(model_metadata["feature_importance_ranking"], list)



