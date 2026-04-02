from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from finsight.application.use_cases.load_model_run import LoadModelRun
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.application.use_cases.train_model import TrainModel
from finsight.config.settings import get_settings
from finsight.infrastructure.features.feature_store import PandasFeatureStore
from finsight.infrastructure.market_data.yfinance_provider import YFinanceMarketDataProvider
from finsight.infrastructure.ml.sklearn.baseline import NaiveBaselineModel
from finsight.infrastructure.ml.registry import LocalModelRegistry


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Composition root for concrete implementations used by adapters."""

    fetch_market_data: FetchMarketData
    train_model: TrainModel
    load_model_run: LoadModelRun


@lru_cache(maxsize=1)
def build_container() -> AppContainer:
    """Build and cache the application container instance."""
    settings = get_settings()

    market_data_provider = YFinanceMarketDataProvider()
    fetch_market_data = FetchMarketData(
        market_data=market_data_provider,
        default_lookback_days=settings.stock_data.default_lookback_days,
        default_interval=settings.stock_data.default_interval,
    )

    feature_store = PandasFeatureStore()
    model = NaiveBaselineModel()
    model_registry = LocalModelRegistry()
    train_model = TrainModel(
        fetch_market_data=fetch_market_data,
        feature_store=feature_store,
        model=model,
        model_registry=model_registry,
        training_tickers=settings.ticker_catalog.symbols(),
        supported_model_types=settings.model_defaults.training_model_ids(),
        default_interval=settings.stock_data.default_interval,
    )
    configured_artifact_root = getattr(settings, "artifact_root", "artifacts/runs")
    trusted_artifact_root = (
        configured_artifact_root.strip()
        if isinstance(configured_artifact_root, str) and configured_artifact_root.strip()
        else "artifacts/runs"
    )
    load_model_run = LoadModelRun(model_registry=model_registry, artifact_root=trusted_artifact_root)

    return AppContainer(
        fetch_market_data=fetch_market_data,
        train_model=train_model,
        load_model_run=load_model_run,
    )

