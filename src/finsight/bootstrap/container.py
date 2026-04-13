from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.application.use_cases.train_model import TrainModel
from finsight.config.settings import get_settings
from finsight.infrastructure.features.feature_store import PandasFeatureStore
from finsight.infrastructure.market_data.yfinance_provider import YFinanceMarketDataProvider
from finsight.infrastructure.ml.sklearn import LinearSklearnModel, NaiveBaselineModel, SklearnModelRouter
from finsight.infrastructure.ml.registry import LocalFileModelRegistry


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Composition root for concrete implementations used by adapters."""

    fetch_market_data: FetchMarketData
    train_model: TrainModel


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
    model = SklearnModelRouter(adapters=[NaiveBaselineModel(), LinearSklearnModel()])
    model_registry = LocalFileModelRegistry()
    train_model = TrainModel(
        fetch_market_data=fetch_market_data,
        feature_store=feature_store,
        model=model,
        model_registry=model_registry,
        training_tickers=settings.ticker_catalog.symbols(),
        supported_model_types=settings.model_defaults.training_model_ids(),
        default_interval=settings.stock_data.default_interval,
    )

    return AppContainer(fetch_market_data=fetch_market_data, train_model=train_model)

