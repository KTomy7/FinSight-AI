from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.application.use_cases.train_model import TrainModel
from finsight.config.settings import get_settings
from finsight.infrastructure.market_data.yfinance_provider import YFinanceMarketDataProvider


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
    train_model = TrainModel(
        fetch_market_data=fetch_market_data,
        training_tickers=settings.training.training_tickers,
    )

    return AppContainer(fetch_market_data=fetch_market_data, train_model=train_model)

