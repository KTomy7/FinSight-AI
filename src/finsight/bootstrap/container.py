from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.config.settings import get_settings
from finsight.infrastructure.market_data.yfinance_provider import YFinanceMarketDataProvider


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Composition root for concrete implementations used by adapters."""

    fetch_market_data: FetchMarketData


@lru_cache(maxsize=1)
def build_container() -> AppContainer:
    """Build and cache the application container instance."""
    settings = get_settings()

    market_data_provider = YFinanceMarketDataProvider()
    fetch_market_data = FetchMarketData(
        market_data=market_data_provider,
        default_period=settings.stock_data.default_period,
        default_interval=settings.stock_data.default_interval,
    )

    return AppContainer(fetch_market_data=fetch_market_data)

