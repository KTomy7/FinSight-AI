from __future__ import annotations

from typing import Protocol, runtime_checkable

from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.value_objects import Interval, Period, Ticker

@runtime_checkable
class MarketDataPort(Protocol):
    def get_price_history(self, ticker: Ticker, period: Period, interval: Interval) -> OHLCVSeries:
        raise NotImplementedError

    def get_summary(self, ticker: Ticker) -> StockSummary:
        raise NotImplementedError

