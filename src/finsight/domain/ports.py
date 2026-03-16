from __future__ import annotations

from typing import Protocol, runtime_checkable

from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.value_objects import DateRange, Interval, Ticker

@runtime_checkable
class MarketDataPort(Protocol):
    def fetch_ohlcv(
        self,
        ticker: Ticker,
        date_range: DateRange,
        interval: Interval,
    ) -> OHLCVSeries:
        raise NotImplementedError

    def get_summary(self, ticker: Ticker) -> StockSummary:
        raise NotImplementedError

