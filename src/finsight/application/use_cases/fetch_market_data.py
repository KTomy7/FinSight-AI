from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.ports import MarketDataPort
from finsight.domain.value_objects import Interval, Period, Ticker

@dataclass(frozen=True, slots=True)
class FetchMarketDataRequest:
    ticker: str
    period: Optional[str] = None
    interval: Optional[str] = None
    include_summary: bool = True


@dataclass(frozen=True, slots=True)
class FetchMarketDataResult:
    history: OHLCVSeries
    summary: Optional[StockSummary]

    @property
    def summary_dict(self) -> Optional[Mapping[str, Any]]:
        return None if self.summary is None else self.summary.data


class FetchMarketData:
    def __init__(
        self,
        market_data: MarketDataPort,
        *,
        default_period: str = "1y",
        default_interval: str = "1d",
    ) -> None:
        self._market_data = market_data
        self._default_period = default_period
        self._default_interval = default_interval

    def execute(self, request: FetchMarketDataRequest) -> FetchMarketDataResult:
        ticker = Ticker(request.ticker)
        period = Period(request.period or self._default_period)
        interval = Interval(request.interval or self._default_interval)

        history = self._market_data.get_price_history(ticker=ticker, period=period, interval=interval)
        summary = self._market_data.get_summary(ticker) if request.include_summary else None

        return FetchMarketDataResult(history=history, summary=summary)

