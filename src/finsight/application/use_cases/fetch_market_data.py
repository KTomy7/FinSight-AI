from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.ports import MarketDataPort
from finsight.domain.value_objects import DateRange, Interval, Ticker

@dataclass(frozen=True, slots=True)
class FetchMarketDataRequest:
    ticker: str
    start_date: Optional[str] = None   # ISO "YYYY-MM-DD"; defaults to today − lookback
    end_date: Optional[str] = None     # ISO "YYYY-MM-DD"; defaults to today
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
        default_lookback_days: int = 365,
        default_interval: str = "1d",
    ) -> None:
        self._market_data = market_data
        self._default_lookback_days = default_lookback_days
        self._default_interval = default_interval

    def execute(self, request: FetchMarketDataRequest) -> FetchMarketDataResult:
        ticker = Ticker(request.ticker)
        interval = Interval(request.interval or self._default_interval)

        end = (
            datetime.date.fromisoformat(request.end_date)
            if request.end_date
            else datetime.date.today()
        )
        start = (
            datetime.date.fromisoformat(request.start_date)
            if request.start_date
            else end - datetime.timedelta(days=self._default_lookback_days)
        )
        date_range = DateRange(start=start, end=end)

        history = self._market_data.fetch_ohlcv(
            ticker=ticker,
            date_range=date_range,
            interval=interval,
        )
        summary = self._market_data.get_summary(ticker) if request.include_summary else None

        return FetchMarketDataResult(history=history, summary=summary)
