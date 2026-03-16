from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union, cast

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

        today = datetime.date.today()

        # Resolve the end boundary.
        # Pass the raw ISO string straight through to DateRange so that DateRange
        # owns all parsing and produces its own clear, consistent error messages.
        # No datetime.date.fromisoformat call here.
        end: Union[datetime.date, str] = (
            request.end_date if request.end_date is not None else today
        )

        # Resolve the start boundary.
        if request.start_date is not None:
            # Raw string — DateRange validates and parses it.
            start: Union[datetime.date, str] = request.start_date
        else:
            # Default: start = end − (lookback − 1) days so that the inclusive
            # range [start, end] contains exactly default_lookback_days calendar days.
            #
            # The arithmetic requires end as a datetime.date.  When the caller
            # supplied an end_date string, delegate its parsing to DateRange
            # (single-day sentinel) so that fromisoformat never appears in this
            # layer and DateRange remains the single validation authority.
            if isinstance(end, str):
                # Parse end via DateRange using a guaranteed-low valid start so
                # invalid end strings raise DateRange.end errors consistently.
                end_as_date: datetime.date = cast(
                    datetime.date,
                    DateRange(start=datetime.date.min, end=end).end,
                )
            else:
                end_as_date = end
            start = end_as_date - datetime.timedelta(days=self._default_lookback_days - 1)

        # DateRange performs all remaining validation (start ≤ end, ISO format).
        date_range = DateRange(start=start, end=end)

        history = self._market_data.fetch_ohlcv(
            ticker=ticker,
            date_range=date_range,
            interval=interval,
        )
        summary = (
            self._market_data.get_summary(ticker) if request.include_summary else None
        )

        return FetchMarketDataResult(history=history, summary=summary)
