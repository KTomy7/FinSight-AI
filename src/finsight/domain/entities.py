from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

from finsight.domain.value_objects import Interval, Period, Ticker

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

@dataclass(frozen=True, slots=True)
class OHLCVSeries:
    """
    Historical market data.

    For the thesis demo, we keep the underlying DataFrame as-is. If you later
    introduce an API layer or persistence, consider converting to a list-of-bars
    DTO at this boundary.
    """

    ticker: Ticker
    period: Period
    interval: Interval
    df: "pd.DataFrame"


@dataclass(frozen=True, slots=True)
class StockSummary:
    ticker: Ticker
    data: Mapping[str, Any]

