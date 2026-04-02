from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

from finsight.domain.value_objects import DateRange, Interval, Ticker

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
    date_range: DateRange
    interval: Interval
    df: "pd.DataFrame"


@dataclass(frozen=True, slots=True)
class StockSummary:
    ticker: Ticker
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelRunRecord:
    """Persisted model-run payload loaded from the model registry."""

    model_run_id: str
    run_dir: str
    metrics: dict[str, float | int | str]
    manifest: dict[str, Any]
    predictions_path: str
    model_artifact_path: str | None = None


