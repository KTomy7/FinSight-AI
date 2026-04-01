from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

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


@runtime_checkable
class FeatureStorePort(Protocol):
    def build_feature_dataset(self, series_list: Sequence[OHLCVSeries]) -> object:
        raise NotImplementedError

    def split_train_test(
        self,
        dataset: object,
        *,
        cutoff_date: str,
        date_col: str = "date",
        inclusive_test: bool = True,
    ) -> tuple[object, object]:
        raise NotImplementedError

    def frame_date_range(self, dataset: object, *, date_col: str = "date") -> tuple[str, str]:
        raise NotImplementedError

    def row_count(self, dataset: object) -> int:
        raise NotImplementedError

    def feature_columns(self) -> tuple[str, ...]:
        raise NotImplementedError


@runtime_checkable
class ModelPort(Protocol):
    def evaluate(
        self,
        *,
        train_dataset: object,
        test_dataset: object,
        model_type: str,
        target_column: str,
        id_columns: Sequence[str] = ("date", "ticker"),
    ) -> tuple[Mapping[str, float], object]:
        raise NotImplementedError

    def supported_model_types(self) -> tuple[str, ...]:
        raise NotImplementedError


@runtime_checkable
class ModelRegistryPort(Protocol):
    def create_run_dir(self, *, artifact_root: str, model_run_id: str) -> str:
        raise NotImplementedError

    def save_metrics(self, *, run_dir: str, metrics: Mapping[str, float | int | str]) -> None:
        raise NotImplementedError

    def save_manifest(self, *, run_dir: str, manifest: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def save_predictions(self, *, run_dir: str, predictions: object) -> None:
        raise NotImplementedError

    def save_run(
        self,
        *,
        artifact_root: str,
        model_run_id: str,
        model_artifact: object,
        manifest: Mapping[str, Any],
        metrics: Mapping[str, float | int | str],
        predictions: object | None = None,
    ) -> str:
        raise NotImplementedError

    def load_run(self, *, artifact_root: str, model_run_id: str) -> Mapping[str, Any]:
        raise NotImplementedError


