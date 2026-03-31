from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from finsight.domain.entities import OHLCVSeries, StockSummary

MetricValue = float | int | str
SerializableScalar = str | int | float | bool | None
SerializableRow = dict[str, SerializableScalar]


@dataclass(frozen=True, slots=True)
class FetchMarketDataRequest:
    ticker: str
    start_date: str | None = None  # ISO "YYYY-MM-DD"; defaults to today - lookback
    end_date: str | None = None  # ISO "YYYY-MM-DD"; defaults to today
    interval: str | None = None
    include_summary: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "interval": self.interval,
            "include_summary": self.include_summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FetchMarketDataRequest:
        # Normalize date and interval fields to str | None as per the type hints.
        raw_start = payload.get("start_date")
        start_date = None if raw_start is None else str(raw_start)

        raw_end = payload.get("end_date")
        end_date = None if raw_end is None else str(raw_end)

        raw_interval = payload.get("interval")
        interval = None if raw_interval is None else str(raw_interval)

        # Interpret include_summary more safely than bool(payload.get(...)):
        raw_include = payload.get("include_summary", True)
        if isinstance(raw_include, bool):
            include_summary = raw_include
        elif isinstance(raw_include, str):
            value = raw_include.strip().casefold()
            if value in {"true", "1", "yes", "y", "on"}:
                include_summary = True
            elif value in {"false", "0", "no", "n", "off"}:
                include_summary = False
            else:
                include_summary = bool(value)
        else:
            include_summary = bool(raw_include)

        return cls(
            ticker=str(payload.get("ticker", "")),
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            include_summary=include_summary,
        )


@dataclass(frozen=True, slots=True)
class FetchMarketDataResult:
    history: OHLCVSeries
    summary: StockSummary | None

    @property
    def summary_dict(self) -> Mapping[str, Any] | None:
        return None if self.summary is None else self.summary.data


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    tickers: tuple[str, ...]
    start_date: str | None
    end_date: str | None
    interval: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tickers": list(self.tickers),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "interval": self.interval,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetSpec:
        return cls(
            tickers=tuple(str(value) for value in payload.get("tickers", [])),
            start_date=payload.get("start_date"),
            end_date=payload.get("end_date"),
            interval=str(payload.get("interval", "1d")),
        )


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    feature_columns: tuple[str, ...]
    target_column: str
    date_column: str = "date"
    id_columns: tuple[str, ...] = ("date", "ticker")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "date_column": self.date_column,
            "id_columns": list(self.id_columns),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        return cls(
            feature_columns=tuple(str(value) for value in payload.get("feature_columns", [])),
            target_column=str(payload.get("target_column", "")),
            date_column=str(payload.get("date_column", "date")),
            id_columns=tuple(str(value) for value in payload.get("id_columns", ["date", "ticker"])),
        )


@dataclass(frozen=True, slots=True)
class TrainModelRequest:
    cutoff_date: str
    years: int = 2
    end: str | None = None
    interval: str | None = None
    model_types: list[str] = field(default_factory=lambda: ["naive_zero", "naive_mean"])
    artifacts_dir: str = "artifacts/runs"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cutoff_date": self.cutoff_date,
            "years": self.years,
            "end": self.end,
            "interval": self.interval,
            "model_types": list(self.model_types),
            "artifacts_dir": self.artifacts_dir,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainModelRequest:
        years_raw = payload.get("years", 2)
        years: int
        try:
            years = int(years_raw)
        except (TypeError, ValueError):
            years = 2

        end_raw = payload.get("end")
        end: str | None = None if end_raw is None else str(end_raw)

        interval_raw = payload.get("interval")
        interval: str | None = None if interval_raw is None else str(interval_raw)

        return cls(
            cutoff_date=str(payload.get("cutoff_date", "")),
            years=years,
            end=end,
            interval=interval,
            model_types=[str(value) for value in payload.get("model_types", ["naive_zero", "naive_mean"])],
            artifacts_dir=str(payload.get("artifacts_dir", "artifacts/runs")),
        )


@dataclass(frozen=True, slots=True)
class TrainModelResult:
    run_dirs: dict[str, str]
    metrics: dict[str, dict[str, MetricValue]]
    dataset_spec: DatasetSpec | None = None
    feature_spec: FeatureSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dirs": dict(self.run_dirs),
            "metrics": {model_id: dict(model_metrics) for model_id, model_metrics in self.metrics.items()},
            "dataset_spec": None if self.dataset_spec is None else self.dataset_spec.to_dict(),
            "feature_spec": None if self.feature_spec is None else self.feature_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainModelResult:
        run_dirs_raw = payload.get("run_dirs", {})
        metrics_raw = payload.get("metrics", {})
        dataset_raw = payload.get("dataset_spec")
        feature_raw = payload.get("feature_spec")

        run_dirs: dict[str, str] = {}
        if isinstance(run_dirs_raw, Mapping):
            run_dirs = {str(key): str(value) for key, value in run_dirs_raw.items()}

        metrics: dict[str, dict[str, MetricValue]] = {}
        if isinstance(metrics_raw, Mapping):
            for model_id, model_metrics in metrics_raw.items():
                if isinstance(model_metrics, Mapping):
                    metrics[str(model_id)] = {
                        str(metric_name): value
                        for metric_name, value in model_metrics.items()
                    }

        dataset_spec = DatasetSpec.from_dict(dataset_raw) if isinstance(dataset_raw, Mapping) else None
        feature_spec = FeatureSpec.from_dict(feature_raw) if isinstance(feature_raw, Mapping) else None
        return cls(
            run_dirs=run_dirs,
            metrics=metrics,
            dataset_spec=dataset_spec,
            feature_spec=feature_spec,
        )


@dataclass(frozen=True, slots=True)
class ForecastResult:
    model_id: str
    ticker: str
    horizon_days: int
    predictions: list[SerializableRow]
    generated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ticker": self.ticker,
            "horizon_days": self.horizon_days,
            "predictions": [dict(row) for row in self.predictions],
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ForecastResult:
        predictions_raw = payload.get("predictions", [])
        predictions: list[SerializableRow] = []
        if isinstance(predictions_raw, list):
            for row in predictions_raw:
                if isinstance(row, Mapping):
                    predictions.append({str(key): row[key] for key in row})

        return cls(
            model_id=str(payload.get("model_id", "")),
            ticker=str(payload.get("ticker", "")),
            horizon_days=int(payload.get("horizon_days", 0)),
            predictions=predictions,
            generated_at=payload.get("generated_at"),
        )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    model_id: str
    metrics: dict[str, MetricValue]
    folds: list[SerializableRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "metrics": dict(self.metrics),
            "folds": [dict(row) for row in self.folds],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BacktestResult:
        metrics_raw = payload.get("metrics", {})
        folds_raw = payload.get("folds", [])

        metrics: dict[str, MetricValue] = {}
        if isinstance(metrics_raw, Mapping):
            metrics = {str(key): value for key, value in metrics_raw.items()}

        folds: list[SerializableRow] = []
        if isinstance(folds_raw, list):
            for row in folds_raw:
                if isinstance(row, Mapping):
                    folds.append({str(key): row[key] for key in row})

        return cls(
            model_id=str(payload.get("model_id", "")),
            metrics=metrics,
            folds=folds,
        )


__all__ = [
    "BacktestResult",
    "DatasetSpec",
    "FeatureSpec",
    "FetchMarketDataRequest",
    "FetchMarketDataResult",
    "ForecastResult",
    "MetricValue",
    "TrainModelRequest",
    "TrainModelResult",
]

