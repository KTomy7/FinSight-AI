from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from finsight.domain.entities import OHLCVSeries, StockSummary

MetricValue = float | int | str
SerializableScalar = str | int | float | bool | None
SerializableRow = dict[str, SerializableScalar]


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_tuple(value: Any, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return default


def _string_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return list(default)


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
        raw_ticker = payload.get("ticker", "")
        if isinstance(raw_ticker, str):
            ticker = raw_ticker.strip()
        else:
            ticker = ""

        raw_start = payload.get("start_date")
        if raw_start is None:
            start_date = None
        elif isinstance(raw_start, str) and raw_start.strip() == "":
            start_date = None
        else:
            start_date = str(raw_start)

        raw_end = payload.get("end_date")
        if raw_end is None:
            end_date = None
        elif isinstance(raw_end, str) and raw_end.strip() == "":
            end_date = None
        else:
            end_date = str(raw_end)

        raw_interval = payload.get("interval")
        if raw_interval is None:
            interval = None
        elif isinstance(raw_interval, str) and raw_interval.strip() == "":
            interval = None
        else:
            interval = str(raw_interval)
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
            ticker=ticker,
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
        raw_interval = payload.get("interval")
        if raw_interval is None:
            interval = "1d"
        else:
            interval_str = str(raw_interval).strip()
            interval = interval_str or "1d"
        return cls(
            tickers=_string_tuple(payload.get("tickers")),
            start_date=_optional_str(payload.get("start_date")),
            end_date=_optional_str(payload.get("end_date")),
            interval=interval,
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
        raw_target = payload.get("target_column")
        if raw_target is None:
            target_column = ""
        else:
            target_column = str(raw_target).strip()

        raw_date = payload.get("date_column")
        if raw_date is None:
            date_column = "date"
        else:
            date_str = str(raw_date).strip()
            date_column = date_str or "date"

        return cls(
            feature_columns=_string_tuple(payload.get("feature_columns")),
            target_column=target_column,
            date_column=date_column,
            id_columns=_string_tuple(payload.get("id_columns"), default=("date", "ticker")),
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
        # Handle cutoff_date without turning None into the literal string "None".
        raw_cutoff = payload.get("cutoff_date")
        if raw_cutoff is None:
            cutoff_date = ""
        elif isinstance(raw_cutoff, str):
            cutoff_date = raw_cutoff.strip()
        else:
            cutoff_date = str(raw_cutoff)

        # Handle artifacts_dir similarly, falling back to the default when missing/None/blank.
        raw_artifacts_dir = payload.get("artifacts_dir")
        if raw_artifacts_dir is None:
            artifacts_dir = "artifacts/runs"
        elif isinstance(raw_artifacts_dir, str):
            artifacts_dir_candidate = raw_artifacts_dir.strip()
            artifacts_dir = artifacts_dir_candidate or "artifacts/runs"
        else:
            artifacts_dir = str(raw_artifacts_dir)

        # Normalise optional date/interval fields: treat whitespace-only as missing.
        raw_end = payload.get("end")
        if raw_end is None:
            end = None
        elif isinstance(raw_end, str):
            end_candidate = raw_end.strip()
            end = end_candidate or None
        else:
            end = str(raw_end)

        raw_interval = payload.get("interval")
        if raw_interval is None:
            interval = None
        elif isinstance(raw_interval, str):
            interval_candidate = raw_interval.strip()
            interval = interval_candidate or None
        else:
            interval = str(raw_interval)

        return cls(
            cutoff_date=cutoff_date,
            years=_safe_int(payload.get("years", 2), default=2),
            end=end,
            interval=interval,
            model_types=_string_list(payload.get("model_types"), default=["naive_zero", "naive_mean"]),
            artifacts_dir=artifacts_dir,
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
class LoadModelRunRequest:
    model_run_id: str


@dataclass(frozen=True, slots=True)
class LoadModelRunResult:
    model_run_id: str
    run_dir: str
    model_artifact: object
    manifest: dict[str, Any]
    metrics: dict[str, MetricValue]
    predictions: object | None = None

    @property
    def predict_metadata(self) -> Mapping[str, Any] | None:
        if isinstance(self.model_artifact, Mapping):
            raw = self.model_artifact.get("predict_metadata")
            if isinstance(raw, Mapping):
                return raw
        return None


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

        model_id = _safe_str(payload.get("model_id", "")).strip()
        ticker = _safe_str(payload.get("ticker", "")).strip()

        return cls(
            model_id=model_id,
            ticker=ticker,
            horizon_days=_safe_int(payload.get("horizon_days", 0), default=0),
            predictions=predictions,
            generated_at=_optional_str(payload.get("generated_at")),
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

        model_id_raw = payload.get("model_id", "")
        model_id = "" if model_id_raw in (None, "") else str(model_id_raw)

        return cls(
            model_id=model_id,
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
    "LoadModelRunRequest",
    "LoadModelRunResult",
    "MetricValue",
    "TrainModelRequest",
    "TrainModelResult",
]

