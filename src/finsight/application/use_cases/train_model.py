from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from finsight.application.use_cases.fetch_market_data import FetchMarketData, FetchMarketDataRequest
from finsight.domain.ports import FeatureStorePort, ModelPort, ModelRegistryPort

TARGET_COLUMN = "target_ret_1d"


@dataclass(frozen=True, slots=True)
class TrainModelRequest:
    cutoff_date: str
    years: int = 2
    end: str | None = None
    interval: str | None = None
    model_types: list[str] = field(default_factory=lambda: ["naive_zero", "naive_mean"])
    artifacts_dir: str = "artifacts/runs"


@dataclass(frozen=True, slots=True)
class TrainModelResponse:
    run_dirs: dict[str, str]
    metrics: dict[str, dict[str, float | int | str]]


def _validate_model_types(model_types: list[str]) -> list[str]:
    if not model_types:
        raise ValueError("model_types must contain at least one model type.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for model_type in model_types:
        if model_type in seen and model_type not in duplicates:
            duplicates.append(model_type)
        seen.add(model_type)

    if duplicates:
        raise ValueError(f"model_types must be unique. Duplicate values: {duplicates}.")

    return model_types


def _validate_supported_model_types(model_types: list[str], supported_model_types: tuple[str, ...]) -> None:
    unsupported = [model_type for model_type in model_types if model_type not in supported_model_types]
    if unsupported:
        raise ValueError(
            f"Unsupported model type(s): {unsupported}. Supported model types: {supported_model_types}."
        )


def _parse_iso_date(iso_str: str) -> date:
    try:
        return date.fromisoformat(iso_str)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO 8601 date for 'end': {iso_str!r}") from exc


def _get_training_tickers(training_tickers: tuple[str, ...] | list[str]) -> list[str]:
    tickers = [ticker for ticker in (str(raw).strip().upper() for raw in training_tickers) if ticker]
    if not tickers:
        raise ValueError("Configured training tickers must contain at least one symbol.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Configured training tickers must not contain duplicates.")
    return tickers



class TrainModel:
    def __init__(
        self,
        fetch_market_data: FetchMarketData,
        feature_store: FeatureStorePort,
        model: ModelPort,
        model_registry: ModelRegistryPort,
        training_tickers: tuple[str, ...] | list[str],
        supported_model_types: tuple[str, ...] | list[str] | None = None,
        default_interval: str = "1d",
    ) -> None:
        self._fetch_market_data = fetch_market_data
        self._feature_store = feature_store
        self._model = model
        self._model_registry = model_registry
        self._training_tickers = tuple(training_tickers)
        self._supported_model_types = tuple(supported_model_types) if supported_model_types is not None else None
        self._default_interval = default_interval

    def execute(self, request: TrainModelRequest) -> TrainModelResponse:
        if request.years <= 0:
            raise ValueError("years must be a positive integer.")

        tickers = _get_training_tickers(self._training_tickers)
        model_types = _validate_model_types(request.model_types)
        supported_model_types = self._supported_model_types or self._model.supported_model_types()
        _validate_supported_model_types(model_types, supported_model_types)
        resolved_interval = request.interval or self._default_interval

        end_date = _parse_iso_date(request.end) if request.end else date.today()
        lookback_days = (request.years * 365) - 1
        start_date = end_date - timedelta(days=lookback_days)

        series_list = []
        for ticker in tickers:
            result = self._fetch_market_data.execute(
                FetchMarketDataRequest(
                    ticker=ticker,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    interval=resolved_interval,
                    include_summary=False,
                )
            )
            series_list.append(result.history)

        feature_dataset = self._feature_store.build_feature_dataset(series_list)
        train_dataset, test_dataset = self._feature_store.split_train_test(
            feature_dataset,
            cutoff_date=request.cutoff_date,
            date_col="date",
            inclusive_test=True,
        )

        model_metrics: dict[str, dict[str, float]] = {}
        predictions_by_model: dict[str, object] = {}
        for model_type in model_types:
            metric_values, predictions = self._model.evaluate(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                model_type=model_type,
                target_column=TARGET_COLUMN,
            )
            model_metrics[model_type] = dict(metric_values)
            predictions_by_model[model_type] = predictions

        train_min_date, train_max_date = self._feature_store.frame_date_range(train_dataset)
        test_min_date, test_max_date = self._feature_store.frame_date_range(test_dataset)
        n_train = self._feature_store.row_count(train_dataset)
        n_test = self._feature_store.row_count(test_dataset)
        feature_columns = list(self._feature_store.feature_columns())

        created_at = datetime.now(timezone.utc).replace(microsecond=0)
        created_at_utc = created_at.isoformat().replace("+00:00", "Z")
        run_prefix = created_at.strftime("%Y-%m-%dT%H%M%SZ")

        run_dirs: dict[str, str] = {}
        metrics: dict[str, dict[str, float | int | str]] = {}

        for model_type in model_types:
            run_id = f"{run_prefix}__{model_type}"
            run_dir = self._model_registry.create_run_dir(
                artifact_root=request.artifacts_dir,
                run_id=run_id,
            )
            run_dir_name = Path(run_dir).name

            enriched_metrics: dict[str, float | int | str] = {
                **model_metrics[model_type],
                "n_train": n_train,
                "n_test": n_test,
                "train_min_date": train_min_date,
                "train_max_date": train_max_date,
                "test_min_date": test_min_date,
                "test_max_date": test_max_date,
            }

            metadata = {
                "run_id": run_dir_name,
                "created_at_utc": created_at_utc,
                "model_type": model_type,
                "tickers": tickers,
                "interval": resolved_interval,
                "years": int(request.years),
                "end": request.end or end_date.isoformat(),
                "cutoff_date": request.cutoff_date,
                "horizon": "1d",
                "feature_columns": feature_columns,
                "target_column": TARGET_COLUMN,
                "n_train": n_train,
                "n_test": n_test,
                "train_min_date": train_min_date,
                "train_max_date": train_max_date,
                "test_min_date": test_min_date,
                "test_max_date": test_max_date,
            }

            self._model_registry.save_metrics(
                run_dir=run_dir,
                metrics=enriched_metrics,
            )
            self._model_registry.save_metadata(
                run_dir=run_dir,
                metadata=metadata,
            )
            self._model_registry.save_predictions(
                run_dir=run_dir,
                predictions=predictions_by_model[model_type],
            )

            run_dirs[model_type] = run_dir
            metrics[model_type] = enriched_metrics

        return TrainModelResponse(run_dirs=run_dirs, metrics=metrics)


