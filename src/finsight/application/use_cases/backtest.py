from __future__ import annotations

from datetime import date, timedelta
from math import isfinite
from typing import Mapping

import pandas as pd

import finsight.application.dto as application_dto
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.domain.ports import FeatureStorePort, ModelPort
from finsight.infrastructure.features import WalkForwardSplitPolicy

TARGET_COLUMN = "target_ret_1d"


def _parse_iso_date(iso_str: str) -> date:
    try:
        return date.fromisoformat(iso_str)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO 8601 date for 'end': {iso_str!r}") from exc


def _validate_model_ids(model_ids: list[str]) -> list[str]:
    if not model_ids:
        raise ValueError("model_ids must contain at least one model id.")

    normalized = [str(model_id).strip() for model_id in model_ids]
    if any(not model_id for model_id in normalized):
        raise ValueError("model_ids must not contain empty values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("model_ids must be unique.")
    return normalized


def _validate_supported_model_ids(model_ids: list[str], supported_model_ids: tuple[str, ...]) -> None:
    unsupported = [model_id for model_id in model_ids if model_id not in supported_model_ids]
    if unsupported:
        raise ValueError(
            f"Unsupported model id(s): {unsupported}. Supported model ids: {supported_model_ids}."
        )


def _get_training_tickers(training_tickers: tuple[str, ...] | list[str]) -> list[str]:
    tickers = [ticker for ticker in (str(raw).strip().upper() for raw in training_tickers) if ticker]
    if not tickers:
        raise ValueError("Configured training tickers must contain at least one symbol.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Configured training tickers must not contain duplicates.")
    return tickers


def _coerce_numeric_metrics(
    metrics: Mapping[str, application_dto.MetricValue],
    *,
    model_id: str,
    fold_index: int,
) -> dict[str, float]:
    coerced: dict[str, float] = {}
    for metric_name, metric_value in metrics.items():
        try:
            numeric_value = float(metric_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metric '{metric_name}' for model '{model_id}' in fold {fold_index} must be numeric."
            ) from exc
        if not isfinite(numeric_value):
            raise ValueError(
                f"Metric '{metric_name}' for model '{model_id}' in fold {fold_index} must be finite."
            )
        coerced[str(metric_name)] = numeric_value
    return coerced


def _aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, application_dto.MetricValue]:
    if not fold_metrics:
        raise ValueError("fold_metrics must contain at least one fold metric set.")

    metric_names = sorted({metric_name for metrics in fold_metrics for metric_name in metrics.keys()})
    aggregated: dict[str, application_dto.MetricValue] = {"fold_count": len(fold_metrics)}
    for metric_name in metric_names:
        values = [metrics[metric_name] for metrics in fold_metrics if metric_name in metrics]
        aggregated[metric_name] = float(sum(values) / len(values))
    return aggregated


class Backtest:
    def __init__(
        self,
        *,
        fetch_market_data: FetchMarketData,
        feature_store: FeatureStorePort,
        model: ModelPort,
        training_tickers: tuple[str, ...] | list[str],
        supported_model_ids: tuple[str, ...] | list[str] | None = None,
        default_interval: str = "1d",
    ) -> None:
        model_supported_model_ids = self._as_tuple(model.supported_model_types())
        configured_supported_model_ids = (
            model_supported_model_ids
            if supported_model_ids is None
            else self._as_tuple(supported_model_ids)
        )
        _validate_model_ids(list(configured_supported_model_ids))
        _validate_supported_model_ids(list(configured_supported_model_ids), model_supported_model_ids)

        self._fetch_market_data = fetch_market_data
        self._feature_store = feature_store
        self._model = model
        self._training_tickers = tuple(training_tickers)
        self._supported_model_ids = configured_supported_model_ids
        self._default_interval = default_interval

    def execute(self, request: application_dto.BacktestRequest) -> application_dto.BacktestReport:
        if request.years <= 0:
            raise ValueError("years must be a positive integer.")

        tickers = _get_training_tickers(self._training_tickers)
        model_ids = _validate_model_ids(request.model_ids)
        _validate_supported_model_ids(model_ids, self._supported_model_ids)

        resolved_interval = request.interval or self._default_interval
        end_date = _parse_iso_date(request.end) if request.end else date.today()
        start_date = end_date - timedelta(days=(request.years * 365) - 1)

        series_list = []
        for ticker in tickers:
            result = self._fetch_market_data.execute(
                application_dto.FetchMarketDataRequest(
                    ticker=ticker,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    interval=resolved_interval,
                    include_summary=False,
                )
            )
            series_list.append(result.history)

        feature_dataset = self._feature_store.build_feature_dataset(series_list)
        if not isinstance(feature_dataset, pd.DataFrame):
            raise TypeError("Feature store must return a pandas DataFrame for backtesting.")

        split_policy = WalkForwardSplitPolicy(
            min_train_size=request.min_train_days,
            test_size=request.test_window_days,
            step_size=request.step_days,
            date_col="date",
            max_folds=request.max_folds,
        )
        folds = split_policy.split_frame(feature_dataset)

        results: list[application_dto.BacktestResult] = []
        for model_id in model_ids:
            fold_rows: list[application_dto.SerializableRow] = []
            fold_metrics: list[dict[str, float]] = []

            for fold in folds:
                evaluation_result = self._model.evaluate(
                    train_dataset=fold.train_df,
                    test_dataset=fold.test_df,
                    model_type=model_id,
                    target_column=TARGET_COLUMN,
                )
                metrics = _coerce_numeric_metrics(
                    evaluation_result.metrics,
                    model_id=model_id,
                    fold_index=fold.fold_index,
                )
                fold_metrics.append(metrics)

                fold_rows.append(
                    application_dto.BacktestFoldSummary(
                        fold_index=fold.fold_index,
                        train_start=fold.train_start.isoformat(),
                        train_end=fold.train_end.isoformat(),
                        test_start=fold.test_start.isoformat(),
                        test_end=fold.test_end.isoformat(),
                        n_train=len(fold.train_df),
                        n_test=len(fold.test_df),
                        metrics=metrics,
                    ).to_dict()
                )

            results.append(
                application_dto.BacktestResult(
                    model_id=model_id,
                    metrics=_aggregate_fold_metrics(fold_metrics),
                    folds=fold_rows,
                )
            )

        return application_dto.BacktestReport(
            results=results,
            dataset_spec=application_dto.DatasetSpec(
                tickers=tuple(tickers),
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                interval=resolved_interval,
            ),
            split_spec={
                "name": "walk_forward",
                "min_train_days": request.min_train_days,
                "test_window_days": request.test_window_days,
                "step_days": request.step_days,
                "max_folds": request.max_folds,
                "fold_count": len(folds),
            },
        )

    @staticmethod
    def _as_tuple(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(values)


__all__ = ["Backtest"]


