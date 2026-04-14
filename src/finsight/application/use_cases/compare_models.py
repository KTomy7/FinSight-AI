from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import finsight.application.dto as application_dto
from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_RMSE
from finsight.domain.ports import ModelRegistryPort


_DEFAULT_DIRECTION_BY_METRIC = {
    METRIC_MAE: "asc",
    METRIC_RMSE: "asc",
    METRIC_DIRECTION_ACCURACY: "desc",
}


def _require_non_empty_text(value: object, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be a non-empty string.")

    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_model_ids(model_ids: Sequence[str]) -> list[str]:
    normalized = [_require_non_empty_text(model_id, field_name="model_ids item") for model_id in model_ids]
    if not normalized:
        raise ValueError("model_ids must contain at least one model id.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("model_ids must be unique.")
    return normalized


def _normalize_rank_by(rank_by: Sequence[str]) -> list[str]:
    normalized = [_require_non_empty_text(metric_name, field_name="rank_by item") for metric_name in rank_by]
    if not normalized:
        raise ValueError("rank_by must contain at least one metric name.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("rank_by must not contain duplicate metric names.")
    return normalized


def _normalize_metric_directions(metric_directions: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for metric_name, direction in metric_directions.items():
        metric_key = _require_non_empty_text(metric_name, field_name="metric_directions key")
        direction_key = _require_non_empty_text(direction, field_name=f"metric_directions['{metric_key}']").lower()
        if direction_key not in {"asc", "desc"}:
            raise ValueError(f"metric_directions['{metric_key}'] must be 'asc' or 'desc'.")
        normalized[metric_key] = direction_key
    return normalized


def _resolve_direction(metric_name: str, metric_directions: Mapping[str, str]) -> str:
    direction = metric_directions.get(metric_name, _DEFAULT_DIRECTION_BY_METRIC.get(metric_name, "asc"))
    if direction not in {"asc", "desc"}:
        raise ValueError(f"Invalid sort direction '{direction}' for metric '{metric_name}'.")
    return direction


def _coerce_metric_value(value: Any, *, metric_name: str, model_id: str) -> float:
    try:
        metric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric '{metric_name}' for model '{model_id}' must be numeric.") from exc

    if not math.isfinite(metric_value):
        raise ValueError(f"Metric '{metric_name}' for model '{model_id}' must be finite.")

    return metric_value


class CompareModels:
    def __init__(self, *, model_registry: ModelRegistryPort) -> None:
        self._model_registry = model_registry

    def execute(self, request: application_dto.CompareModelsRequest) -> application_dto.CompareModelsResult:
        model_ids = _normalize_model_ids(request.model_ids)
        rank_by = _normalize_rank_by(request.rank_by)
        metric_directions = _normalize_metric_directions(request.metric_directions)

        rows: list[application_dto.ModelComparisonRow] = []
        for model_id in model_ids:
            run_id = self._model_registry.latest_run_id(artifact_root=request.artifacts_dir, model_id=model_id)
            run_artifacts = self._model_registry.load_run_artifacts(artifact_root=request.artifacts_dir, run_id=run_id)

            metrics_raw = getattr(run_artifacts, "metrics", None)
            if not isinstance(metrics_raw, Mapping):
                raise TypeError(f"Loaded artifacts for model '{model_id}' must expose metrics as a mapping.")

            row_metrics: dict[str, application_dto.MetricValue] = {
                str(metric_name): metric_value for metric_name, metric_value in metrics_raw.items()
            }

            sort_key: list[float | str] = []
            for metric_name in rank_by:
                if metric_name not in row_metrics:
                    raise ValueError(f"Model '{model_id}' is missing comparison metric '{metric_name}'.")

                metric_value = _coerce_metric_value(row_metrics[metric_name], metric_name=metric_name, model_id=model_id)
                direction = _resolve_direction(metric_name, metric_directions)
                sort_key.append(metric_value if direction == "asc" else -metric_value)

            sort_key.append(model_id)
            sort_key.append(str(run_id))

            rows.append(
                application_dto.ModelComparisonRow(
                    rank=0,
                    model_id=model_id,
                    run_id=str(run_id),
                    metrics=row_metrics,
                    sort_key=tuple(sort_key),
                )
            )

        ordered_rows = sorted(rows, key=lambda row: row.sort_key)
        ranked_rows = [replace(row, rank=index + 1) for index, row in enumerate(ordered_rows)]

        return application_dto.CompareModelsResult(
            rows=ranked_rows,
            rank_by=rank_by,
            metric_directions={metric_name: _resolve_direction(metric_name, metric_directions) for metric_name in rank_by},
        )


__all__ = ["CompareModels"]


