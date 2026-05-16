from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import finsight.application.dto as application_dto
from finsight.domain.ports import ModelRegistryPort
from finsight.infrastructure.ml.model_ranker import ModelRanker


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


class CompareModels:
    def __init__(self, *, model_registry: ModelRegistryPort) -> None:
        self._model_registry = model_registry

    def execute(self, request: application_dto.CompareModelsRequest) -> application_dto.CompareModelsResult:
        model_ids = _normalize_model_ids(request.model_ids)

        # Create a ranker instance with the request's ranking preferences
        ranker = ModelRanker(
            rank_by=request.rank_by,
            metric_directions=request.metric_directions,
        )

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

            # Use ranker to compute the sort key
            sort_key = ranker.compute_sort_key(row_metrics, model_id=model_id, run_id=str(run_id))

            rows.append(
                application_dto.ModelComparisonRow(
                    rank=0,
                    model_id=model_id,
                    run_id=str(run_id),
                    metrics=row_metrics,
                    sort_key=sort_key,
                )
            )

        ordered_rows = sorted(rows, key=lambda row: row.sort_key)
        ranked_rows = [replace(row, rank=index + 1) for index, row in enumerate(ordered_rows)]

        return application_dto.CompareModelsResult(
            rows=ranked_rows,
            rank_by=ranker.rank_by,
            metric_directions={metric_name: ranker.get_direction(metric_name) for metric_name in ranker.rank_by},
        )


__all__ = ["CompareModels"]


