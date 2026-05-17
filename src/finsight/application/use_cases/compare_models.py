from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import finsight.application.dto as application_dto
from finsight.domain.ports import ModelRegistryPort
from finsight.domain.ports import RunRegistryPort
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
    def __init__(self, *, model_registry: ModelRegistryPort, run_registry: RunRegistryPort | None = None) -> None:
        self._model_registry = model_registry
        self._run_registry = run_registry

    def execute(self, request: application_dto.CompareModelsRequest) -> application_dto.CompareModelsResult:
        model_ids = _normalize_model_ids(request.model_ids)
        use_best_runs = bool(getattr(request, "use_best_runs", False))

        # Create a ranker instance with the request's ranking preferences
        ranker = ModelRanker(
            rank_by=request.rank_by,
            metric_directions=request.metric_directions,
        )

        registry_snapshot = None
        if use_best_runs and self._run_registry is not None:
            registry_snapshot = self._run_registry.load_registry(artifact_root=request.artifacts_dir)

        best_by_model: Mapping[str, object] = {}
        if registry_snapshot is not None:
            best_by_model = getattr(registry_snapshot, "best_by_model", {}) or {}

        rows: list[application_dto.ModelComparisonRow] = []
        for model_id in model_ids:
            run_id = self._resolve_run_id(
                artifact_root=request.artifacts_dir,
                model_id=model_id,
                use_best_runs=use_best_runs,
                best_by_model=best_by_model,
            )
            run_artifacts = self._model_registry.load_run_artifacts(artifact_root=request.artifacts_dir, run_id=run_id)

            metrics_raw = getattr(run_artifacts, "metrics", None)
            if not isinstance(metrics_raw, Mapping):
                raise TypeError(f"Loaded artifacts for model '{model_id}' must expose metrics as a mapping.")

            row_metrics: dict[str, application_dto.MetricValue] = {
                str(metric_name): metric_value for metric_name, metric_value in metrics_raw.items()
            }

            # Use ranker to compute the sort key
            ranking_metrics: dict[str, float] = {}
            for metric_name in ranker.rank_by:
                if metric_name not in row_metrics:
                    raise ValueError(f"Model '{model_id}' is missing comparison metric '{metric_name}'.")
                ranking_metrics[metric_name] = ranker.coerce_metric_value(
                    row_metrics[metric_name],
                    metric_name=metric_name,
                    model_id=model_id,
                )
            sort_key = ranker.compute_sort_key(ranking_metrics, model_id=model_id, run_id=str(run_id))

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

    def _resolve_run_id(
        self,
        *,
        artifact_root: str,
        model_id: str,
        use_best_runs: bool,
        best_by_model: Mapping[str, object],
    ) -> str:
        if use_best_runs:
            best_entry = best_by_model.get(model_id)
            if isinstance(best_entry, Mapping):
                best_run_id = best_entry.get("run_id")
                if best_run_id:
                    return str(best_run_id)

        return self._model_registry.latest_run_id(artifact_root=artifact_root, model_id=model_id)


__all__ = ["CompareModels"]


