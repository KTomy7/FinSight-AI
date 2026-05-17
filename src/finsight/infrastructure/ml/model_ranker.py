from __future__ import annotations

import math
from typing import Any, Mapping

from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_RMSE


# Default metric directions: asc = lower is better, desc = higher is better
_DEFAULT_DIRECTION_BY_METRIC = {
    METRIC_MAE: "asc",
    METRIC_RMSE: "asc",
    METRIC_DIRECTION_ACCURACY: "desc",
}


class ModelRanker:
    """
    Centralized ranking engine for model comparison.

    Encapsulates the logic for:
    - Determining metric direction (ascending/descending)
    - Comparing metrics between runs
    - Computing sort keys for ranking
    """

    def __init__(self, *, rank_by: list[str], metric_directions: dict[str, str] | None = None) -> None:
        """
        Initialize the ranker.

        Args:
            rank_by: List of metric names in priority order for ranking.
            metric_directions: Dict mapping metric name to 'asc' or 'desc'.
                              If not provided, uses defaults from domain.metrics.
        """
        self.rank_by = self._normalize_rank_by(rank_by)
        self.metric_directions = self._normalize_metric_directions(metric_directions or {})

    @staticmethod
    def _normalize_rank_by(rank_by: list[str]) -> list[str]:
        """Validate and normalize rank_by list."""
        if not rank_by:
            raise ValueError("rank_by must contain at least one metric name.")

        normalized = [str(m).strip() for m in rank_by]
        if not all(normalized):
            raise ValueError("rank_by items must be non-empty strings.")

        if len(set(normalized)) != len(normalized):
            raise ValueError("rank_by must not contain duplicate metric names.")

        return normalized

    @staticmethod
    def _normalize_metric_directions(metric_directions: dict[str, str]) -> dict[str, str]:
        """Validate and normalize metric_directions dict."""
        normalized: dict[str, str] = {}
        for metric_name, direction in metric_directions.items():
            metric_key = str(metric_name).strip()
            direction_key = str(direction).strip().lower()

            if not metric_key:
                raise ValueError("metric_directions keys must be non-empty strings.")
            if direction_key not in {"asc", "desc"}:
                raise ValueError(f"metric_directions['{metric_key}'] must be 'asc' or 'desc'.")

            normalized[metric_key] = direction_key

        return normalized

    def get_direction(self, metric_name: str) -> str:
        """
        Get the direction (asc or desc) for a metric.

        Uses explicit metric_directions if provided, otherwise falls back to defaults.
        """
        direction = self.metric_directions.get(metric_name, _DEFAULT_DIRECTION_BY_METRIC.get(metric_name, "asc"))
        if direction not in {"asc", "desc"}:
            raise ValueError(f"Invalid sort direction '{direction}' for metric '{metric_name}'.")
        return direction

    def coerce_metric_value(self, value: Any, *, metric_name: str, model_id: str) -> float:
        """
        Convert a metric value to float, validating that it is numeric and finite.

        Raises ValueError if the value is not numeric or not finite.
        """
        try:
            metric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric '{metric_name}' for model '{model_id}' must be numeric.") from exc

        if not math.isfinite(metric_value):
            raise ValueError(f"Metric '{metric_name}' for model '{model_id}' must be finite.")

        return metric_value

    def is_better(self, new_metrics: dict[str, float], current_metrics: dict[str, float]) -> bool:
        """
        Compare two metric dictionaries and return True if new is better than current.

        Uses the rank_by order and metric_directions to determine "better".
        Returns False if any metric is missing in either dict.
        """
        # Build sort keys for both
        new_key = self._compute_raw_sort_key(new_metrics)
        current_key = self._compute_raw_sort_key(current_metrics)

        if new_key is None or current_key is None:
            return False

        return new_key < current_key

    def _compute_raw_sort_key(self, metrics: dict[str, float]) -> tuple[float, ...] | None:
        """
        Compute the normalized sort key tuple for a metrics dict.

        Returns None if any required metric is missing.
        """
        sort_key: list[float] = []
        for metric_name in self.rank_by:
            if metric_name not in metrics:
                return None

            metric_value = metrics[metric_name]
            direction = self.get_direction(metric_name)

            # Normalize: asc stays positive, desc gets negated so we can sort ascending
            normalized = metric_value if direction == "asc" else -metric_value
            sort_key.append(normalized)

        return tuple(sort_key)

    def compute_sort_key(self, metrics: dict[str, float], model_id: str, run_id: str) -> tuple:
        """
        Compute a full sort key tuple for ranking rows.

        Includes metrics (normalized by direction) plus tiebreakers (model_id, run_id).
        Used by CompareModels for final row ranking.
        """
        sort_key: list[float | str] = []

        # Add normalized metrics in rank_by order
        for metric_name in self.rank_by:
            if metric_name not in metrics:
                raise ValueError(f"Model '{model_id}' is missing comparison metric '{metric_name}'.")

            metric_value = self.coerce_metric_value(metrics[metric_name], metric_name=metric_name, model_id=model_id)
            direction = self.get_direction(metric_name)

            # Normalize: asc stays positive, desc gets negated
            normalized = metric_value if direction == "asc" else -metric_value
            sort_key.append(normalized)

        # Tiebreakers: model_id, then run_id
        sort_key.append(model_id)
        sort_key.append(str(run_id))

        return tuple(sort_key)


__all__ = ["ModelRanker"]

