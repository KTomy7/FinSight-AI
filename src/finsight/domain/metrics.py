from __future__ import annotations

import math
from collections.abc import Sequence

METRIC_MAE = "mae"
METRIC_RMSE = "rmse"
METRIC_DIRECTION_ACCURACY = "direction_accuracy"

SUPPORTED_METRIC_NAMES = (
    METRIC_MAE,
    METRIC_RMSE,
    METRIC_DIRECTION_ACCURACY,
)


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_values, pred_values = _coerce_and_validate_inputs(y_true, y_pred)
    return float(sum(abs(a - b) for a, b in zip(true_values, pred_values)) / len(true_values))


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_values, pred_values = _coerce_and_validate_inputs(y_true, y_pred)
    mse = sum((a - b) ** 2 for a, b in zip(true_values, pred_values)) / len(true_values)
    return float(math.sqrt(mse))


def direction_accuracy(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    positive_threshold: float = 0.0,
) -> float:
    true_values, pred_values = _coerce_and_validate_inputs(y_true, y_pred)
    matching_directions = sum(
        int((a > positive_threshold) == (b > positive_threshold))
        for a, b in zip(true_values, pred_values)
    )
    return float(matching_directions / len(true_values))


def forecast_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    positive_threshold: float = 0.0,
) -> dict[str, float]:
    true_values, pred_values = _coerce_and_validate_inputs(y_true, y_pred)

    count = len(true_values)
    abs_error_sum = 0.0
    squared_error_sum = 0.0
    matching_directions = 0
    
    for true_val, pred_val in zip(true_values, pred_values):
        error = true_val - pred_val
        abs_error_sum += abs(error)
        squared_error_sum += error ** 2
        matching_directions += int((true_val > positive_threshold) == (pred_val > positive_threshold))
    
    return {
        METRIC_MAE: float(abs_error_sum / count),
        METRIC_RMSE: float(math.sqrt(squared_error_sum / count)),
        METRIC_DIRECTION_ACCURACY: float(matching_directions / count),
    }


def _coerce_and_validate_inputs(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> tuple[list[float], list[float]]:
    true_values = [float(value) for value in y_true]
    pred_values = [float(value) for value in y_pred]

    if not true_values or not pred_values:
        raise ValueError("y_true and y_pred must be non-empty.")

    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length.")

    return true_values, pred_values
