from __future__ import annotations

import math
from collections.abc import Sequence

METRIC_MAE = "mae"
METRIC_RMSE = "rmse"
METRIC_R2 = "r2"
METRIC_DIRECTION_ACCURACY = "direction_accuracy"

SUPPORTED_METRIC_NAMES = (
    METRIC_MAE,
    METRIC_RMSE,
    METRIC_R2,
    METRIC_DIRECTION_ACCURACY,
)


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    count = _validate_input_lengths(y_true, y_pred)
    return float(sum(abs(float(a) - float(b)) for a, b in zip(y_true, y_pred)) / count)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    count = _validate_input_lengths(y_true, y_pred)
    mse = sum((float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred)) / count
    return float(math.sqrt(mse))


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    count = _validate_input_lengths(y_true, y_pred)

    true_values = [float(value) for value in y_true]
    pred_values = [float(value) for value in y_pred]
    mean_true = sum(true_values) / count

    ss_res = sum((true - pred) ** 2 for true, pred in zip(true_values, pred_values))
    ss_tot = sum((true - mean_true) ** 2 for true in true_values)

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1.0 - (ss_res / ss_tot))


def direction_accuracy(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    positive_threshold: float = 0.0,
) -> float:
    count = _validate_input_lengths(y_true, y_pred)
    matching_directions = sum(
        int((float(a) > positive_threshold) == (float(b) > positive_threshold))
        for a, b in zip(y_true, y_pred)
    )
    return float(matching_directions / count)


def forecast_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    positive_threshold: float = 0.0,
) -> dict[str, float]:
    count = _validate_input_lengths(y_true, y_pred)
    abs_error_sum = 0.0
    squared_error_sum = 0.0
    true_values: list[float] = []
    matching_directions = 0

    for true_raw, pred_raw in zip(y_true, y_pred):
        true_val = float(true_raw)
        pred_val = float(pred_raw)
        true_values.append(true_val)
        error = true_val - pred_val
        abs_error_sum += abs(error)
        squared_error_sum += error ** 2
        matching_directions += int((true_val > positive_threshold) == (pred_val > positive_threshold))

    return {
        METRIC_MAE: float(abs_error_sum / count),
        METRIC_RMSE: float(math.sqrt(squared_error_sum / count)),
        METRIC_R2: r2_score(true_values, y_pred),
        METRIC_DIRECTION_ACCURACY: float(matching_directions / count),
    }


def _validate_input_lengths(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> int:
    true_len = len(y_true)
    pred_len = len(y_pred)

    if true_len == 0 or pred_len == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    if true_len != pred_len:
        raise ValueError("y_true and y_pred must have the same length.")

    return true_len
