import math

import pytest

from finsight.domain.metrics import (
    METRIC_DIRECTION_ACCURACY,
    METRIC_MAE,
    METRIC_RMSE,
    SUPPORTED_METRIC_NAMES,
    direction_accuracy,
    forecast_metrics,
    mean_absolute_error,
    root_mean_squared_error,
)


def test_mean_absolute_error() -> None:
    y_true = [0.10, -0.20, 0.30]
    y_pred = [0.00, -0.10, 0.20]

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.10)


def test_root_mean_squared_error() -> None:
    y_true = [0.10, -0.20, 0.30]
    y_pred = [0.00, -0.10, 0.20]

    expected_rmse = math.sqrt((0.01 + 0.01 + 0.01) / 3)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(expected_rmse)


def test_direction_accuracy_with_default_threshold() -> None:
    y_true = [0.10, -0.20, 0.00, 0.30]
    y_pred = [0.05, 0.10, -0.01, -0.20]

    # Matches on items 0 and 2 when using the existing > 0 directional convention.
    assert direction_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_direction_accuracy_allows_custom_threshold() -> None:
    y_true = [0.01, 0.03, 0.00]
    y_pred = [0.04, -0.01, 0.00]

    assert direction_accuracy(y_true, y_pred, positive_threshold=0.02) == pytest.approx(1 / 3)


def test_forecast_metrics_returns_canonical_metric_keys() -> None:
    y_true = [0.10, -0.20, 0.30]
    y_pred = [0.00, -0.10, 0.20]

    metrics = forecast_metrics(y_true, y_pred)

    assert tuple(metrics.keys()) == SUPPORTED_METRIC_NAMES
    assert metrics[METRIC_MAE] == pytest.approx(0.10)
    assert metrics[METRIC_RMSE] == pytest.approx(math.sqrt((0.01 + 0.01 + 0.01) / 3))
    assert metrics[METRIC_DIRECTION_ACCURACY] == pytest.approx(2 / 3)


def test_forecast_metrics_allows_custom_direction_threshold() -> None:
    y_true = [0.01, 0.03, 0.00]
    y_pred = [0.04, -0.01, 0.00]

    metrics = forecast_metrics(y_true, y_pred, positive_threshold=0.02)

    assert metrics[METRIC_DIRECTION_ACCURACY] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([], []),
        ([0.1], []),
        ([], [0.1]),
    ],
)
def test_metric_functions_reject_empty_inputs(y_true: list[float], y_pred: list[float]) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mean_absolute_error(y_true, y_pred)

    with pytest.raises(ValueError, match="non-empty"):
        root_mean_squared_error(y_true, y_pred)

    with pytest.raises(ValueError, match="non-empty"):
        direction_accuracy(y_true, y_pred)

    with pytest.raises(ValueError, match="non-empty"):
        forecast_metrics(y_true, y_pred)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([0.1], [0.1, 0.2]),
        ([0.1, 0.2], [0.1]),
    ],
)
def test_metric_functions_reject_length_mismatch(y_true: list[float], y_pred: list[float]) -> None:
    with pytest.raises(ValueError, match="same length"):
        mean_absolute_error(y_true, y_pred)

    with pytest.raises(ValueError, match="same length"):
        root_mean_squared_error(y_true, y_pred)

    with pytest.raises(ValueError, match="same length"):
        direction_accuracy(y_true, y_pred)

    with pytest.raises(ValueError, match="same length"):
        forecast_metrics(y_true, y_pred)


