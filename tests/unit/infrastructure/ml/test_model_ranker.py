from __future__ import annotations

import pytest

from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_RMSE
from finsight.infrastructure.ml.model_ranker import ModelRanker


class TestModelRankerInitialization:
    """Test ModelRanker initialization and validation."""

    def test_init_with_valid_rank_by(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE, METRIC_RMSE])
        assert ranker.rank_by == [METRIC_MAE, METRIC_RMSE]

    def test_init_with_empty_rank_by_raises_error(self) -> None:
        with pytest.raises(ValueError, match="rank_by must contain at least one metric"):
            ModelRanker(rank_by=[])

    def test_init_with_duplicate_metrics_in_rank_by_raises_error(self) -> None:
        with pytest.raises(ValueError, match="must not contain duplicate metric"):
            ModelRanker(rank_by=[METRIC_MAE, METRIC_MAE])

    def test_init_with_valid_metric_directions(self) -> None:
        ranker = ModelRanker(
            rank_by=[METRIC_MAE],
            metric_directions={METRIC_MAE: "asc"},
        )
        assert ranker.metric_directions == {METRIC_MAE: "asc"}

    def test_init_with_invalid_direction_raises_error(self) -> None:
        with pytest.raises(ValueError, match="must be 'asc' or 'desc'"):
            ModelRanker(
                rank_by=[METRIC_MAE],
                metric_directions={METRIC_MAE: "invalid"},
            )

    def test_init_with_none_metric_directions_defaults_to_empty(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE], metric_directions=None)
        assert ranker.metric_directions == {}


class TestGetDirection:
    """Test get_direction method."""

    def test_get_direction_returns_explicit_direction(self) -> None:
        ranker = ModelRanker(
            rank_by=[METRIC_MAE],
            metric_directions={METRIC_MAE: "desc"},
        )
        assert ranker.get_direction(METRIC_MAE) == "desc"

    def test_get_direction_returns_default_for_mae(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        assert ranker.get_direction(METRIC_MAE) == "asc"

    def test_get_direction_returns_default_for_rmse(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_RMSE])
        assert ranker.get_direction(METRIC_RMSE) == "asc"

    def test_get_direction_returns_default_for_direction_accuracy(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_DIRECTION_ACCURACY])
        assert ranker.get_direction(METRIC_DIRECTION_ACCURACY) == "desc"

    def test_get_direction_defaults_to_asc_for_unknown_metric(self) -> None:
        ranker = ModelRanker(rank_by=["unknown_metric"])
        assert ranker.get_direction("unknown_metric") == "asc"


class TestCoerceMetricValue:
    """Test coerce_metric_value method."""

    def test_coerce_numeric_string_to_float(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        value = ranker.coerce_metric_value("0.123", metric_name=METRIC_MAE, model_id="test_model")
        assert value == 0.123

    def test_coerce_int_to_float(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        value = ranker.coerce_metric_value(5, metric_name=METRIC_MAE, model_id="test_model")
        assert value == 5.0

    def test_coerce_non_numeric_raises_error(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        with pytest.raises(ValueError, match="must be numeric"):
            ranker.coerce_metric_value("not_a_number", metric_name=METRIC_MAE, model_id="test_model")

    def test_coerce_nan_raises_error(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        with pytest.raises(ValueError, match="must be finite"):
            ranker.coerce_metric_value(float("nan"), metric_name=METRIC_MAE, model_id="test_model")

    def test_coerce_inf_raises_error(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        with pytest.raises(ValueError, match="must be finite"):
            ranker.coerce_metric_value(float("inf"), metric_name=METRIC_MAE, model_id="test_model")


class TestIsBetter:
    """Test is_better method for comparing metric dictionaries."""

    def test_is_better_with_lower_mae(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        new_metrics = {METRIC_MAE: 0.10}
        current_metrics = {METRIC_MAE: 0.15}
        assert ranker.is_better(new_metrics, current_metrics) is True

    def test_is_better_with_higher_mae_is_false(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        new_metrics = {METRIC_MAE: 0.20}
        current_metrics = {METRIC_MAE: 0.15}
        assert ranker.is_better(new_metrics, current_metrics) is False

    def test_is_better_with_higher_direction_accuracy(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_DIRECTION_ACCURACY])
        new_metrics = {METRIC_DIRECTION_ACCURACY: 0.90}
        current_metrics = {METRIC_DIRECTION_ACCURACY: 0.80}
        assert ranker.is_better(new_metrics, current_metrics) is True

    def test_is_better_with_lower_direction_accuracy_is_false(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_DIRECTION_ACCURACY])
        new_metrics = {METRIC_DIRECTION_ACCURACY: 0.70}
        current_metrics = {METRIC_DIRECTION_ACCURACY: 0.80}
        assert ranker.is_better(new_metrics, current_metrics) is False

    def test_is_better_with_multiple_metrics_prioritizes_first(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE, METRIC_RMSE])
        # Same MAE, but RMSE is better in new_metrics
        new_metrics = {METRIC_MAE: 0.10, METRIC_RMSE: 0.15}
        current_metrics = {METRIC_MAE: 0.10, METRIC_RMSE: 0.20}
        assert ranker.is_better(new_metrics, current_metrics) is True

    def test_is_better_with_multiple_metrics_first_takes_precedence(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE, METRIC_RMSE])
        # MAE is worse, even though RMSE is better
        new_metrics = {METRIC_MAE: 0.15, METRIC_RMSE: 0.10}
        current_metrics = {METRIC_MAE: 0.10, METRIC_RMSE: 0.20}
        assert ranker.is_better(new_metrics, current_metrics) is False

    def test_is_better_returns_false_if_new_metric_missing(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        new_metrics = {METRIC_RMSE: 0.10}
        current_metrics = {METRIC_MAE: 0.15}
        assert ranker.is_better(new_metrics, current_metrics) is False

    def test_is_better_returns_false_if_current_metric_missing(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        new_metrics = {METRIC_MAE: 0.10}
        current_metrics = {METRIC_RMSE: 0.15}
        assert ranker.is_better(new_metrics, current_metrics) is False


class TestComputeSortKey:
    """Test compute_sort_key method."""

    def test_compute_sort_key_builds_tuple(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        metrics = {METRIC_MAE: 0.123}
        sort_key = ranker.compute_sort_key(metrics, model_id="test_model", run_id="run_123")
        assert isinstance(sort_key, tuple)
        assert len(sort_key) == 3  # metric + model_id + run_id

    def test_compute_sort_key_includes_model_id_and_run_id_as_tiebreakers(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        metrics = {METRIC_MAE: 0.123}
        sort_key = ranker.compute_sort_key(metrics, model_id="alpha", run_id="run_456")
        assert sort_key[-2:] == ("alpha", "run_456")

    def test_compute_sort_key_normalizes_ascending_metric(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        metrics = {METRIC_MAE: 0.123}
        sort_key = ranker.compute_sort_key(metrics, model_id="test", run_id="run")
        # For ascending, metric stays positive
        assert sort_key[0] == 0.123

    def test_compute_sort_key_normalizes_descending_metric(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_DIRECTION_ACCURACY])
        metrics = {METRIC_DIRECTION_ACCURACY: 0.85}
        sort_key = ranker.compute_sort_key(metrics, model_id="test", run_id="run")
        # For descending, metric gets negated
        assert sort_key[0] == -0.85

    def test_compute_sort_key_with_multiple_metrics(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE, METRIC_RMSE])
        metrics = {METRIC_MAE: 0.10, METRIC_RMSE: 0.20}
        sort_key = ranker.compute_sort_key(metrics, model_id="test", run_id="run")
        assert sort_key[0] == 0.10  # MAE (asc)
        assert sort_key[1] == 0.20  # RMSE (asc)
        assert sort_key[2:] == ("test", "run")  # tiebreakers

    def test_compute_sort_key_raises_on_missing_metric(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        metrics = {METRIC_RMSE: 0.10}
        with pytest.raises(ValueError, match="missing comparison metric"):
            ranker.compute_sort_key(metrics, model_id="test", run_id="run")

    def test_compute_sort_key_coerces_numeric_strings(self) -> None:
        ranker = ModelRanker(rank_by=[METRIC_MAE])
        metrics = {METRIC_MAE: "0.123"}
        sort_key = ranker.compute_sort_key(metrics, model_id="test", run_id="run")
        assert sort_key[0] == 0.123


class TestMultiMetricRanking:
    """Test ranking with multiple metrics and mixed directions."""

    def test_rank_models_by_mae_then_direction_accuracy(self) -> None:
        """Test that MAE takes priority, then direction_accuracy."""
        ranker = ModelRanker(rank_by=[METRIC_MAE, METRIC_DIRECTION_ACCURACY])

        # Model alpha: MAE 0.10, direction_accuracy 0.70
        alpha_metrics = {METRIC_MAE: 0.10, METRIC_DIRECTION_ACCURACY: 0.70}
        # Model beta: MAE 0.15, direction_accuracy 0.90
        beta_metrics = {METRIC_MAE: 0.15, METRIC_DIRECTION_ACCURACY: 0.90}

        # Alpha should be better because MAE is 0.10 < 0.15
        assert ranker.is_better(alpha_metrics, beta_metrics) is True
        assert ranker.is_better(beta_metrics, alpha_metrics) is False

    def test_rank_models_tie_on_mae_breaks_on_direction_accuracy(self) -> None:
        """When MAE is equal, direction_accuracy decides."""
        ranker = ModelRanker(rank_by=[METRIC_MAE, METRIC_DIRECTION_ACCURACY])

        # Same MAE
        new = {METRIC_MAE: 0.10, METRIC_DIRECTION_ACCURACY: 0.85}
        current = {METRIC_MAE: 0.10, METRIC_DIRECTION_ACCURACY: 0.75}

        # New is better because direction_accuracy is higher (0.85 > 0.75)
        assert ranker.is_better(new, current) is True

