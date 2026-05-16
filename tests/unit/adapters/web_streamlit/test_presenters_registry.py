"""
Unit tests for registry-aware presenter functionality.
"""
from __future__ import annotations

import pytest
import pandas as pd

import finsight.application.dto as application_dto
from finsight.adapters.web_streamlit.presenters import ComparisonPresenter


class TestComparisonPresenterWithRegistry:
    """Tests for ComparisonPresenter.format_leaderboard_with_best_runs."""

    @pytest.fixture
    def sample_result(self) -> application_dto.CompareModelsResult:
        """Create a sample comparison result."""
        return application_dto.CompareModelsResult(
            rank_by=["mae", "rmse"],
            rows=[
                application_dto.ModelComparisonRow(
                    rank=1,
                    model_id="ridge",
                    run_id="2026-05-16T120000Z__ridge",
                    metrics={"mae": 0.123, "rmse": 0.45, "direction_accuracy": 0.78},
                    sort_key=(0, 0.123, 0.45, "ridge", "2026-05-16T120000Z__ridge"),
                ),
                application_dto.ModelComparisonRow(
                    rank=2,
                    model_id="xgboost",
                    run_id="2026-05-16T110000Z__xgboost",
                    metrics={"mae": 0.150, "rmse": 0.55, "direction_accuracy": 0.75},
                    sort_key=(0, 0.150, 0.55, "xgboost", "2026-05-16T110000Z__xgboost"),
                ),
            ],
        )

    @pytest.fixture
    def sample_registry(self) -> application_dto.RegistrySnapshot:
        """Create a sample registry snapshot."""
        return application_dto.RegistrySnapshot(
            updated_at="2026-05-16T12:00:00Z",
            best_by_model={
                "ridge": {
                    "run_id": "2026-05-16T120000Z__ridge",
                    "created_at": "2026-05-16T12:00:00Z",
                    "metrics": {"mae": 0.123, "rmse": 0.45},
                },
                "xgboost": None,  # No best run yet
            },
        )

    @pytest.fixture
    def label_lookup(self) -> dict[str, str]:
        """Create a label lookup."""
        return {
            "ridge": "Ridge Regression",
            "xgboost": "XGBoost",
        }

    def test_format_leaderboard_with_best_runs_enriches_rows(
        self, sample_result: application_dto.CompareModelsResult,
        sample_registry: application_dto.RegistrySnapshot,
        label_lookup: dict[str, str],
    ):
        """Test that the method enriches rows with registry metadata."""
        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            sample_result,
            label_lookup=label_lookup,
            registry_snapshot=sample_registry,
        )

        assert not frame.empty
        assert len(frame) == 2

        # Check that new columns exist
        assert "is_best_run" in frame.columns
        assert "best_run_since" in frame.columns

        # Check first row is marked as best run
        assert frame.iloc[0]["is_best_run"] == True
        assert frame.iloc[0]["best_run_since"] == "2026-05-16T12:00:00Z"

        # Check second row is not marked as best run
        assert frame.iloc[1]["is_best_run"] == False
        assert pd.isna(frame.iloc[1]["best_run_since"]) or frame.iloc[1]["best_run_since"] is None

    def test_format_leaderboard_with_best_runs_without_registry(
        self, sample_result: application_dto.CompareModelsResult,
        label_lookup: dict[str, str],
    ):
        """Test that all rows are marked as not best when registry is None."""
        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            sample_result,
            label_lookup=label_lookup,
            registry_snapshot=None,
        )

        assert not frame.empty
        assert "is_best_run" in frame.columns

        # All rows should be marked as not best
        assert (frame["is_best_run"] == False).all()

    def test_format_leaderboard_with_best_runs_empty_result(
        self, label_lookup: dict[str, str],
        sample_registry: application_dto.RegistrySnapshot,
    ):
        """Test that empty result returns empty DataFrame."""
        empty_result = application_dto.CompareModelsResult(
            rank_by=["mae"],
            rows=[],
        )

        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            empty_result,
            label_lookup=label_lookup,
            registry_snapshot=sample_registry,
        )

        assert frame.empty

    def test_format_leaderboard_with_best_runs_column_order(
        self, sample_result: application_dto.CompareModelsResult,
        sample_registry: application_dto.RegistrySnapshot,
        label_lookup: dict[str, str],
    ):
        """Test that columns are in the expected order."""
        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            sample_result,
            label_lookup=label_lookup,
            registry_snapshot=sample_registry,
        )

        expected_prefix = ["rank", "model", "model_id", "run_id", "is_best_run", "best_run_since"]
        actual_columns = list(frame.columns)

        # Check that expected columns appear in order at the beginning
        for i, col in enumerate(expected_prefix):
            assert actual_columns[i] == col, f"Column {col} not at position {i}"

    def test_format_leaderboard_with_best_runs_preserves_metrics(
        self, sample_result: application_dto.CompareModelsResult,
        sample_registry: application_dto.RegistrySnapshot,
        label_lookup: dict[str, str],
    ):
        """Test that metric columns are preserved."""
        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            sample_result,
            label_lookup=label_lookup,
            registry_snapshot=sample_registry,
        )

        # Check that all metric columns exist
        assert "mae" in frame.columns
        assert "rmse" in frame.columns
        assert "direction_accuracy" in frame.columns

    def test_format_leaderboard_with_best_runs_handles_missing_metadata(
        self, sample_result: application_dto.CompareModelsResult,
        label_lookup: dict[str, str],
    ):
        """Test that missing best-run entry is handled gracefully."""
        registry = application_dto.RegistrySnapshot(
            updated_at="2026-05-16T12:00:00Z",
            best_by_model={
                # ridge has no entry
                "ridge": None,
                # xgboost missing entirely
            },
        )

        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            sample_result,
            label_lookup=label_lookup,
            registry_snapshot=registry,
        )

        # All rows should be marked as not best
        assert (frame["is_best_run"] == False).all()

    def test_format_leaderboard_with_best_runs_matches_by_run_id(
        self, label_lookup: dict[str, str],
    ):
        """Test that best-run matching is done by run_id."""
        result = application_dto.CompareModelsResult(
            rank_by=["mae"],
            rows=[
                application_dto.ModelComparisonRow(
                    rank=1,
                    model_id="ridge",
                    run_id="run_A",
                    metrics={"mae": 0.100},
                    sort_key=(0, 0.100, "ridge", "run_A"),
                ),
                application_dto.ModelComparisonRow(
                    rank=2,
                    model_id="ridge",
                    run_id="run_B",
                    metrics={"mae": 0.120},
                    sort_key=(0, 0.120, "ridge", "run_B"),
                ),
            ],
        )

        registry = application_dto.RegistrySnapshot(
            updated_at="2026-05-16T12:00:00Z",
            best_by_model={
                "ridge": {
                    "run_id": "run_B",  # Best is run_B, not run_A
                    "created_at": "2026-05-16T11:00:00Z",
                    "metrics": {"mae": 0.120},
                },
            },
        )

        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            result,
            label_lookup=label_lookup,
            registry_snapshot=registry,
        )

        assert frame.iloc[0]["is_best_run"] == False  # run_A is not best
        assert frame.iloc[1]["is_best_run"] == True   # run_B is best

    def test_format_leaderboard_with_best_runs_handles_invalid_metadata(
        self, sample_result: application_dto.CompareModelsResult,
        label_lookup: dict[str, str],
    ):
        """Test that invalid metadata entries are skipped."""
        registry = application_dto.RegistrySnapshot(
            updated_at="2026-05-16T12:00:00Z",
            best_by_model={
                "ridge": "invalid_string_instead_of_dict",  # Invalid format
                "xgboost": {"run_id": "different_run_id"},  # Different run_id, won't match
            },
        )

        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            sample_result,
            label_lookup=label_lookup,
            registry_snapshot=registry,
        )

        # Data should still be formatted, but no matches
        assert not frame.empty
        assert (frame["is_best_run"] == False).all()


class TestComparisonPresenterBackwardCompatibility:
    """Tests to ensure backward compatibility of existing presenter method."""

    def test_format_leaderboard_frame_still_works(self):
        """Test that the original format_leaderboard_frame method still works."""
        result = application_dto.CompareModelsResult(
            rank_by=["mae"],
            rows=[
                application_dto.ModelComparisonRow(
                    rank=1,
                    model_id="ridge",
                    run_id="2026-05-16T120000Z__ridge",
                    metrics={"mae": 0.123},
                    sort_key=(0, 0.123, "ridge", "2026-05-16T120000Z__ridge"),
                ),
            ],
        )

        label_lookup = {"ridge": "Ridge Regression"}

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup=label_lookup)

        assert not frame.empty
        assert len(frame) == 1
        assert frame.iloc[0]["model"] == "Ridge Regression"

        # Should NOT have registry columns
        assert "is_best_run" not in frame.columns
        assert "best_run_since" not in frame.columns




