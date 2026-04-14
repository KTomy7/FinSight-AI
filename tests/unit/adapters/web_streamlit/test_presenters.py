"""Tests for presenter formatting functions."""
from __future__ import annotations

import pandas as pd

from finsight.adapters.web_streamlit.presenters import ComparisonPresenter, ForecastPresenter
from finsight.application.dto import CompareModelsResult, ForecastResult, ModelComparisonRow


class TestForecastPresenter:
    """Tests for ForecastPresenter formatting logic."""

    def test_format_predictions_table_returns_empty_dataframe_for_empty_predictions(self) -> None:
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=[],
        )

        frame = ForecastPresenter.format_predictions_table(result)

        assert isinstance(frame, pd.DataFrame)
        assert frame.empty

    def test_format_predictions_table_converts_predictions_to_dataframe(self) -> None:
        predictions = [
            {"date": "2026-04-15", "pred_close": 150.0},
            {"date": "2026-04-16", "pred_close": 151.5},
        ]
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=predictions,
        )

        frame = ForecastPresenter.format_predictions_table(result)

        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 2
        assert list(frame.columns) == ["date", "pred_close"]
        assert frame.iloc[0]["pred_close"] == 150.0

    def test_format_predictions_table_preserves_all_columns(self) -> None:
        predictions = [
            {"date": "2026-04-15", "pred_close": 150.0, "pred_volume": 1000000},
        ]
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=predictions,
        )

        frame = ForecastPresenter.format_predictions_table(result)

        assert set(frame.columns) == {"date", "pred_close", "pred_volume"}

    def test_format_price_chart_data_returns_none_for_empty_predictions(self) -> None:
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=[],
        )

        chart_df = ForecastPresenter.format_price_chart_data(result)

        assert chart_df is None

    def test_format_price_chart_data_returns_none_if_missing_date_column(self) -> None:
        predictions = [
            {"pred_close": 150.0},  # Missing date column
        ]
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=predictions,
        )

        chart_df = ForecastPresenter.format_price_chart_data(result)

        assert chart_df is None

    def test_format_price_chart_data_returns_none_if_missing_pred_close_column(self) -> None:
        predictions = [
            {"date": "2026-04-15"},  # Missing pred_close column
        ]
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=predictions,
        )

        chart_df = ForecastPresenter.format_price_chart_data(result)

        assert chart_df is None

    def test_format_price_chart_data_extracts_and_indexes_by_date(self) -> None:
        predictions = [
            {"date": "2026-04-15", "pred_close": 150.0, "extra": "ignored"},
            {"date": "2026-04-16", "pred_close": 151.5, "extra": "ignored"},
        ]
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=predictions,
        )

        chart_df = ForecastPresenter.format_price_chart_data(result)

        assert chart_df is not None
        assert isinstance(chart_df, pd.DataFrame)
        assert list(chart_df.columns) == ["pred_close"]
        assert chart_df.index.name == "date"
        assert len(chart_df) == 2

    def test_format_price_chart_data_handles_invalid_dates(self) -> None:
        predictions = [
            {"date": "invalid-date", "pred_close": 150.0},
            {"date": "2026-04-16", "pred_close": 151.5},
        ]
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=predictions,
        )

        chart_df = ForecastPresenter.format_price_chart_data(result)

        # Should drop invalid date rows and return only valid ones
        assert chart_df is not None
        assert len(chart_df) == 1
        assert chart_df.iloc[0]["pred_close"] == 151.5

    def test_format_price_chart_data_returns_none_if_all_dates_invalid(self) -> None:
        predictions = [
            {"date": "not-a-date", "pred_close": 150.0},
            {"date": "also-invalid", "pred_close": 151.5},
        ]
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=predictions,
        )

        chart_df = ForecastPresenter.format_price_chart_data(result)

        assert chart_df is None


class TestComparisonPresenter:
    """Tests for ComparisonPresenter formatting logic."""

    def test_format_leaderboard_frame_returns_empty_dataframe_for_empty_rows(self) -> None:
        result = CompareModelsResult(
            rows=[],
            rank_by=["mae", "rmse"],
            metric_directions={"mae": "asc", "rmse": "asc"},
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        assert isinstance(frame, pd.DataFrame)
        assert frame.empty

    def test_format_leaderboard_frame_includes_base_columns(self) -> None:
        rows = [
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-10T120000Z__ridge",
                metrics={"mae": 0.09},
                sort_key=(0.09, "ridge", "2026-04-10T120000Z__ridge"),
            )
        ]
        result = CompareModelsResult(
            rows=rows,
            rank_by=["mae"],
            metric_directions={"mae": "asc"},
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        assert "rank" in frame.columns
        assert "model" in frame.columns
        assert "model_id" in frame.columns
        assert "run_id" in frame.columns

    def test_format_leaderboard_frame_applies_label_lookup(self) -> None:
        rows = [
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-10T120000Z__ridge",
                metrics={"mae": 0.09},
                sort_key=(0.09, "ridge", "2026-04-10T120000Z__ridge"),
            )
        ]
        result = CompareModelsResult(
            rows=rows,
            rank_by=["mae"],
            metric_directions={"mae": "asc"},
        )
        label_lookup = {"ridge": "Ridge Regression"}

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup=label_lookup)

        assert frame.iloc[0]["model"] == "Ridge Regression"
        assert frame.iloc[0]["model_id"] == "ridge"

    def test_format_leaderboard_frame_uses_model_id_as_fallback_label(self) -> None:
        rows = [
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-10T120000Z__ridge",
                metrics={"mae": 0.09},
                sort_key=(0.09, "ridge", "2026-04-10T120000Z__ridge"),
            )
        ]
        result = CompareModelsResult(
            rows=rows,
            rank_by=["mae"],
            metric_directions={"mae": "asc"},
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        # Should use model_id as fallback when no label found
        assert frame.iloc[0]["model"] == "ridge"

    def test_format_leaderboard_frame_orders_columns_correctly(self) -> None:
        rows = [
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-10T120000Z__ridge",
                metrics={"mae": 0.09, "rmse": 0.18, "direction_accuracy": 0.83, "extra": 7},
                sort_key=(0.09, 0.18, -0.83, "ridge", "2026-04-10T120000Z__ridge"),
            )
        ]
        result = CompareModelsResult(
            rows=rows,
            rank_by=["mae", "rmse", "direction_accuracy"],
            metric_directions={"mae": "asc", "rmse": "asc", "direction_accuracy": "desc"},
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        expected_columns = ["rank", "model", "model_id", "run_id", "mae", "rmse", "direction_accuracy", "extra"]
        assert list(frame.columns) == expected_columns

    def test_format_leaderboard_frame_puts_ranking_metrics_first(self) -> None:
        rows = [
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-10T120000Z__ridge",
                metrics={"extra": 7, "mae": 0.09, "rmse": 0.18},
                sort_key=(0.09, 0.18, "ridge", "2026-04-10T120000Z__ridge"),
            )
        ]
        result = CompareModelsResult(
            rows=rows,
            rank_by=["mae", "rmse"],
            metric_directions={"mae": "asc", "rmse": "asc"},
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        # Ranking metrics should come before other metrics
        mae_idx = list(frame.columns).index("mae")
        rmse_idx = list(frame.columns).index("rmse")
        extra_idx = list(frame.columns).index("extra")
        assert mae_idx < rmse_idx < extra_idx

    def test_format_leaderboard_frame_sorts_remaining_columns_alphabetically(self) -> None:
        rows = [
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-10T120000Z__ridge",
                metrics={"zebra": 1, "apple": 2, "banana": 3, "mae": 0.09},
                sort_key=(0.09, "ridge", "2026-04-10T120000Z__ridge"),
            )
        ]
        result = CompareModelsResult(
            rows=rows,
            rank_by=["mae"],
            metric_directions={"mae": "asc"},
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        # Other columns should be sorted alphabetically after ranking metrics
        remaining_after_rank = list(frame.columns)[5:]  # After rank, model, model_id, run_id, mae
        assert remaining_after_rank == ["apple", "banana", "zebra"]

    def test_format_leaderboard_frame_preserves_data_values(self) -> None:
        rows = [
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-10T120000Z__ridge",
                metrics={"mae": 0.09, "rmse": 0.18},
                sort_key=(0.09, 0.18, "ridge", "2026-04-10T120000Z__ridge"),
            )
        ]
        result = CompareModelsResult(
            rows=rows,
            rank_by=["mae", "rmse"],
            metric_directions={"mae": "asc", "rmse": "asc"},
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        assert frame.iloc[0]["rank"] == 1
        assert frame.iloc[0]["model_id"] == "ridge"
        assert frame.iloc[0]["run_id"] == "2026-04-10T120000Z__ridge"
        assert frame.iloc[0]["mae"] == 0.09
        assert frame.iloc[0]["rmse"] == 0.18

    def test_format_leaderboard_frame_handles_multiple_rows(self) -> None:
        rows = [
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-10T120000Z__ridge",
                metrics={"mae": 0.09},
                sort_key=(0.09, "ridge", "2026-04-10T120000Z__ridge"),
            ),
            ModelComparisonRow(
                rank=2,
                model_id="linear",
                run_id="2026-04-10T120000Z__linear",
                metrics={"mae": 0.12},
                sort_key=(0.12, "linear", "2026-04-10T120000Z__linear"),
            ),
        ]
        result = CompareModelsResult(
            rows=rows,
            rank_by=["mae"],
            metric_directions={"mae": "asc"},
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        assert len(frame) == 2
        assert frame.iloc[0]["rank"] == 1
        assert frame.iloc[1]["rank"] == 2
        assert frame.iloc[0]["model_id"] == "ridge"
        assert frame.iloc[1]["model_id"] == "linear"

