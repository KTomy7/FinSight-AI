"""Tests for presenter formatting functions."""
from __future__ import annotations

import pandas as pd
import pytest
from typing import cast

from finsight.adapters.web_streamlit.presenters import BacktestPresenter, ComparisonPresenter, ForecastPresenter, TrainPresenter
from finsight.application.dto import BacktestReport, BacktestResult, CompareModelsResult, ForecastResult, ModelComparisonRow, TrainModelResult
import finsight.application.dto as application_dto


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

    def test_format_predictions_table_raises_value_error_when_dataframe_conversion_fails(self, monkeypatch) -> None:
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=[{"date": "2026-04-15", "pred_close": 150.0}],
        )

        def _boom(*_args, **_kwargs):
            raise ValueError("bad frame")

        monkeypatch.setattr("finsight.adapters.web_streamlit.presenters.pd.DataFrame", _boom)

        with pytest.raises(ValueError, match="Failed to convert predictions to DataFrame"):
            ForecastPresenter.format_predictions_table(result)

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

    def test_format_price_chart_data_returns_none_when_predictions_table_raises(self, monkeypatch) -> None:
        result = ForecastResult(
            model_id="ridge",
            ticker="AAPL",
            horizon_days=7,
            predictions=[{"date": "2026-04-15", "pred_close": 150.0}],
        )

        monkeypatch.setattr(
            ForecastPresenter,
            "format_predictions_table",
            staticmethod(lambda _result: (_ for _ in ()).throw(ValueError("boom"))),
        )

        assert ForecastPresenter.format_price_chart_data(result) is None


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

    def test_format_leaderboard_frame_handles_empty_dataframe_after_construction(self, monkeypatch) -> None:
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

        real_dataframe = pd.DataFrame
        monkeypatch.setattr(
            "finsight.adapters.web_streamlit.presenters.pd.DataFrame",
            lambda *_args, **_kwargs: real_dataframe(),
        )

        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={})

        assert frame.empty


class TestTrainPresenter:
    def test_format_metrics_frame_applies_labels_and_orders_columns(self) -> None:
        result = TrainModelResult(
            run_dirs={"ridge": "artifacts/runs/run_1"},
            metrics={"ridge": {"mae": 0.12, "rmse": 0.34}},
        )

        frame = TrainPresenter.format_metrics_frame(result, label_lookup={"ridge": "Ridge Regression"})

        assert list(frame.columns) == ["model", "model_id", "mae", "rmse"]
        assert frame.iloc[0]["model"] == "Ridge Regression"
        assert frame.iloc[0]["model_id"] == "ridge"
        assert frame.iloc[0]["mae"] == 0.12
        assert frame.iloc[0]["rmse"] == 0.34

    def test_format_metrics_frame_returns_empty_dataframe_for_empty_metrics(self) -> None:
        result = TrainModelResult(run_dirs={}, metrics={})

        frame = TrainPresenter.format_metrics_frame(result, label_lookup={})

        assert isinstance(frame, pd.DataFrame)
        assert frame.empty

    def test_load_predictions_csv_returns_none_when_file_is_missing(self, tmp_path) -> None:
        run_dir = tmp_path / "missing-run"

        assert TrainPresenter.load_predictions_csv(str(run_dir)) is None

    def test_load_predictions_csv_returns_none_when_read_fails(self, tmp_path, monkeypatch) -> None:
        run_dir = tmp_path / "run_with_bad_csv"
        run_dir.mkdir()
        (run_dir / "predictions.csv").write_text("date,ticker,y_pred\n2026-04-10,AAPL,0.12\n", encoding="utf-8")

        monkeypatch.setattr("finsight.adapters.web_streamlit.presenters.pd.read_csv", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad csv")))

        assert TrainPresenter.load_predictions_csv(str(run_dir)) is None

    def test_assemble_backtest_for_ticker_reconstructs_actual_prices_from_y_true(self) -> None:
        predictions_df = pd.DataFrame(
            [
                {"date": "2026-04-10", "ticker": "AAPL", "y_pred": 0.02, "y_true": 0.05},
                {"date": "2026-04-13", "ticker": "AAPL", "y_pred": -0.01, "y_true": -0.02},
            ]
        )
        market_history_df = pd.DataFrame(
            [
                {"Date": "2026-04-10", "Close": 100.0},
                {"Date": "2026-04-13", "Close": 105.0},
            ]
        )

        frame = TrainPresenter.assemble_backtest_for_ticker(predictions_df, market_history_df, "AAPL")

        assert len(frame) == 2
        assert list(frame["input_date"]) == ["2026-04-10", "2026-04-13"]
        # Next date should prefer the next available prediction row, falling back to calendar day
        assert list(frame["next_date"]) == ["2026-04-13", "2026-04-14"]
        assert frame.iloc[0]["base_close"] == 100.0
        assert frame.iloc[0]["pred_next_close"] == 102.0
        assert frame.iloc[0]["actual_next_close"] == 105.0
        assert frame.iloc[1]["base_close"] == 105.0
        assert frame.iloc[1]["pred_next_close"] == pytest.approx(103.95)
        assert frame.iloc[1]["actual_next_close"] == pytest.approx(102.9)

    def test_assemble_backtest_for_ticker_walks_back_over_missing_weekend_dates(self) -> None:
        predictions_df = pd.DataFrame(
            [
                {"date": "2026-04-13", "ticker": "AAPL", "y_pred": 0.02, "y_true": 0.01},
            ]
        )
        market_history_df = pd.DataFrame(
            [
                {"Date": "2026-04-10", "Close": 100.0},
            ]
        )

        frame = TrainPresenter.assemble_backtest_for_ticker(predictions_df, market_history_df, "AAPL")

        assert len(frame) == 1
        assert frame.iloc[0]["input_date"] == "2026-04-13"
        assert frame.iloc[0]["base_close"] == 100.0
        assert frame.iloc[0]["pred_next_close"] == pytest.approx(102.0)
        assert frame.iloc[0]["actual_next_close"] == pytest.approx(101.0)


class TestBacktestPresenter:
    def test_format_model_metrics_frame_returns_empty_dataframe_for_empty_report(self) -> None:
        report = BacktestReport(results=[])

        frame = BacktestPresenter.format_model_metrics_frame(report, label_lookup={})

        assert isinstance(frame, pd.DataFrame)
        assert frame.empty

    def test_format_model_metrics_frame_applies_labels(self) -> None:
        report = BacktestReport(
            results=[
                BacktestResult(
                    model_id="ridge",
                    metrics={"fold_count": 2, "mae": 0.1, "rmse": 0.2},
                    folds=[],
                )
            ]
        )

        frame = BacktestPresenter.format_model_metrics_frame(report, label_lookup={"ridge": "Ridge Regression"})

        assert list(frame.columns) == ["model", "model_id", "fold_count", "mae", "rmse"]
        assert frame.iloc[0]["model"] == "Ridge Regression"
        assert frame.iloc[0]["model_id"] == "ridge"

    def test_format_model_metrics_frame_falls_back_to_model_id_and_preserves_extra_metrics(self) -> None:
        report = BacktestReport(
            results=[
                BacktestResult(
                    model_id="naive_zero",
                    metrics={"z_metric": 1.0, "mae": 0.1},
                    folds=[],
                )
            ]
        )

        frame = BacktestPresenter.format_model_metrics_frame(report, label_lookup={})

        assert list(frame.columns) == ["model", "model_id", "z_metric", "mae"]
        assert frame.iloc[0]["model"] == "naive_zero"
        assert frame.iloc[0]["model_id"] == "naive_zero"
        assert frame.iloc[0]["z_metric"] == 1.0

    def test_format_fold_frame_returns_empty_dataframe_for_empty_folds(self) -> None:
        result = BacktestResult(model_id="ridge", metrics={"fold_count": 0}, folds=[])

        frame = BacktestPresenter.format_fold_frame(result)

        assert isinstance(frame, pd.DataFrame)
        assert frame.empty

    def test_format_fold_frame_flattens_metrics(self) -> None:
        result = BacktestResult(
            model_id="ridge",
            metrics={"fold_count": 1, "mae": 0.1},
            folds=cast(
                list[application_dto.SerializableRow],
                cast(
                    object,
                [
                    {
                        "fold_index": 1,
                        "train_start": "2024-01-01",
                        "train_end": "2024-06-01",
                        "test_start": "2024-06-02",
                        "test_end": "2024-07-01",
                        "n_train": 120,
                        "n_test": 20,
                        "metrics": {"mae": 0.12, "rmse": 0.19},
                    }
                ],
                ),
            ),
        )

        frame = BacktestPresenter.format_fold_frame(result)

        assert len(frame) == 1
        assert "metrics" not in frame.columns
        assert frame.iloc[0]["fold_index"] == 1
        assert frame.iloc[0]["metric_mae"] == 0.12
        assert frame.iloc[0]["metric_rmse"] == 0.19

    def test_format_fold_frame_treats_non_mapping_metrics_as_empty_and_sorts_remaining_columns(self) -> None:
        result = BacktestResult(
            model_id="ridge",
            metrics={"fold_count": 1},
            folds=cast(
                list[application_dto.SerializableRow],
                cast(
                    object,
                    [
                        {
                            "fold_index": 1,
                            "n_test": 20,
                            "z_extra": "last",
                            "a_extra": "first",
                            "metrics": None,
                        }
                    ],
                ),
            ),
        )

        frame = BacktestPresenter.format_fold_frame(result)

        assert "metrics" not in frame.columns
        assert list(frame.columns) == ["fold_index", "n_test", "a_extra", "z_extra"]
        assert frame.iloc[0]["a_extra"] == "first"

