"""
Presenters for converting domain/DTO objects into display-ready formats.

This module provides pure formatting functions that convert use-case outputs
into data structures optimized for UI rendering. Presenters do not contain
Streamlit calls (st.write, st.dataframe, etc.) to maintain separation of
concerns and improve testability.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

import finsight.application.dto as application_dto


class ForecastPresenter:
    """Converts ForecastResult into display-ready formats."""

    @staticmethod
    def format_predictions_table(result: application_dto.ForecastResult) -> pd.DataFrame:
        """
        Convert forecast predictions into a DataFrame for tabular display.

        Args:
            result: ForecastResult from the Forecast use case.

        Returns:
            DataFrame with predictions, or empty DataFrame if no predictions.

        Raises:
            ValueError: If predictions cannot be converted to a valid DataFrame.
        """
        if not result.predictions:
            return pd.DataFrame()

        try:
            frame = pd.DataFrame(result.predictions)
            return frame
        except (TypeError, ValueError) as exc:
            raise ValueError("Failed to convert predictions to DataFrame.") from exc

    @staticmethod
    def format_price_chart_data(
        result: application_dto.ForecastResult,
    ) -> pd.DataFrame | None:
        """
        Extract date and pred_close columns for price chart visualization.

        Returns None if required columns are missing or if the DataFrame is empty.
        Otherwise returns a DataFrame with date as the index and pred_close as the value.

        Args:
            result: ForecastResult from the Forecast use case.

        Returns:
            DataFrame indexed by date with pred_close column, or None if chart cannot be rendered.
        """
        predictions_df = ForecastPresenter.format_predictions_table(result)
        if predictions_df.empty:
            return None

        required_cols = {"date", "pred_close"}
        if not required_cols.issubset(predictions_df.columns):
            return None

        try:
            chart_df = predictions_df[["date", "pred_close"]].copy()
            # Forecast dates are emitted as ISO calendar dates (YYYY-MM-DD).
            chart_df["date"] = pd.to_datetime(chart_df["date"], format="%Y-%m-%d", errors="coerce")
            chart_df = chart_df.dropna(subset=["date"]).set_index("date")

            if chart_df.empty:
                return None

            return chart_df
        except (TypeError, KeyError, ValueError):
            return None


class ComparisonPresenter:
    """Converts CompareModelsResult into display-ready formats."""

    @staticmethod
    def format_leaderboard_frame(
        result: application_dto.CompareModelsResult,
        *,
        label_lookup: Mapping[str, str],
    ) -> pd.DataFrame:
        """
        Convert comparison result into a formatted leaderboard DataFrame.

        Columns are ordered: rank, model (with label), model_id, run_id, then
        ranking metrics (in priority order), then remaining columns (sorted).

        Args:
            result: CompareModelsResult from the CompareModels use case.
            label_lookup: Mapping from model_id to human-readable label.

        Returns:
            Formatted DataFrame with columns in display order, or empty DataFrame if no rows.
        """
        if not result.rows:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for row in result.rows:
            record: dict[str, object] = {
                "rank": row.rank,
                "model": label_lookup.get(row.model_id, row.model_id),
                "model_id": row.model_id,
                "run_id": row.run_id,
            }
            record.update(row.metrics)
            rows.append(record)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame

        # Reorder columns: rank, model, model_id, run_id, ranking metrics, then others
        base_columns = ["rank", "model", "model_id", "run_id"]
        metric_columns = [column for column in result.rank_by if column in frame.columns]
        remaining_columns = [
            column
            for column in frame.columns
            if column not in base_columns and column not in metric_columns
        ]
        return frame[base_columns + metric_columns + sorted(remaining_columns)]


__all__ = ["ForecastPresenter", "ComparisonPresenter"]


