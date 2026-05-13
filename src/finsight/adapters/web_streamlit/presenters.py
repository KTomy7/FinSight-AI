"""
Presenters for converting domain/DTO objects into display-ready formats.

This module provides pure formatting functions that convert use-case outputs
into data structures optimized for UI rendering. Presenters do not contain
Streamlit calls (st.write, st.dataframe, etc.) to maintain separation of
concerns and improve testability.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import pandas as pd

import finsight.application.dto as application_dto


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


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
            raise ValueError(f"Failed to convert predictions to DataFrame: {exc}") from exc

    @staticmethod
    def format_price_chart_data(
        result: application_dto.ForecastResult,
    ) -> pd.DataFrame | None:
        """
        Extract date and pred_close columns for price chart visualization.

        Returns None if required columns are missing or if the DataFrame is empty.
        Otherwise, returns a DataFrame with date as the index and pred_close as the value.

        Args:
            result: ForecastResult from the Forecast use case.

        Returns:
            DataFrame indexed by date with pred_close column, or None if chart cannot be rendered.
        """
        try:
            predictions_df = ForecastPresenter.format_predictions_table(result)
            if predictions_df.empty:
                return None

            required_cols = {"date", "pred_close"}
            if not required_cols.issubset(predictions_df.columns):
                return None

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


class TrainPresenter:
    """Converts TrainModelResult and run artifacts into display-ready frames.

    Presenters remain free of Streamlit calls; they prepare pandas DataFrames
    or None to be rendered by views.
    """

    @staticmethod
    def format_metrics_frame(result: application_dto.TrainModelResult, *, label_lookup: Mapping[str, str]) -> pd.DataFrame:
        if not result.metrics:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for model_id, model_metrics in result.metrics.items():
            record: dict[str, object] = {
                "model_id": model_id,
                "model": label_lookup.get(model_id, model_id),
            }
            record.update(model_metrics)
            rows.append(record)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        # Prefer ordering model column first
        cols = [c for c in ["model", "model_id"] if c in frame.columns] + [c for c in frame.columns if c not in ("model", "model_id")]
        return frame[cols]

    @staticmethod
    def load_predictions_csv(run_dir: str) -> pd.DataFrame | None:
        from pathlib import Path

        path = Path(run_dir) / "predictions.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            return None

    @staticmethod
    def assemble_backtest_for_ticker(predictions_df: pd.DataFrame, market_history_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Return a DataFrame with per-row: input_date, next_date, base_close, pred_next_close, actual_next_close, y_true, y_pred

        predictions_df: must contain columns 'date' and 'y_pred' (and optionally 'y_true' and 'ticker').
        market_history_df: OHLCV frame (Date/Close or date/close) for the ticker.
        """
        if predictions_df is None or predictions_df.empty:
            return pd.DataFrame()

        # Normalize prediction dates
        working = predictions_df.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working = working.dropna(subset=["date"]).sort_values(["date"]).reset_index(drop=True)

        # Normalize market history
        history = market_history_df.copy()
        if "Date" in history.columns:
            date_col = "Date"
        elif "date" in history.columns:
            date_col = "date"
        else:
            raise ValueError("Market history missing a date column. Expected 'Date' or 'date'.")

        if "Close" in history.columns:
            close_col = "Close"
        elif "close" in history.columns:
            close_col = "close"
        else:
            raise ValueError("Market history missing a close column. Expected 'Close' or 'close'.")

        history[date_col] = pd.to_datetime(history[date_col], errors="coerce")
        history = history.dropna(subset=[date_col]).sort_values([date_col]).reset_index(drop=True)
        # Map date -> close
        history_map = {dt.date(): float(val) for dt, val in zip(history[date_col], history[close_col]) if pd.notna(val)}

        rows: list[dict[str, object]] = []
        for _, row in working.iterrows():
            input_dt = row["date"].date()
            y_pred = _as_float(row.get("y_pred"))
            if y_pred is None:
                continue
            y_true = row.get("y_true")

            # base close try exact date, else last available before date
            base_close = None
            candidate = input_dt
            # search up to 7 days backwards
            for _ in range(8):
                if candidate in history_map:
                    base_close = history_map[candidate]
                    break
                candidate = candidate - pd.Timedelta(days=1)
                candidate = candidate.date() if isinstance(candidate, pd.Timestamp) else candidate

            if base_close is None:
                # can't reconstruct
                continue

            pred_next_close = base_close * (1.0 + y_pred)

            # actual next close: can be reconstructed from y_true and base_close
            actual_next_close = None
            actual_y_true = _as_float(y_true)
            if actual_y_true is not None:
                actual_next_close = base_close * (1.0 + actual_y_true)

            rows.append(
                {
                    "input_date": input_dt.isoformat(),
                    "next_date": (input_dt + timedelta(days=1)).isoformat(),
                    "ticker": ticker,
                    "base_close": float(base_close),
                    "pred_next_close": float(pred_next_close),
                    "actual_next_close": float(actual_next_close) if actual_next_close is not None else None,
                    "y_pred": y_pred,
                    "y_true": actual_y_true,
                }
            )

        return pd.DataFrame(rows)


__all__ = ["ForecastPresenter", "ComparisonPresenter", "TrainPresenter"]


