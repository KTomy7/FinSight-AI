from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

import finsight.application.dto as application_dto
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.domain.entities import OHLCVSeries
from finsight.domain.ports import FeatureStorePort, ModelRegistryPort


def _require_non_empty_text(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _resolve_feature_columns(manifest: Any) -> list[str]:
    if not isinstance(manifest, Mapping):
        raise TypeError("Model manifest must be a mapping.")

    raw_feature_columns = manifest.get("feature_columns")
    if not isinstance(raw_feature_columns, Sequence) or isinstance(raw_feature_columns, (str, bytes)):
        raise TypeError("Model manifest key 'feature_columns' must be a non-empty sequence of strings.")

    feature_columns = [str(column) for column in raw_feature_columns if str(column).strip()]
    if not feature_columns:
        raise ValueError("Model manifest key 'feature_columns' must contain at least one feature name.")

    return feature_columns


def _extract_last_close(history_df: pd.DataFrame) -> float:
    if history_df.empty:
        raise ValueError("Cannot forecast with empty market history.")

    if "Close" in history_df.columns:
        close_col = "Close"
    elif "close" in history_df.columns:
        close_col = "close"
    else:
        raise ValueError("Market history is missing a close price column ('Close' or 'close').")

    close_series = pd.to_numeric(history_df[close_col], errors="coerce")
    if close_series.isna().all():
        raise ValueError("Market history close column does not contain numeric values.")

    return float(close_series.dropna().iloc[-1])


def _extract_last_volume(history_df: pd.DataFrame) -> float:
    if "Volume" in history_df.columns:
        volume_col = "Volume"
    elif "volume" in history_df.columns:
        volume_col = "volume"
    else:
        raise ValueError("Market history is missing a volume column ('Volume' or 'volume').")

    volume_series = pd.to_numeric(history_df[volume_col], errors="coerce")
    if volume_series.isna().all():
        raise ValueError("Market history volume column does not contain numeric values.")
    return float(volume_series.dropna().iloc[-1])


def _resolve_history_columns(history_df: pd.DataFrame) -> dict[str, str]:
    pairs = {
        "date": ("Date", "date"),
        "open": ("Open", "open"),
        "high": ("High", "high"),
        "low": ("Low", "low"),
        "close": ("Close", "close"),
        "volume": ("Volume", "volume"),
    }

    resolved: dict[str, str] = {}
    for logical_name, candidates in pairs.items():
        column = next((candidate for candidate in candidates if candidate in history_df.columns), None)
        if column is None:
            raise ValueError(f"Market history is missing required column for '{logical_name}': {candidates}")
        resolved[logical_name] = column
    return resolved


def _next_business_day(current_date: date) -> date:
    candidate = current_date + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate = candidate + timedelta(days=1)
    return candidate


def _append_synthetic_history_row(
    history_df: pd.DataFrame,
    *,
    columns: Mapping[str, str],
    next_date: date,
    next_close: float,
    next_volume: float,
) -> pd.DataFrame:
    synthetic_row = {
        columns["date"]: pd.Timestamp(next_date),
        columns["open"]: float(next_close),
        columns["high"]: float(next_close),
        columns["low"]: float(next_close),
        columns["close"]: float(next_close),
        columns["volume"]: float(next_volume),
    }
    history_df.loc[len(history_df)] = synthetic_row
    return history_df

class Forecast:
    def __init__(
        self,
        *,
        fetch_market_data: FetchMarketData,
        feature_store: FeatureStorePort,
        model_registry: ModelRegistryPort,
    ) -> None:
        self._fetch_market_data = fetch_market_data
        self._feature_store = feature_store
        self._model_registry = model_registry

    def execute(self, request: application_dto.ForecastRequest) -> application_dto.ForecastResult:
        ticker = _require_non_empty_text(request.ticker, field_name="ticker").upper()
        model_id = _require_non_empty_text(request.model_id, field_name="model_id")

        if request.horizon_days <= 0:
            raise ValueError("horizon_days must be a positive integer.")

        run_id = self._model_registry.latest_run_id(
            artifact_root=request.artifacts_dir,
            model_id=model_id,
        )
        run_artifacts = self._model_registry.load_run_artifacts(
            artifact_root=request.artifacts_dir,
            run_id=run_id,
        )

        feature_columns = _resolve_feature_columns(run_artifacts.manifest)

        market_data = self._fetch_market_data.execute(
            application_dto.FetchMarketDataRequest(
                ticker=ticker,
                include_summary=False,
            )
        )

        model = run_artifacts.model
        if not hasattr(model, "predict"):
            raise TypeError("Loaded model artifact does not implement predict(...).")

        history_series = market_data.history
        history_df = history_series.df.copy()
        history_columns = _resolve_history_columns(history_df)
        history_dates = pd.to_datetime(history_df[history_columns["date"]], errors="coerce").dropna()
        if history_dates.empty:
            raise ValueError("Market history date column does not contain valid dates.")
        current_date = history_dates.iloc[-1].date()
        previous_close = _extract_last_close(history_df)
        previous_volume = _extract_last_volume(history_df)

        current_history = OHLCVSeries(
            ticker=history_series.ticker,
            date_range=history_series.date_range,
            interval=history_series.interval,
            df=history_df,
        )

        predictions: list[application_dto.SerializableRow] = []

        for _ in range(request.horizon_days):
            inference_frame = self._feature_store.build_inference_feature_dataset([current_history])
            if not isinstance(inference_frame, pd.DataFrame):
                raise TypeError("Feature store must return a pandas DataFrame for inference dataset.")
            if inference_frame.empty:
                raise ValueError("Inference feature dataset is empty; cannot produce a forecast.")
            if "ticker" not in inference_frame.columns or "date" not in inference_frame.columns:
                raise ValueError("Inference feature dataset must include 'ticker' and 'date' columns.")

            ticker_mask = inference_frame["ticker"].astype(str).str.upper().str.strip() == ticker
            ticker_frame = inference_frame.loc[ticker_mask]
            if ticker_frame.empty:
                raise ValueError(f"Inference feature dataset does not contain rows for ticker '{ticker}'.")

            missing = [column for column in feature_columns if column not in ticker_frame.columns]
            if missing:
                raise ValueError(f"Inference feature dataset is missing required columns from manifest: {missing}")

            latest_row = ticker_frame.sort_values(["date"]).iloc[-1]

            x_input = latest_row.loc[feature_columns].to_numpy(dtype=float).reshape(1, -1)
            y_pred_raw = model.predict(x_input)
            try:
                predicted_return = float(y_pred_raw[0])  # type: ignore[index]
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError("Model predict(...) must return at least one numeric value.") from exc
            if not math.isfinite(predicted_return):
                raise ValueError("Model predict(...) must return a finite numeric value.")

            next_date = _next_business_day(current_date)
            next_close = previous_close * (1.0 + predicted_return)
            next_volume = previous_volume * (1.0 + 0.05 * predicted_return)
            predictions.append(
                {
                    "date": next_date.isoformat(),
                    "pred_ret_1d": predicted_return,
                    "pred_close": float(next_close),
                }
            )

            history_df = _append_synthetic_history_row(
                history_df,
                columns=history_columns,
                next_date=next_date,
                next_close=next_close,
                next_volume=next_volume,
            )
            current_history = OHLCVSeries(
                ticker=history_series.ticker,
                date_range=history_series.date_range,
                interval=history_series.interval,
                df=history_df,
            )
            current_date = next_date
            previous_close = float(next_close)
            previous_volume = float(next_volume)

        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        return application_dto.ForecastResult(
            model_id=model_id,
            ticker=ticker,
            horizon_days=request.horizon_days,
            predictions=predictions,
            generated_at=generated_at,
        )

