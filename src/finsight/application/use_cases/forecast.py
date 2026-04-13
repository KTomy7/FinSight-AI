from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import pandas as pd

import finsight.application.dto as application_dto
from finsight.application.use_cases.fetch_market_data import FetchMarketData
from finsight.domain.ports import FeatureStorePort, ModelRegistryPort


def _require_non_empty_text(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _resolve_feature_columns(manifest: Any) -> list[str]:
    if not isinstance(manifest, dict):
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
        ticker = _require_non_empty_text(request.ticker, field_name="ticker")
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

        inference_frame = self._feature_store.build_inference_feature_dataset([market_data.history])
        if not isinstance(inference_frame, pd.DataFrame):
            raise TypeError("Feature store must return a pandas DataFrame for inference dataset.")
        if inference_frame.empty:
            raise ValueError("Inference feature dataset is empty; cannot produce a forecast.")

        missing = [column for column in feature_columns if column not in inference_frame.columns]
        if missing:
            raise ValueError(f"Inference feature dataset is missing required columns from manifest: {missing}")

        latest_row = inference_frame.sort_values(["date", "ticker"]).iloc[-1]
        latest_date = pd.to_datetime(latest_row["date"], errors="coerce")
        if pd.isna(latest_date):
            raise ValueError("Inference feature row contains an invalid 'date' value.")

        x_single = latest_row.loc[feature_columns].to_dict()
        x_infer = pd.DataFrame([x_single for _ in range(request.horizon_days)], columns=feature_columns)

        model = run_artifacts.model
        if not hasattr(model, "predict"):
            raise TypeError("Loaded model artifact does not implement predict(...).")

        y_pred_raw = model.predict(x_infer)
        y_pred = [float(value) for value in y_pred_raw]
        if len(y_pred) < request.horizon_days:
            raise ValueError("Model returned fewer predictions than the requested horizon_days.")
        y_pred = y_pred[: request.horizon_days]

        previous_close = _extract_last_close(market_data.history.df)
        predictions: list[application_dto.SerializableRow] = []

        for day_offset, predicted_return in enumerate(y_pred, start=1):
            previous_close = previous_close * (1.0 + predicted_return)
            prediction_date = (latest_date + timedelta(days=day_offset)).date().isoformat()
            predictions.append(
                {
                    "date": prediction_date,
                    "pred_ret_1d": float(predicted_return),
                    "pred_close": float(previous_close),
                }
            )

        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        return application_dto.ForecastResult(
            model_id=model_id,
            ticker=ticker,
            horizon_days=request.horizon_days,
            predictions=predictions,
            generated_at=generated_at,
        )

