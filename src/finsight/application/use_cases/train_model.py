from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from finsight.application.use_cases.fetch_market_data import FetchMarketData, FetchMarketDataRequest
from finsight.infrastructure.features.feature_pipeline import FEATURE_COLUMNS, build_feature_dataset
from finsight.infrastructure.features.policies import TimeSplitPolicy

TARGET_COLUMN = "target_ret_1d"
SUPPORTED_MODEL_TYPES = ("naive_zero", "naive_mean")


@dataclass(frozen=True, slots=True)
class TrainModelRequest:
    tickers: list[str]
    cutoff_date: str
    years: int = 2
    end: str | None = None
    interval: str = "1d"
    model_types: list[str] = field(default_factory=lambda: ["naive_zero", "naive_mean"])
    artifacts_dir: str = "artifacts/runs"


@dataclass(frozen=True, slots=True)
class TrainModelResponse:
    run_dirs: dict[str, str]
    metrics: dict[str, dict[str, float | int | str]]


def _sign(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, 1, np.where(values < 0.0, -1, 0))


def _frame_date_range(df: pd.DataFrame, *, date_col: str = "date") -> tuple[str, str]:
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    min_ts = parsed.min()
    max_ts = parsed.max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        raise ValueError(f"DataFrame column '{date_col}' does not contain valid dates.")
    return min_ts.date().isoformat(), max_ts.date().isoformat()


def evaluate_naive_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_types: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Both train_df and test_df must contain '{TARGET_COLUMN}'.")

    if train_df.empty or test_df.empty:
        raise ValueError("train_df and test_df must be non-empty for evaluation.")

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, pd.DataFrame] = {}

    pred_cols = [col for col in ("date", "ticker") if col in test_df.columns]

    for model_type in model_types:
        if model_type == "naive_zero":
            y_pred = np.zeros_like(y_test, dtype=float)
        elif model_type == "naive_mean":
            y_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)
        else:
            raise ValueError(
                f"Unsupported model type '{model_type}'. Supported model types: {SUPPORTED_MODEL_TYPES}."
            )

        abs_errors = np.abs(y_test - y_pred)
        sq_errors = np.square(y_test - y_pred)

        y_pred_dir = (y_pred > 0).astype(int)
        y_true_dir = (y_test > 0).astype(int)

        direction_accuracy = np.mean(y_pred_dir == y_true_dir)

        metrics[model_type] = {
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(sq_errors))),
            "direction_accuracy": float(direction_accuracy),
        }

        pred_df = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = y_pred
        predictions[model_type] = pred_df.reset_index(drop=True)

    return metrics, predictions


class TrainModel:
    def __init__(self, fetch_market_data: FetchMarketData) -> None:
        self._fetch_market_data = fetch_market_data

    def execute(self, request: TrainModelRequest) -> TrainModelResponse:
        if not request.tickers:
            raise ValueError("tickers must contain at least one symbol.")
        if request.years <= 0:
            raise ValueError("years must be a positive integer.")

        end_date = date.fromisoformat(request.end) if request.end else date.today()
        lookback_days = (request.years * 365) - 1
        start_date = end_date - timedelta(days=lookback_days)

        series_list = []
        for ticker in request.tickers:
            result = self._fetch_market_data.execute(
                FetchMarketDataRequest(
                    ticker=ticker,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    interval=request.interval,
                    include_summary=False,
                )
            )
            series_list.append(result.history)

        features_df = build_feature_dataset(series_list)

        policy = TimeSplitPolicy(
            cutoff_date=request.cutoff_date,
            date_col="date",
            inclusive_test=True,
        )
        train_df, test_df = policy.split_frame(features_df)

        model_metrics, predictions = evaluate_naive_models(train_df, test_df, request.model_types)

        train_min_date, train_max_date = _frame_date_range(train_df)
        test_min_date, test_max_date = _frame_date_range(test_df)

        artifact_root = Path(request.artifacts_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).replace(microsecond=0)
        created_at_utc = created_at.isoformat().replace("+00:00", "Z")
        run_prefix = created_at.strftime("%Y-%m-%dT%H%M%SZ")

        run_dirs: dict[str, str] = {}
        metrics: dict[str, dict[str, float | int | str]] = {}

        for model_type in request.model_types:
            run_id = f"{run_prefix}__{model_type}"
            run_dir = self._create_unique_run_dir(artifact_root / run_id)

            enriched_metrics: dict[str, float | int | str] = {
                **model_metrics[model_type],
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "train_min_date": train_min_date,
                "train_max_date": train_max_date,
                "test_min_date": test_min_date,
                "test_max_date": test_max_date,
            }

            metadata = {
                "run_id": run_dir.name,
                "created_at_utc": created_at_utc,
                "model_type": model_type,
                "tickers": request.tickers,
                "interval": request.interval,
                "years": int(request.years),
                "end": request.end or end_date.isoformat(),
                "cutoff_date": request.cutoff_date,
                "horizon": "1d",
                "feature_columns": list(FEATURE_COLUMNS),
                "target_column": TARGET_COLUMN,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "train_min_date": train_min_date,
                "train_max_date": train_max_date,
                "test_min_date": test_min_date,
                "test_max_date": test_max_date,
            }

            (run_dir / "metrics.json").write_text(
                json.dumps(enriched_metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            predictions[model_type].to_csv(run_dir / "predictions.csv", index=False)

            run_dirs[model_type] = str(run_dir)
            metrics[model_type] = enriched_metrics

        return TrainModelResponse(run_dirs=run_dirs, metrics=metrics)

    @staticmethod
    def _create_unique_run_dir(base_path: Path) -> Path:
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=False)
            return base_path

        suffix = 1
        while True:
            candidate = Path(f"{base_path}_{suffix}")
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            suffix += 1

