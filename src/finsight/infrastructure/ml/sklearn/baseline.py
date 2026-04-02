from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from finsight.domain.entities import ModelEvaluationResult
from finsight.domain.metrics import forecast_metrics
from finsight.domain.ports import ModelPort

SUPPORTED_MODEL_TYPES = ("naive_zero", "naive_mean")


@dataclass(frozen=True, slots=True)
class NaiveBaselineArtifact:
    baseline_value: float

    def predict(self, X: object) -> np.ndarray:
        n_rows = self._row_count(X)
        return np.full(n_rows, fill_value=self.baseline_value, dtype=float)

    @staticmethod
    def _row_count(X: object) -> int:
        if hasattr(X, "shape") and getattr(X, "shape"):
            return int(getattr(X, "shape")[0])
        try:
            return int(len(X))  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - defensive fallback
            raise TypeError("predict input must be a sized collection or array-like object.") from exc


class NaiveBaselineModel(ModelPort):
    def evaluate(
        self,
        *,
        train_dataset: object,
        test_dataset: object,
        model_type: str,
        target_column: str,
        id_columns: Sequence[str] = ("date", "ticker"),
    ) -> ModelEvaluationResult:
        train_df = self._require_dataframe(train_dataset, arg_name="train_dataset")
        test_df = self._require_dataframe(test_dataset, arg_name="test_dataset")

        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type '{model_type}'. Supported model types: {SUPPORTED_MODEL_TYPES}."
            )

        if target_column not in train_df.columns or target_column not in test_df.columns:
            raise ValueError(f"Both train_df and test_df must contain '{target_column}'.")

        if train_df.empty or test_df.empty:
            raise ValueError("train_df and test_df must be non-empty for evaluation.")

        y_train = train_df[target_column].to_numpy(dtype=float)
        y_test = test_df[target_column].to_numpy(dtype=float)

        if model_type == "naive_zero":
            y_pred = np.zeros_like(y_test, dtype=float)
            baseline_value = 0.0
        else:
            baseline_value = float(np.mean(y_train))
            y_pred = np.full_like(y_test, fill_value=baseline_value, dtype=float)

        metrics = forecast_metrics(y_true=y_test.tolist(), y_pred=y_pred.tolist())

        pred_cols = [col for col in id_columns if col in test_df.columns]
        predictions = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
        predictions["y_true"] = y_test
        predictions["y_pred"] = y_pred

        return ModelEvaluationResult(
            metrics=metrics,
            predictions=predictions.reset_index(drop=True),
            trained_artifact=NaiveBaselineArtifact(baseline_value=baseline_value),
            model_metadata={
                "adapter": "NaiveBaselineModel",
                "model_id": model_type,
                "baseline_value": baseline_value,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
            },
        )

    def supported_model_types(self) -> tuple[str, ...]:
        return SUPPORTED_MODEL_TYPES

    @staticmethod
    def _require_dataframe(dataset: object, *, arg_name: str) -> pd.DataFrame:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"{arg_name} must be a pandas DataFrame.")
        return dataset

