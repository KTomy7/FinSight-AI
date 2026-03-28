from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from finsight.domain.ports import ModelPort

SUPPORTED_MODEL_TYPES = ("naive_zero", "naive_mean")


class NaiveBaselineModel(ModelPort):
    def evaluate(
        self,
        *,
        train_dataset: object,
        test_dataset: object,
        model_type: str,
        target_column: str,
        id_columns: Sequence[str] = ("date", "ticker"),
    ) -> tuple[dict[str, float], pd.DataFrame]:
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
        else:
            y_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)

        abs_errors = np.abs(y_test - y_pred)
        sq_errors = np.square(y_test - y_pred)

        y_pred_dir = (y_pred > 0).astype(int)
        y_true_dir = (y_test > 0).astype(int)

        metrics = {
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(sq_errors))),
            "direction_accuracy": float(np.mean(y_pred_dir == y_true_dir)),
        }

        pred_cols = [col for col in id_columns if col in test_df.columns]
        predictions = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
        predictions["y_true"] = y_test
        predictions["y_pred"] = y_pred

        return metrics, predictions.reset_index(drop=True)

    @staticmethod
    def _require_dataframe(dataset: object, *, arg_name: str) -> pd.DataFrame:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"{arg_name} must be a pandas DataFrame.")
        return dataset

