from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.linear_model import Ridge

from finsight.domain.metrics import forecast_metrics
from finsight.domain.ports import ModelPort

SUPPORTED_MODEL_TYPES = ("ridge",)


class LinearSklearnModel(ModelPort):
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

        feature_columns = self._feature_columns(train_df, test_df, target_column=target_column, id_columns=id_columns)
        if not feature_columns:
            raise ValueError("No numeric feature columns available for linear model evaluation.")

        x_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
        x_test = test_df.loc[:, feature_columns].to_numpy(dtype=float)
        y_train = train_df[target_column].to_numpy(dtype=float)
        y_test = test_df[target_column].to_numpy(dtype=float)

        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = forecast_metrics(y_true=y_test.tolist(), y_pred=y_pred.tolist())

        pred_cols = [col for col in id_columns if col in test_df.columns]
        predictions = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
        predictions["y_true"] = y_test
        predictions["y_pred"] = y_pred

        return metrics, predictions.reset_index(drop=True)

    def supported_model_types(self) -> tuple[str, ...]:
        return SUPPORTED_MODEL_TYPES

    @staticmethod
    def _feature_columns(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        target_column: str,
        id_columns: Sequence[str],
    ) -> list[str]:
        excluded = {target_column, *id_columns}
        common = [col for col in train_df.columns if col in test_df.columns and col not in excluded]
        numeric = [col for col in common if pd.api.types.is_numeric_dtype(train_df[col])]
        return numeric

    @staticmethod
    def _require_dataframe(dataset: object, *, arg_name: str) -> pd.DataFrame:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"{arg_name} must be a pandas DataFrame.")
        return dataset

