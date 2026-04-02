from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finsight.domain.entities import ModelEvaluationResult
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

        feature_columns = self._feature_columns(train_df, test_df, target_column=target_column, id_columns=id_columns)
        if not feature_columns:
            raise ValueError("No numeric feature columns available for linear model evaluation.")

        x_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
        x_test = test_df.loc[:, feature_columns].to_numpy(dtype=float)
        y_train = train_df[target_column].to_numpy(dtype=float)
        y_test = test_df[target_column].to_numpy(dtype=float)

        alpha = 100.0
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        ridge_model = model.named_steps["ridge"]
        coefficient_ranking = self._coefficient_ranking(feature_columns, ridge_model.coef_)

        metrics = forecast_metrics(y_true=y_test.tolist(), y_pred=y_pred.tolist())

        pred_cols = [col for col in id_columns if col in test_df.columns]
        predictions = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
        predictions["y_true"] = y_test
        predictions["y_pred"] = y_pred

        return ModelEvaluationResult(
            metrics=metrics,
            predictions=predictions.reset_index(drop=True),
            trained_artifact=model,
            model_metadata={
                "adapter": "LinearSklearnModel",
                "model_id": model_type,
                "estimator": model.__class__.__name__,
                "base_estimator": ridge_model.__class__.__name__,
                "feature_columns": feature_columns,
                "n_features": len(feature_columns),
                "hyperparams": {"alpha": alpha},
                "preprocessing": {"scaler": "StandardScaler"},
                "coefficient_space": "standardized",
                "coefficients": {item["feature"]: item["coefficient"] for item in coefficient_ranking},
                "coefficient_ranking": coefficient_ranking,
            },
        )

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

    @staticmethod
    def _coefficient_ranking(feature_columns: list[str], coefficients: object) -> list[dict[str, float | str]]:
        coef_series = pd.Series(coefficients, index=feature_columns, dtype=float)
        ranking = coef_series.abs().sort_values(ascending=False)
        return [
            {
                "feature": str(feature),
                "coefficient": float(coef_series.loc[feature]),
                "abs_coefficient": float(abs_value),
            }
            for feature, abs_value in ranking.items()
        ]

