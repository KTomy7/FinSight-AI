from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from finsight.domain.entities import ModelEvaluationResult
from finsight.domain.metrics import forecast_metrics
from finsight.domain.ports import ModelPort

SUPPORTED_MODEL_TYPES = ("xgboost",)
RANDOM_STATE = 42


class XGBoostModel(ModelPort):
    """XGBoost gradient boosting regressor adapter."""

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
            raise ValueError("No numeric feature columns available for XGBoost model evaluation.")

        x_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
        x_test = test_df.loc[:, feature_columns].to_numpy(dtype=float)
        y_train = train_df[target_column].to_numpy(dtype=float)
        y_test = test_df[target_column].to_numpy(dtype=float)

        # Hyperparameters for XGBoost
        n_estimators = 300
        learning_rate = 0.03
        max_depth = 3
        subsample = 0.8
        colsample_bytree = 0.8
        reg_lambda = 1.0

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Compute feature importances (native XGBoost importance preferred)
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                feature_importance = self._feature_importance_ranking(feature_columns, importances)
            else:
                raise AttributeError
        except AttributeError:
            # Fallback: use permutation importance if direct importance is unavailable
            perm_importance = permutation_importance(
                model, x_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
            )
            feature_importance = self._feature_importance_ranking(
                feature_columns, perm_importance.importances_mean
            )

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
                "adapter": "XGBoostModel",
                "model_id": model_type,
                "estimator": model.__class__.__name__,
                "base_estimator": model.__class__.__name__,
                "feature_columns": feature_columns,
                "n_features": len(feature_columns),
                "hyperparams": {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "reg_lambda": reg_lambda,
                    "random_state": RANDOM_STATE,
                },
                "preprocessing": {},
                "feature_importance": {item["feature"]: item["importance"] for item in feature_importance},
                "feature_importance_ranking": feature_importance,
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
        numeric = [
            col
            for col in common
            if pd.api.types.is_numeric_dtype(train_df[col])
            and pd.api.types.is_numeric_dtype(test_df[col])
        ]
        return numeric

    @staticmethod
    def _require_dataframe(dataset: object, *, arg_name: str) -> pd.DataFrame:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"{arg_name} must be a pandas DataFrame.")
        return dataset

    @staticmethod
    def _feature_importance_ranking(
        feature_columns: list[str], importances: object
    ) -> list[dict[str, float | str]]:
        importance_series = pd.Series(importances, index=feature_columns, dtype=float)
        ranking = importance_series.sort_values(ascending=False)
        return [
            {
                "feature": str(feature),
                "importance": float(importance_series.loc[feature]),
                "rank": int(idx + 1),
            }
            for idx, (feature, _) in enumerate(ranking.items())
        ]

