"""Unit tests for NaiveBaselineModel in finsight.infrastructure.ml.sklearn.baseline."""
from __future__ import annotations

import pandas as pd
import pytest

from finsight.infrastructure.ml.sklearn.baseline import NaiveBaselineModel


def _make_df(rows: int = 5, target_col: str = "target_ret_1d") -> pd.DataFrame:
    """Return a small deterministic OHLCV-style DataFrame with a target column."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "ticker": ["AAPL"] * rows,
            target_col: [0.01, -0.02, 0.03, 0.01, -0.01][:rows],
        }
    )


class TestNaiveBaselineModelEvaluate:
    def setup_method(self) -> None:
        self.model = NaiveBaselineModel()
        self.train = _make_df(rows=3)
        self.test = _make_df(rows=2)

    def test_unsupported_model_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model type"):
            self.model.evaluate(
                train_dataset=self.train,
                test_dataset=self.test,
                model_type="naive_last",
                target_column="target_ret_1d",
            )

    def test_missing_target_column_raises(self) -> None:
        with pytest.raises(ValueError, match="Both train_df and test_df must contain"):
            self.model.evaluate(
                train_dataset=self.train,
                test_dataset=self.test,
                model_type="naive_zero",
                target_column="nonexistent_column",
            )

    def test_empty_train_df_raises(self) -> None:
        empty = _make_df(rows=3).iloc[0:0]  # zero-row slice keeps columns
        with pytest.raises(ValueError, match="must be non-empty"):
            self.model.evaluate(
                train_dataset=empty,
                test_dataset=self.test,
                model_type="naive_zero",
                target_column="target_ret_1d",
            )

    def test_empty_test_df_raises(self) -> None:
        empty = _make_df(rows=3).iloc[0:0]
        with pytest.raises(ValueError, match="must be non-empty"):
            self.model.evaluate(
                train_dataset=self.train,
                test_dataset=empty,
                model_type="naive_zero",
                target_column="target_ret_1d",
            )

    def test_naive_zero_predictions_are_all_zero(self) -> None:
        _, preds = self.model.evaluate(
            train_dataset=self.train,
            test_dataset=self.test,
            model_type="naive_zero",
            target_column="target_ret_1d",
        )
        assert (preds["y_pred"] == 0.0).all()

    def test_naive_mean_predictions_equal_train_mean(self) -> None:
        import numpy as np

        expected_mean = float(self.train["target_ret_1d"].mean())
        _, preds = self.model.evaluate(
            train_dataset=self.train,
            test_dataset=self.test,
            model_type="naive_mean",
            target_column="target_ret_1d",
        )
        assert np.allclose(preds["y_pred"].to_numpy(), expected_mean)

    def test_metrics_keys_present(self) -> None:
        metrics, _ = self.model.evaluate(
            train_dataset=self.train,
            test_dataset=self.test,
            model_type="naive_zero",
            target_column="target_ret_1d",
        )
        assert {"mae", "rmse", "direction_accuracy"}.issubset(metrics.keys())


class TestRequireDataframe:
    def test_non_dataframe_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            NaiveBaselineModel._require_dataframe([1, 2, 3], arg_name="train_dataset")

    def test_dataframe_returned_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        result = NaiveBaselineModel._require_dataframe(df, arg_name="train_dataset")
        assert result is df
