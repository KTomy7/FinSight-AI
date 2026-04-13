import numpy as np
import pandas as pd
import pytest

from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, SUPPORTED_METRIC_NAMES
from finsight.infrastructure.ml.sklearn.linear import LinearSklearnModel


def _synthetic_linear_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "ret_1d": [0.1, 0.2, 0.3, 0.4],
            "mom_20d": [1.0, 2.0, 3.0, 4.0],
            "target_ret_1d": [0.2, 0.4, 0.6, 0.8],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-06"]),
            "ticker": ["AAA", "AAA"],
            "ret_1d": [0.5, 0.6],
            "mom_20d": [5.0, 6.0],
            "target_ret_1d": [1.0, 1.2],
        }
    )
    return train_df, test_df


def test_linear_sklearn_model_ridge_smoke_predicts_and_reports_metrics() -> None:
    model = LinearSklearnModel()
    train_df, test_df = _synthetic_linear_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="ridge",
        target_column="target_ret_1d",
    )

    metrics = result.metrics
    predictions = result.predictions
    assert set(SUPPORTED_METRIC_NAMES).issubset(metrics)
    assert 0.0 <= metrics[METRIC_DIRECTION_ACCURACY] <= 1.0
    assert list(predictions.columns) == ["date", "ticker", "y_true", "y_pred"]
    assert len(predictions) == len(test_df)
    assert np.isfinite(predictions["y_pred"]).all()
    assert result.trained_artifact.__class__.__name__ == "Pipeline"
    assert tuple(result.trained_artifact.named_steps.keys()) == ("scaler", "ridge")
    assert result.model_metadata["model_id"] == "ridge"
    assert result.model_metadata["base_estimator"] == "Ridge"
    assert result.model_metadata["preprocessing"]["scaler"] == "StandardScaler"
    assert result.model_metadata["coefficient_space"] == "standardized"
    assert "coefficients" in result.model_metadata
    assert "coefficient_ranking" in result.model_metadata
    ranking = result.model_metadata["coefficient_ranking"]
    assert isinstance(ranking, list)
    assert len(ranking) == len(result.model_metadata["feature_columns"])
    assert ranking[0]["abs_coefficient"] >= ranking[-1]["abs_coefficient"]


def test_linear_sklearn_model_rejects_unsupported_model_type() -> None:
    model = LinearSklearnModel()
    train_df, test_df = _synthetic_linear_frames()

    with pytest.raises(ValueError, match="Unsupported model type"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="lasso",
            target_column="target_ret_1d",
        )


def test_linear_sklearn_model_rejects_missing_numeric_features() -> None:
    model = LinearSklearnModel()
    train_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAA", "AAA"],
            "target_ret_1d": [0.1, 0.2],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03"]),
            "ticker": ["AAA"],
            "target_ret_1d": [0.3],
        }
    )

    with pytest.raises(ValueError, match="No numeric feature columns"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="ridge",
            target_column="target_ret_1d",
        )


def test_linear_sklearn_model_reports_supported_model_types() -> None:
    assert LinearSklearnModel().supported_model_types() == ("ridge",)

