import numpy as np
import pandas as pd
import pytest

from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, SUPPORTED_METRIC_NAMES
from finsight.infrastructure.ml.sklearn.xgboost import XGBoostModel


def _synthetic_xgboost_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic training and test data for XGBoost testing."""
    train_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "ticker": ["AAA", "AAA", "AAA", "AAA", "AAA"],
            "ret_1d": [0.1, 0.2, 0.3, 0.4, 0.5],
            "mom_20d": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volatility": [0.5, 0.6, 0.7, 0.8, 0.9],
            "target_ret_1d": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-08"]),
            "ticker": ["AAA", "AAA", "AAA"],
            "ret_1d": [0.6, 0.7, 0.8],
            "mom_20d": [6.0, 7.0, 8.0],
            "volatility": [1.0, 1.1, 1.2],
            "target_ret_1d": [1.2, 1.4, 1.6],
        }
    )
    return train_df, test_df


def test_xgboost_smoke_predicts_and_reports_metrics() -> None:
    """Test that XGBoost model fits, predicts, and returns valid metrics."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="xgboost",
        target_column="target_ret_1d",
    )

    metrics = result.metrics
    predictions = result.predictions

    # Verify all required metrics are present
    assert set(SUPPORTED_METRIC_NAMES).issubset(metrics)
    assert 0.0 <= metrics[METRIC_DIRECTION_ACCURACY] <= 1.0

    # Verify predictions structure
    assert list(predictions.columns) == ["date", "ticker", "y_true", "y_pred"]
    assert len(predictions) == len(test_df)
    assert np.isfinite(predictions["y_pred"]).all()

    # Verify trained artifact
    assert result.trained_artifact.__class__.__name__ == "XGBRegressor"

    # Verify model metadata
    assert result.model_metadata["model_id"] == "xgboost"
    assert result.model_metadata["base_estimator"] == "XGBRegressor"
    assert "hyperparams" in result.model_metadata
    assert "feature_columns" in result.model_metadata


def test_xgboost_metrics_has_required_keys() -> None:
    """Test that all required metrics are computed."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="xgboost",
        target_column="target_ret_1d",
    )

    metrics = result.metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "direction_accuracy" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_xgboost_predictions_shape_and_columns() -> None:
    """Test that predictions have correct shape and columns."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="xgboost",
        target_column="target_ret_1d",
    )

    predictions = result.predictions

    # Check shape
    assert predictions.shape[0] == len(test_df)
    assert predictions.shape[1] == 4  # date, ticker, y_true, y_pred

    # Check columns exist
    assert "date" in predictions.columns
    assert "ticker" in predictions.columns
    assert "y_true" in predictions.columns
    assert "y_pred" in predictions.columns

    # Check y_true matches test targets
    np.testing.assert_array_almost_equal(
        predictions["y_true"].values,
        test_df["target_ret_1d"].values,
    )


def test_xgboost_artifact_is_fitted_estimator() -> None:
    """Test that trained artifact is a fitted XGBRegressor."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="xgboost",
        target_column="target_ret_1d",
    )

    artifact = result.trained_artifact

    # Verify it's an XGBRegressor
    assert artifact.__class__.__name__ == "XGBRegressor"

    # Verify it has the predict method
    assert hasattr(artifact, "predict")
    assert callable(artifact.predict)

    # Verify it has feature_importances_
    assert hasattr(artifact, "feature_importances_")
    assert artifact.feature_importances_ is not None


def test_xgboost_feature_importance_present() -> None:
    """Test that feature importance is computed and ranked."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="xgboost",
        target_column="target_ret_1d",
    )

    metadata = result.model_metadata

    # Check importance dicts exist
    assert "feature_importance" in metadata
    assert "feature_importance_ranking" in metadata

    # Check feature_importance is a dict
    importance_dict = metadata["feature_importance"]
    assert isinstance(importance_dict, dict)
    assert len(importance_dict) > 0

    # Check all features are present
    expected_features = {"ret_1d", "mom_20d", "volatility"}
    assert set(importance_dict.keys()) == expected_features

    # Check feature_importance_ranking is a list
    ranking = metadata["feature_importance_ranking"]
    assert isinstance(ranking, list)
    assert len(ranking) == 3

    # Check ranking structure
    for i, item in enumerate(ranking):
        assert "feature" in item
        assert "importance" in item
        assert "rank" in item
        assert item["rank"] == i + 1

    # Check ranking is sorted by importance (descending)
    for i in range(len(ranking) - 1):
        assert ranking[i]["importance"] >= ranking[i + 1]["importance"]


def test_xgboost_metadata_includes_hyperparams() -> None:
    """Test that model metadata includes hyperparameters."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="xgboost",
        target_column="target_ret_1d",
    )

    metadata = result.model_metadata
    hyperparams = metadata["hyperparams"]

    # Check key hyperparameters are present
    assert "n_estimators" in hyperparams
    assert "learning_rate" in hyperparams
    assert "max_depth" in hyperparams
    assert "subsample" in hyperparams
    assert "colsample_bytree" in hyperparams
    assert "reg_lambda" in hyperparams
    assert "random_state" in hyperparams
    assert "n_jobs" in hyperparams

    # Check values match expected defaults
    assert hyperparams["n_estimators"] == 300
    assert hyperparams["learning_rate"] == 0.03
    assert hyperparams["max_depth"] == 3
    assert hyperparams["subsample"] == 0.8
    assert hyperparams["colsample_bytree"] == 0.8
    assert hyperparams["reg_lambda"] == 1.0
    assert hyperparams["random_state"] == 42
    assert hyperparams["n_jobs"] == 1


def test_xgboost_rejects_unsupported_model_type() -> None:
    """Test that unsupported model types raise ValueError."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    with pytest.raises(ValueError, match="Unsupported model type"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="some_other_model",
            target_column="target_ret_1d",
        )


def test_xgboost_rejects_missing_numeric_features() -> None:
    """Test that missing numeric features raise ValueError."""
    model = XGBoostModel()
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
            model_type="xgboost",
            target_column="target_ret_1d",
        )


def test_xgboost_rejects_missing_target_column() -> None:
    """Test that missing target column raises ValueError."""
    model = XGBoostModel()
    train_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAA", "AAA"],
            "ret_1d": [0.1, 0.2],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03"]),
            "ticker": ["AAA"],
            "ret_1d": [0.3],
        }
    )

    with pytest.raises(ValueError, match="must contain"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="xgboost",
            target_column="target_ret_1d",
        )


def test_xgboost_rejects_non_dataframe_input() -> None:
    """Test that non-DataFrame inputs raise TypeError."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        model.evaluate(
            train_dataset=[],  # Not a DataFrame
            test_dataset=test_df,
            model_type="xgboost",
            target_column="target_ret_1d",
        )


def test_xgboost_handles_custom_id_columns() -> None:
    """Test that custom id_columns are respected."""
    model = XGBoostModel()
    train_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "ret_1d": [0.1, 0.2, 0.3],
            "mom_20d": [1.0, 2.0, 3.0],
            "target_ret_1d": [0.2, 0.4, 0.6],
        }
    )
    test_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "symbol": ["AAA", "AAA"],
            "ret_1d": [0.4, 0.5],
            "mom_20d": [4.0, 5.0],
            "target_ret_1d": [0.8, 1.0],
        }
    )

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="xgboost",
        target_column="target_ret_1d",
        id_columns=("timestamp", "symbol"),
    )

    predictions = result.predictions
    assert "timestamp" in predictions.columns
    assert "symbol" in predictions.columns


def test_xgboost_supported_model_types() -> None:
    """Test that supported model types are correctly reported."""
    model = XGBoostModel()
    assert model.supported_model_types() == ("xgboost",)


def test_xgboost_produces_reasonable_predictions() -> None:
    """Test that XGBoost produces numerically reasonable predictions."""
    model = XGBoostModel()
    train_df, test_df = _synthetic_xgboost_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="xgboost",
        target_column="target_ret_1d",
    )

    predictions = result.predictions

    # Check no NaN or infinite values
    assert not predictions["y_pred"].isna().any()
    assert np.isfinite(predictions["y_pred"]).all()

    # Check predictions are in a reasonable range (close to training targets)
    y_train = train_df["target_ret_1d"]
    y_pred = predictions["y_pred"]

    assert y_pred.min() >= y_train.min() - 2 * y_train.std()
    assert y_pred.max() <= y_train.max() + 2 * y_train.std()

