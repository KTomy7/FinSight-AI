import numpy as np
import pandas as pd
import pytest

from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, SUPPORTED_METRIC_NAMES
from finsight.infrastructure.ml.sklearn.tree import HistGradientBoostingModel, RANDOM_STATE


def _synthetic_tree_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic training and test data with numeric features."""
    train_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "ret_1d": [0.1, 0.2, 0.3, 0.4],
            "mom_20d": [1.0, 2.0, 3.0, 4.0],
            "volatility": [0.05, 0.06, 0.07, 0.08],
            "target_ret_1d": [0.2, 0.4, 0.6, 0.8],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-06"]),
            "ticker": ["AAA", "AAA"],
            "ret_1d": [0.5, 0.6],
            "mom_20d": [5.0, 6.0],
            "volatility": [0.09, 0.10],
            "target_ret_1d": [1.0, 1.2],
        }
    )
    return train_df, test_df


def test_hist_gbdt_evaluates_and_returns_metrics() -> None:
    """Test basic evaluation produces valid metrics and predictions."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="hist_gbdt",
        target_column="target_ret_1d",
    )

    metrics = result.metrics
    predictions = result.predictions

    # Verify metrics contain expected keys
    assert set(SUPPORTED_METRIC_NAMES).issubset(metrics)
    assert 0.0 <= metrics[METRIC_DIRECTION_ACCURACY] <= 1.0

    # Verify prediction structure
    assert list(predictions.columns) == ["date", "ticker", "y_true", "y_pred"]
    assert len(predictions) == len(test_df)
    assert np.isfinite(predictions["y_pred"]).all()

    # Verify trained artifact
    assert result.trained_artifact.__class__.__name__ == "Pipeline"
    assert tuple(result.trained_artifact.named_steps.keys()) == ("scaler", "hist_gbdt")


def test_hist_gbdt_reports_supported_model_types() -> None:
    """Test that supported_model_types returns correct tuple."""
    model = HistGradientBoostingModel()
    assert model.supported_model_types() == ("hist_gbdt",)


def test_hist_gbdt_rejects_unsupported_model_type() -> None:
    """Test evaluation rejects unknown model type."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()

    with pytest.raises(ValueError, match="Unsupported model type"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="random_forest",  # Unsupported
            target_column="target_ret_1d",
        )


def test_hist_gbdt_rejects_missing_numeric_features() -> None:
    """Test evaluation fails when no numeric features are available."""
    model = HistGradientBoostingModel()
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
            model_type="hist_gbdt",
            target_column="target_ret_1d",
        )


def test_hist_gbdt_rejects_non_dataframe_train() -> None:
    """Test evaluation rejects non-DataFrame train input."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()

    with pytest.raises(TypeError, match="train_dataset must be a pandas DataFrame"):
        model.evaluate(
            train_dataset=[],  # Not a DataFrame
            test_dataset=test_df,
            model_type="hist_gbdt",
            target_column="target_ret_1d",
        )


def test_hist_gbdt_rejects_non_dataframe_test() -> None:
    """Test evaluation rejects non-DataFrame test input."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()

    with pytest.raises(TypeError, match="test_dataset must be a pandas DataFrame"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset={},  # Not a DataFrame
            model_type="hist_gbdt",
            target_column="target_ret_1d",
        )


def test_hist_gbdt_rejects_empty_train_dataset() -> None:
    """Test evaluation rejects empty training dataset."""
    model = HistGradientBoostingModel()
    # Empty DataFrame has no columns, so target validation happens first
    train_df = pd.DataFrame(columns=["date", "ticker", "ret_1d", "target_ret_1d"])
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "ticker": ["AAA"],
            "ret_1d": [0.5],
            "target_ret_1d": [1.0],
        }
    )

    with pytest.raises(ValueError, match="non-empty"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="hist_gbdt",
            target_column="target_ret_1d",
        )


def test_hist_gbdt_rejects_empty_test_dataset() -> None:
    """Test evaluation rejects empty test dataset."""
    model = HistGradientBoostingModel()
    train_df, _ = _synthetic_tree_frames()
    # Empty DataFrame with columns to pass target column check
    test_df = pd.DataFrame(columns=["date", "ticker", "ret_1d", "target_ret_1d"])

    with pytest.raises(ValueError, match="non-empty"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="hist_gbdt",
            target_column="target_ret_1d",
        )


def test_hist_gbdt_deterministic_with_random_state() -> None:
    """Test that same random_state produces deterministic results."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()

    # First evaluation
    result1 = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="hist_gbdt",
        target_column="target_ret_1d",
    )

    # Second evaluation with same data
    result2 = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="hist_gbdt",
        target_column="target_ret_1d",
    )

    # Predictions should be identical
    np.testing.assert_array_almost_equal(
        result1.predictions["y_pred"].values,
        result2.predictions["y_pred"].values,
        decimal=10,
    )

    # Metrics should be identical
    assert result1.metrics == result2.metrics


def test_hist_gbdt_metadata_contains_required_fields() -> None:
    """Test that model_metadata contains all required serializable fields."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="hist_gbdt",
        target_column="target_ret_1d",
    )

    metadata = result.model_metadata

    # Required fields
    assert metadata["adapter"] == "HistGradientBoostingModel"
    assert metadata["model_id"] == "hist_gbdt"
    assert metadata["estimator"] == "Pipeline"
    assert metadata["base_estimator"] == "HistGradientBoostingRegressor"
    assert isinstance(metadata["feature_columns"], list)
    assert metadata["n_features"] == len(metadata["feature_columns"])
    assert isinstance(metadata["hyperparams"], dict)
    assert metadata["hyperparams"]["random_state"] == RANDOM_STATE
    assert metadata["preprocessing"]["scaler"] == "StandardScaler"
    assert isinstance(metadata["feature_importance"], dict)
    assert isinstance(metadata["feature_importance_ranking"], list)


def test_hist_gbdt_feature_importance_ranking_ordered() -> None:
    """Test that feature importance ranking is ordered by importance descending."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="hist_gbdt",
        target_column="target_ret_1d",
    )

    ranking = result.model_metadata["feature_importance_ranking"]

    # Verify structure
    assert len(ranking) == result.model_metadata["n_features"]
    for item in ranking:
        assert "feature" in item
        assert "importance" in item
        assert "rank" in item
        assert isinstance(item["importance"], float)
        assert item["importance"] >= 0.0

    # Verify ordering (descending by importance)
    for i in range(len(ranking) - 1):
        assert ranking[i]["importance"] >= ranking[i + 1]["importance"]


def test_hist_gbdt_predictions_row_count_matches_test() -> None:
    """Test that predictions DataFrame has same row count as test set."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()

    result = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="hist_gbdt",
        target_column="target_ret_1d",
    )

    assert len(result.predictions) == len(test_df)
    assert result.predictions["y_true"].notna().all()
    assert result.predictions["y_pred"].notna().all()


def test_hist_gbdt_missing_target_column_in_train() -> None:
    """Test evaluation fails when target column is missing from train set."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()
    train_df = train_df.drop(columns=["target_ret_1d"])

    with pytest.raises(ValueError, match="must contain"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="hist_gbdt",
            target_column="target_ret_1d",
        )


def test_hist_gbdt_missing_target_column_in_test() -> None:
    """Test evaluation fails when target column is missing from test set."""
    model = HistGradientBoostingModel()
    train_df, test_df = _synthetic_tree_frames()
    test_df = test_df.drop(columns=["target_ret_1d"])

    with pytest.raises(ValueError, match="must contain"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="hist_gbdt",
            target_column="target_ret_1d",
        )
