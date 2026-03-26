import numpy as np
import pandas as pd
import pytest

from finsight.application.use_cases.train_model import evaluate_naive_models
from finsight.infrastructure.features import TimeSplitPolicy


def _synthetic_feature_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ]
    )
    rows = []
    targets_a = [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
    targets_b = [-0.02, 0.01, 0.00, -0.01, 0.02, -0.03]

    for ticker, targets in (("AAA", targets_a), ("BBB", targets_b)):
        for idx, day in enumerate(dates):
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "ret_1d": float(idx) / 100.0,
                    "mom_20d": float(idx) / 50.0,
                    "target_ret_1d": targets[idx],
                }
            )

    return pd.DataFrame(rows)


def test_evaluate_naive_models_predictions_and_metrics() -> None:
    features_df = _synthetic_feature_frame()

    policy = TimeSplitPolicy(cutoff_date="2024-01-04", date_col="date", inclusive_test=True)
    train_df, test_df = policy.split_frame(features_df)

    assert train_df["date"].max().date().isoformat() == "2024-01-03"
    assert test_df["date"].min().date().isoformat() == "2024-01-04"

    metrics, predictions = evaluate_naive_models(
        train_df,
        test_df,
        ["naive_zero", "naive_mean"],
    )

    assert np.allclose(predictions["naive_zero"]["y_pred"].to_numpy(), 0.0)

    expected_mean = float(train_df["target_ret_1d"].mean())
    assert np.allclose(predictions["naive_mean"]["y_pred"].to_numpy(), expected_mean)

    for model_type in ("naive_zero", "naive_mean"):
        assert {"mae", "rmse", "direction_accuracy"}.issubset(metrics[model_type].keys())
        assert 0.0 <= metrics[model_type]["direction_accuracy"] <= 1.0
        assert list(predictions[model_type].columns) == ["date", "ticker", "y_true", "y_pred"]
        assert len(predictions[model_type]) == len(test_df)


def test_evaluate_naive_models_rejects_unsupported_model_type() -> None:
    features_df = _synthetic_feature_frame()
    train_df, test_df = TimeSplitPolicy("2024-01-04").split_frame(features_df)

    with pytest.raises(ValueError, match="Unsupported model type"):
        evaluate_naive_models(train_df, test_df, ["naive_last"])

