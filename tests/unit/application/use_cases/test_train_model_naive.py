import numpy as np
import pandas as pd
import pytest

from finsight.application.use_cases.train_model import (
    _get_training_tickers,
    _parse_iso_date,
    _validate_model_types,
    _validate_supported_model_types,
)
from finsight.domain.metrics import SUPPORTED_METRIC_NAMES, METRIC_DIRECTION_ACCURACY
from finsight.infrastructure.features import TimeSplitPolicy
from finsight.infrastructure.ml.sklearn import NaiveBaselineModel


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


def test_naive_baseline_model_predictions_and_metrics() -> None:
    features_df = _synthetic_feature_frame()

    policy = TimeSplitPolicy(cutoff_date="2024-01-04", date_col="date", inclusive_test=True)
    train_df, test_df = policy.split_frame(features_df)

    assert train_df["date"].max().date().isoformat() == "2024-01-03"
    assert test_df["date"].min().date().isoformat() == "2024-01-04"

    model = NaiveBaselineModel()
    zero_metrics, zero_predictions = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="naive_zero",
        target_column="target_ret_1d",
    )
    mean_metrics, mean_predictions = model.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="naive_mean",
        target_column="target_ret_1d",
    )

    assert np.allclose(zero_predictions["y_pred"].to_numpy(), 0.0)

    expected_mean = float(train_df["target_ret_1d"].mean())
    assert np.allclose(mean_predictions["y_pred"].to_numpy(), expected_mean)

    for metrics, predictions in ((zero_metrics, zero_predictions), (mean_metrics, mean_predictions)):
        assert set(SUPPORTED_METRIC_NAMES).issubset(metrics.keys())
        assert 0.0 <= metrics[METRIC_DIRECTION_ACCURACY] <= 1.0
        assert list(predictions.columns) == ["date", "ticker", "y_true", "y_pred"]
        assert len(predictions) == len(test_df)


def test_naive_baseline_model_rejects_unsupported_model_type() -> None:
    features_df = _synthetic_feature_frame()
    train_df, test_df = TimeSplitPolicy("2024-01-04").split_frame(features_df)

    model = NaiveBaselineModel()

    with pytest.raises(ValueError, match="Unsupported model type"):
        model.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="naive_last",
            target_column="target_ret_1d",
        )


def test_naive_baseline_model_reports_supported_model_types() -> None:
    assert NaiveBaselineModel().supported_model_types() == ("naive_zero", "naive_mean")


def test_validate_model_types_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="model_types must be unique"):
        _validate_model_types(["naive_zero", "naive_zero"])


def test_validate_model_types_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one model type"):
        _validate_model_types([])


def test_validate_supported_model_types_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match=r"Unsupported model type\(s\)"):
        _validate_supported_model_types(["naive_last"], ("naive_zero", "naive_mean"))


def test_parse_iso_date_rejects_invalid_date() -> None:
    with pytest.raises(ValueError, match="Invalid ISO 8601 date"):
        _parse_iso_date("2026-99-01")


def test_get_training_tickers_normalizes_case_and_whitespace() -> None:
    assert _get_training_tickers([" aapl ", " msft"]) == ["AAPL", "MSFT"]


def test_get_training_tickers_rejects_blank_and_duplicates() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        _get_training_tickers(["   ", ""])

    with pytest.raises(ValueError, match="must not contain duplicates"):
        _get_training_tickers(["aapl", " AAPL "])
