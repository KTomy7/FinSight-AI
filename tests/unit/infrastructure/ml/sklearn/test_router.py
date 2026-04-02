import pandas as pd
import pytest

from finsight.infrastructure.ml.sklearn import LinearSklearnModel, NaiveBaselineModel, SklearnModelRouter


def _synthetic_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "AAA", "AAA"],
            "target_ret_1d": [0.1, -0.1, 0.2],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "ticker": ["AAA", "AAA"],
            "target_ret_1d": [0.05, -0.02],
        }
    )
    return train_df, test_df


def test_router_reports_supported_model_types_from_adapters() -> None:
    router = SklearnModelRouter(adapters=[NaiveBaselineModel()])

    assert router.supported_model_types() == ("naive_zero", "naive_mean")


def test_router_reports_supported_model_types_from_multiple_adapters() -> None:
    router = SklearnModelRouter(adapters=[NaiveBaselineModel(), LinearSklearnModel()])

    assert router.supported_model_types() == ("naive_zero", "naive_mean", "ridge")


def test_router_delegates_to_matching_adapter() -> None:
    router = SklearnModelRouter(adapters=[NaiveBaselineModel()])
    train_df, test_df = _synthetic_train_test()

    result = router.evaluate(
        train_dataset=train_df,
        test_dataset=test_df,
        model_type="naive_zero",
        target_column="target_ret_1d",
    )

    metrics = result.metrics
    predictions = result.predictions
    assert "mae" in metrics
    assert list(predictions.columns) == ["date", "ticker", "y_true", "y_pred"]
    assert len(predictions) == len(test_df)


def test_router_rejects_unknown_model_type() -> None:
    router = SklearnModelRouter(adapters=[NaiveBaselineModel()])
    train_df, test_df = _synthetic_train_test()

    with pytest.raises(ValueError, match="Unsupported model type"):
        router.evaluate(
            train_dataset=train_df,
            test_dataset=test_df,
            model_type="ridge",
            target_column="target_ret_1d",
        )


def test_router_rejects_duplicate_model_ids_across_adapters() -> None:
    class _DuplicateNaiveAdapter(NaiveBaselineModel):
        def supported_model_types(self) -> tuple[str, ...]:
            return ("naive_zero",)

    with pytest.raises(ValueError, match="Duplicate model id"):
        SklearnModelRouter(adapters=[NaiveBaselineModel(), _DuplicateNaiveAdapter()])


