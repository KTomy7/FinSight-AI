from __future__ import annotations

from finsight.application.dto import (
    BacktestFoldSummary,
    BacktestReport,
    BacktestRequest,
    BacktestResult,
    CompareModelsRequest,
    CompareModelsResult,
    DatasetSpec,
    FeatureSpec,
    FetchMarketDataRequest,
    ForecastRequest,
    ForecastResult,
    ModelComparisonRow,
    TrainModelRequest,
    TrainModelResult,
)


def test_train_model_request_roundtrip() -> None:
    request = TrainModelRequest(
        cutoff_date="2025-06-01",
        years=2,
        end="2026-03-17",
        interval="1d",
        model_types=["naive_zero", "naive_mean"],
        artifacts_dir="artifacts/runs",
    )

    payload = request.to_dict()
    restored = TrainModelRequest.from_dict(payload)

    assert restored == request


def test_fetch_market_data_request_roundtrip() -> None:
    request = FetchMarketDataRequest(
        ticker="AAPL",
        start_date="2026-01-01",
        end_date="2026-03-31",
        interval="1d",
        include_summary=False,
    )

    payload = request.to_dict()
    restored = FetchMarketDataRequest.from_dict(payload)

    assert restored == request


def test_fetch_market_data_request_ticker_none_and_nonstring() -> None:
    # None becomes ""
    req1 = FetchMarketDataRequest.from_dict({"ticker": None})
    assert req1.ticker == ""
    # int/float become ""
    req2 = FetchMarketDataRequest.from_dict({"ticker": 123})
    assert req2.ticker == ""
    req3 = FetchMarketDataRequest.from_dict({"ticker": 12.5})
    assert req3.ticker == ""
    # whitespace is stripped
    req4 = FetchMarketDataRequest.from_dict({"ticker": "  AAPL  "})
    assert req4.ticker == "AAPL"
    # missing ticker is ""
    req5 = FetchMarketDataRequest.from_dict({})
    assert req5.ticker == ""


def test_train_model_result_roundtrip_with_specs() -> None:
    result = TrainModelResult(
        run_dirs={"naive_zero": "artifacts/runs/2026-03-31T123000Z__naive_zero"},
        metrics={"naive_zero": {"mae": 0.1, "rmse": 0.2, "n_train": 100, "window": "2y"}},
        dataset_spec=DatasetSpec(
            tickers=("AAPL", "JPM"),
            start_date="2024-03-18",
            end_date="2026-03-17",
            interval="1d",
        ),
        feature_spec=FeatureSpec(
            feature_columns=("ret_1d", "mom_20d"),
            target_column="target_ret_1d",
        ),
    )

    payload = result.to_dict()
    restored = TrainModelResult.from_dict(payload)

    assert restored == result


def test_compare_models_request_and_result_roundtrip() -> None:
    request = CompareModelsRequest(
        model_ids=["naive_zero", "ridge"],
        artifacts_dir="artifacts/runs",
        rank_by=["mae", "r2", "direction_accuracy"],
        metric_directions={"r2": "desc", "direction_accuracy": "desc"},
        use_best_runs=True,
    )
    result = CompareModelsResult(
        rows=[
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-12T120000Z__ridge",
                metrics={"mae": 0.09, "r2": 0.92, "direction_accuracy": 0.87},
                sort_key=(0.09, -0.92, -0.87, "ridge", "2026-04-12T120000Z__ridge"),
            )
        ],
        rank_by=["mae", "r2", "direction_accuracy"],
        metric_directions={"mae": "asc", "r2": "desc", "direction_accuracy": "desc"},
    )

    assert CompareModelsRequest.from_dict(request.to_dict()) == request
    assert CompareModelsResult.from_dict(result.to_dict()) == result


def test_compare_models_request_defaults_include_r2_priority() -> None:
    request = CompareModelsRequest(model_ids=["naive_zero"])

    assert request.rank_by == ["mae", "rmse", "r2", "direction_accuracy"]


def test_forecast_and_backtest_results_are_serializable() -> None:
    forecast = ForecastResult(
        model_id="naive_mean",
        ticker="AAPL",
        horizon_days=7,
        predictions=[
            {"date": "2026-04-01", "y_pred": 0.01},
            {"date": "2026-04-02", "y_pred": 0.012},
        ],
        generated_at="2026-03-31T12:30:00Z",
    )
    backtest = BacktestResult(
        model_id="naive_mean",
        metrics={"mae": 0.1, "rmse": 0.2},
        folds=[{"fold": 1, "mae": 0.1}, {"fold": 2, "mae": 0.11}],
    )

    forecast_payload = forecast.to_dict()
    backtest_payload = backtest.to_dict()

    assert ForecastResult.from_dict(forecast_payload) == forecast
    assert BacktestResult.from_dict(backtest_payload) == backtest


def test_backtest_request_roundtrip() -> None:
    request = BacktestRequest(
        model_ids=["naive_zero", "ridge"],
        years=3,
        end="2026-03-31",
        interval="1d",
        min_train_days=200,
        test_window_days=30,
        step_days=15,
        max_folds=6,
    )

    assert BacktestRequest.from_dict(request.to_dict()) == request


def test_backtest_report_roundtrip() -> None:
    fold = BacktestFoldSummary(
        fold_index=1,
        train_start="2024-01-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-01-31",
        n_train=252,
        n_test=21,
        metrics={"mae": 0.12, "rmse": 0.18},
    )
    report = BacktestReport(
        results=[
            BacktestResult(
                model_id="ridge",
                metrics={"mae": 0.11, "rmse": 0.17},
                folds=[fold.to_dict()],
            )
        ],
        dataset_spec=DatasetSpec(
            tickers=("AAPL", "JPM"),
            start_date="2024-01-01",
            end_date="2026-03-31",
            interval="1d",
        ),
        split_spec={"min_train_days": 252, "test_window_days": 21, "step_days": 21},
    )

    assert BacktestReport.from_dict(report.to_dict()) == report


def test_train_model_result_is_constructible() -> None:
    result = TrainModelResult(run_dirs={}, metrics={})
    assert isinstance(result, TrainModelResult)


def test_from_dict_handles_non_sequence_fields_without_char_splitting() -> None:
    dataset = DatasetSpec.from_dict({"tickers": "AAPL", "interval": "1d"})
    features = FeatureSpec.from_dict({"feature_columns": "ret_1d", "target_column": "target_ret_1d"})

    assert dataset.tickers == ()
    assert features.feature_columns == ()


def test_forecast_request_roundtrip() -> None:
    request = ForecastRequest(
        ticker="AAPL",
        model_id="ridge",
        horizon_days=14,
        artifacts_dir="artifacts/runs",
    )

    payload = request.to_dict()
    restored = ForecastRequest.from_dict(payload)

    assert restored == request


def test_forecast_request_from_dict_normalizes_strings_and_defaults_artifacts_dir() -> None:
    request = ForecastRequest.from_dict(
        {
            "ticker": "  AAPL  ",
            "model_id": "  ridge  ",
            "horizon_days": "7",
            "artifacts_dir": "   ",
        }
    )

    assert request.ticker == "AAPL"
    assert request.model_id == "ridge"
    assert request.horizon_days == 7
    assert request.artifacts_dir == "artifacts/runs"


def test_from_dict_parses_safe_defaults_for_invalid_scalar_types() -> None:
    train_request = TrainModelRequest.from_dict({"years": "bad", "model_types": "naive_zero"})
    forecast_request = ForecastRequest.from_dict({"horizon_days": "bad", "artifacts_dir": None})
    forecast = ForecastResult.from_dict({"horizon_days": "bad", "generated_at": 123})

    assert train_request.years == 2
    assert train_request.model_types == ["naive_zero", "naive_mean"]
    assert forecast_request.horizon_days == 0
    assert forecast_request.artifacts_dir == "artifacts/runs"
    assert forecast.horizon_days == 0
    assert forecast.generated_at == "123"


def test_forecast_result_from_dict_normalizes_identifiers() -> None:
    forecast = ForecastResult.from_dict({"model_id": None, "ticker": "  AAPL  "})
    assert forecast.model_id == ""
    assert forecast.ticker == "AAPL"

    forecast_nonstring = ForecastResult.from_dict({"model_id": 123, "ticker": None})
    assert forecast_nonstring.model_id == ""
    assert forecast_nonstring.ticker == ""

