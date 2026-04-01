from __future__ import annotations

from finsight.application.dto import (
    BacktestResult,
    DatasetSpec,
    FeatureSpec,
    FetchMarketDataRequest,
    ForecastResult,
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


def test_train_model_result_is_constructible() -> None:
    result = TrainModelResult(run_dirs={}, metrics={})
    assert isinstance(result, TrainModelResult)


def test_from_dict_handles_non_sequence_fields_without_char_splitting() -> None:
    dataset = DatasetSpec.from_dict({"tickers": "AAPL", "interval": "1d"})
    features = FeatureSpec.from_dict({"feature_columns": "ret_1d", "target_column": "target_ret_1d"})

    assert dataset.tickers == ()
    assert features.feature_columns == ()


def test_from_dict_parses_safe_defaults_for_invalid_scalar_types() -> None:
    train_request = TrainModelRequest.from_dict({"years": "bad", "model_types": "naive_zero"})
    forecast = ForecastResult.from_dict({"horizon_days": "bad", "generated_at": 123})

    assert train_request.years == 2
    assert train_request.model_types == ["naive_zero", "naive_mean"]
    assert forecast.horizon_days == 0
    assert forecast.generated_at == "123"


def test_forecast_result_from_dict_normalizes_identifiers() -> None:
    forecast = ForecastResult.from_dict({"model_id": None, "ticker": "  AAPL  "})
    assert forecast.model_id == ""
    assert forecast.ticker == "AAPL"

    forecast_nonstring = ForecastResult.from_dict({"model_id": 123, "ticker": None})
    assert forecast_nonstring.model_id == ""
    assert forecast_nonstring.ticker == ""


