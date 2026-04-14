import pytest
import json

import finsight.cli.main as cli_main
from finsight.cli.main import _build_parser, _run_train
from finsight.application.dto import CompareModelsResult, ForecastResult, ModelComparisonRow
from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_RMSE
from finsight.config.settings import ModelCatalogEntry, ModelDefaults, Settings, TickerCatalogSettings


def test_train_parser_accepts_train_without_tickers_arg() -> None:
    parser = _build_parser()

    args = parser.parse_args(["train", "--cutoff", "2025-06-01"])

    assert args.command == "train"
    assert args.cutoff == "2025-06-01"


def test_train_parser_rejects_tickers_arg() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--cutoff", "2025-06-01", "--tickers", "AAPL", "MSFT"])


def test_train_parser_rejects_interval_arg() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--cutoff", "2025-06-01", "--interval", "1d"])


def test_compare_parser_accepts_compare_model_ids_and_rank_by() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        ["compare", "--model-ids", "naive_zero", "ridge", "--rank-by", "mae", "direction_accuracy"]
    )

    assert args.command == "compare"
    assert args.model_ids == ["naive_zero", "ridge"]
    assert args.rank_by == ["mae", "direction_accuracy"]


def test_forecast_parser_accepts_required_args() -> None:
    parser = _build_parser()

    args = parser.parse_args(["forecast", "--ticker", "AAPL", "--model-id", "ridge", "--horizon", "30"])

    assert args.command == "forecast"
    assert args.ticker == "AAPL"
    assert args.model_id == "ridge"
    assert args.horizon == 30
    assert args.artifacts_dir == "artifacts/runs"
    assert args.as_json is False


def test_forecast_parser_accepts_json_flag() -> None:
    parser = _build_parser()

    args = parser.parse_args(["forecast", "--ticker", "AAPL", "--model-id", "ridge", "--horizon", "30", "--json"])

    assert args.as_json is True


def test_main_dispatches_forecast_command_to_run_forecast(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "_run_forecast", lambda _args: 7)
    monkeypatch.setattr("sys.argv", ["finsight", "forecast", "--ticker", "AAPL", "--model-id", "ridge", "--horizon", "5"])

    assert cli_main.main() == 7


def test_run_forecast_prints_prediction_rows(monkeypatch, capsys) -> None:
    class _StubForecast:
        last_request = None

        @classmethod
        def execute(cls, request):
            cls.last_request = request
            return ForecastResult(
                model_id="ridge",
                ticker="AAPL",
                horizon_days=2,
                predictions=[
                    {"date": "2026-04-13", "pred_ret_1d": 0.1, "pred_close": 119.9},
                    {"date": "2026-04-14", "pred_ret_1d": -0.05, "pred_close": 113.905},
                ],
                generated_at="2026-04-12T12:00:00Z",
            )

    class _StubContainer:
        forecast = _StubForecast()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())

    args = _build_parser().parse_args(["forecast", "--ticker", "AAPL", "--model-id", "ridge", "--horizon", "2"])
    exit_code = cli_main._run_forecast(args)

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert _StubForecast.last_request.ticker == "AAPL"
    assert _StubForecast.last_request.model_id == "ridge"
    assert _StubForecast.last_request.horizon_days == 2
    assert _StubForecast.last_request.artifacts_dir == "artifacts/runs"
    assert "[ridge] ticker=AAPL horizon_days=2 rows=2" in captured
    assert "2026-04-13 pred_ret_1d=0.1 pred_close=119.9" in captured
    assert "2026-04-14 pred_ret_1d=-0.05 pred_close=113.905" in captured


def test_run_forecast_prints_json_when_requested(monkeypatch, capsys) -> None:
    class _StubForecast:
        @staticmethod
        def execute(_request):
            return ForecastResult(
                model_id="ridge",
                ticker="AAPL",
                horizon_days=1,
                predictions=[{"date": "2026-04-13", "pred_ret_1d": 0.1, "pred_close": 119.9}],
                generated_at="2026-04-12T12:00:00Z",
            )

    class _StubContainer:
        forecast = _StubForecast()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())

    args = _build_parser().parse_args(["forecast", "--ticker", "AAPL", "--model-id", "ridge", "--horizon", "1", "--json"])
    exit_code = cli_main._run_forecast(args)

    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert exit_code == 0
    assert payload["model_id"] == "ridge"
    assert payload["ticker"] == "AAPL"
    assert payload["horizon_days"] == 1
    assert payload["predictions"][0]["pred_ret_1d"] == 0.1
    assert payload["predictions"][0]["pred_close"] == 119.9


def test_run_forecast_handles_empty_ticker_validation_error(monkeypatch, capsys) -> None:
    class _StubForecast:
        @staticmethod
        def execute(_request):
            raise ValueError("ticker must be a non-empty string.")

    class _StubContainer:
        forecast = _StubForecast()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())

    args = _build_parser().parse_args(["forecast", "--ticker", "", "--model-id", "ridge", "--horizon", "5"])
    exit_code = cli_main._run_forecast(args)

    captured_err = capsys.readouterr().err
    assert exit_code == 1
    assert "Forecast validation error" in captured_err
    assert "ticker must be a non-empty string" in captured_err


def test_run_forecast_handles_invalid_horizon_validation_error(monkeypatch, capsys) -> None:
    class _StubForecast:
        @staticmethod
        def execute(_request):
            raise ValueError("horizon_days must be a positive integer.")

    class _StubContainer:
        forecast = _StubForecast()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())

    args = _build_parser().parse_args(["forecast", "--ticker", "AAPL", "--model-id", "ridge", "--horizon", "0"])
    exit_code = cli_main._run_forecast(args)

    captured_err = capsys.readouterr().err
    assert exit_code == 1
    assert "Forecast validation error" in captured_err
    assert "horizon_days must be a positive integer" in captured_err


def test_run_forecast_handles_missing_run_file_not_found_error(monkeypatch, capsys) -> None:
    class _StubForecast:
        @staticmethod
        def execute(_request):
            raise FileNotFoundError("No runs found for model_id 'unknown_model' under artifact root: artifacts/runs")

    class _StubContainer:
        forecast = _StubForecast()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())

    args = _build_parser().parse_args(["forecast", "--ticker", "AAPL", "--model-id", "unknown_model", "--horizon", "5"])
    exit_code = cli_main._run_forecast(args)

    captured_err = capsys.readouterr().err
    assert exit_code == 1
    assert "Forecast artifact error" in captured_err
    assert "No runs found for model_id" in captured_err


def test_run_forecast_handles_model_type_error(monkeypatch, capsys) -> None:
    class _StubForecast:
        @staticmethod
        def execute(_request):
            raise TypeError("Loaded model artifact does not implement predict(...).")

    class _StubContainer:
        forecast = _StubForecast()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())

    args = _build_parser().parse_args(["forecast", "--ticker", "AAPL", "--model-id", "ridge", "--horizon", "5"])
    exit_code = cli_main._run_forecast(args)

    captured_err = capsys.readouterr().err
    assert exit_code == 1
    assert "Forecast runtime error" in captured_err
    assert "does not implement predict" in captured_err


def test_run_forecast_handles_unexpected_exception(monkeypatch, capsys) -> None:
    class _StubForecast:
        @staticmethod
        def execute(_request):
            raise RuntimeError("Unexpected internal error occurred.")

    class _StubContainer:
        forecast = _StubForecast()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())

    args = _build_parser().parse_args(["forecast", "--ticker", "AAPL", "--model-id", "ridge", "--horizon", "5"])
    exit_code = cli_main._run_forecast(args)

    captured_err = capsys.readouterr().err
    assert exit_code == 1
    assert "Forecast unexpected error" in captured_err
    assert "Unexpected internal error occurred" in captured_err


def test_run_train_prints_metrics_using_canonical_metric_keys(monkeypatch, capsys) -> None:
    class _StubTrainModel:
        @staticmethod
        def execute(_request):
            return type(
                "Response",
                (),
                {
                    "run_dirs": {"naive_zero": "artifacts/runs/example"},
                    "metrics": {
                        "naive_zero": {
                            METRIC_MAE: 0.1,
                            METRIC_RMSE: 0.2,
                            METRIC_DIRECTION_ACCURACY: 0.75,
                        }
                    },
                },
            )()

    class _StubContainer:
        train_model = _StubTrainModel()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())

    args = _build_parser().parse_args(["train", "--cutoff", "2025-06-01"])
    exit_code = _run_train(args)

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "[naive_zero] run_dir=artifacts/runs/example" in captured
    assert "MAE=0.100000 RMSE=0.200000 DirectionAcc=0.7500" in captured


def test_run_compare_prints_leaderboard_table(monkeypatch, capsys) -> None:
    class _StubCompareModels:
        @staticmethod
        def execute(_request):
            return CompareModelsResult(
                rows=[
                    ModelComparisonRow(
                        rank=1,
                        model_id="naive_zero",
                        run_id="2026-04-12T120000Z__naive_zero",
                        metrics={METRIC_MAE: 0.1, METRIC_RMSE: 0.2, METRIC_DIRECTION_ACCURACY: 0.75},
                        sort_key=(0.1, 0.2, -0.75, "naive_zero", "2026-04-12T120000Z__naive_zero"),
                    )
                ],
                rank_by=[METRIC_MAE, METRIC_RMSE, METRIC_DIRECTION_ACCURACY],
                metric_directions={METRIC_MAE: "asc", METRIC_RMSE: "asc", METRIC_DIRECTION_ACCURACY: "desc"},
            )

    class _StubContainer:
        compare_models = _StubCompareModels()

    monkeypatch.setattr(cli_main, "build_container", lambda: _StubContainer())
    monkeypatch.setattr(
        cli_main,
        "get_settings",
        lambda: Settings(
            model_defaults=ModelDefaults(
                catalog=(
                    ModelCatalogEntry(id="naive_zero", label="Naive (Zero)", supports_training=True, supports_prediction=True),
                ),
                default_model_id="naive_zero",
            ),
            ticker_catalog=TickerCatalogSettings(entries=()),
        ),
    )

    args = _build_parser().parse_args(["compare", "--model-ids", "naive_zero"])
    exit_code = cli_main._run_compare(args)

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Naive (Zero)" in captured
    assert "mae" in captured


