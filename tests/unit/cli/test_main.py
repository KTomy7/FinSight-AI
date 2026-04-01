import pytest

import finsight.cli.main as cli_main
from finsight.cli.main import _build_parser, _run_train
from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_RMSE


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


