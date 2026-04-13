import pytest

import finsight.cli.main as cli_main
from finsight.cli.main import _build_parser, _run_train
from finsight.application.dto import CompareModelsResult, ModelComparisonRow
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
    assert "naive_zero" in captured
    assert "2026-04-12T120000Z__naive_zero" in captured
    assert "mae" in captured


