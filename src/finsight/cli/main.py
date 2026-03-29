from __future__ import annotations

import argparse

from finsight.application.use_cases.train_model import TrainModelRequest
from finsight.bootstrap.container import build_container
from finsight.config.settings import get_settings
from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_RMSE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finsight")
    subparsers = parser.add_subparsers(dest="command", required=True)

    settings = get_settings()
    tickers = ", ".join(settings.ticker_catalog.symbols())
    training_model_ids = list(settings.model_defaults.training_model_ids())
    train_parser = subparsers.add_parser(
        "train",
        help=f"Train/evaluate baseline models on fixed tickers: {tickers}",
    )
    train_parser.add_argument("--years", type=int, default=2, help="Lookback window in years")
    train_parser.add_argument("--end", default=None, help="Inclusive end date (YYYY-MM-DD)")
    train_parser.add_argument("--cutoff", required=True, help="Global time split cutoff date (YYYY-MM-DD)")
    train_parser.add_argument(
        "--model-types",
        nargs="+",
        default=training_model_ids,
        help=f"Model IDs to evaluate (defaults: {', '.join(training_model_ids)})",
    )
    train_parser.add_argument(
        "--artifacts-dir",
        default="artifacts/runs",
        help="Directory for run artifacts",
    )

    return parser


def _run_train(args: argparse.Namespace) -> int:
    container = build_container()
    response = container.train_model.execute(
        TrainModelRequest(
            years=args.years,
            end=args.end,
            cutoff_date=args.cutoff,
            model_types=args.model_types,
            artifacts_dir=args.artifacts_dir,
        )
    )

    for model_type, run_dir in response.run_dirs.items():
        model_metrics = response.metrics[model_type]
        print(f"[{model_type}] run_dir={run_dir}")
        print(
            "  "
            f"MAE={float(model_metrics[METRIC_MAE]):.6f} "
            f"RMSE={float(model_metrics[METRIC_RMSE]):.6f} "
            f"DirectionAcc={float(model_metrics[METRIC_DIRECTION_ACCURACY]):.4f}"
        )

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        return _run_train(args)

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


