from __future__ import annotations

import argparse

from finsight.application.use_cases.train_model import TrainModelRequest
from finsight.bootstrap.container import build_container
from finsight.config.settings import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finsight")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tickers = ", ".join(get_settings().training.training_tickers)
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
        default=["naive_zero", "naive_mean"],
        help="Model types to evaluate",
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
            f"MAE={float(model_metrics['mae']):.6f} "
            f"RMSE={float(model_metrics['rmse']):.6f} "
            f"DirectionAcc={float(model_metrics['direction_accuracy']):.4f}"
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


