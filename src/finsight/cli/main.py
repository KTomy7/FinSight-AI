from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from finsight.application.dto import CompareModelsRequest, ForecastRequest, TrainModelRequest
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

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare trained model runs and print a leaderboard table",
    )
    compare_model_ids = list(settings.model_defaults.training_model_ids())
    compare_parser.add_argument(
        "--model-ids",
        nargs="+",
        default=compare_model_ids,
        help=f"Model IDs to compare (defaults: {', '.join(compare_model_ids)})",
    )
    compare_parser.add_argument(
        "--rank-by",
        nargs="+",
        default=[METRIC_MAE, METRIC_RMSE, METRIC_DIRECTION_ACCURACY],
        help=(
            "Ordered comparison metrics used for ranking "
            f"(defaults: {METRIC_MAE}, {METRIC_RMSE}, {METRIC_DIRECTION_ACCURACY})"
        ),
    )
    compare_parser.add_argument(
        "--artifacts-dir",
        default="artifacts/runs",
        help="Directory containing model run artifacts",
    )

    forecast_parser = subparsers.add_parser(
        "forecast",
        help="Forecast future prices from the latest trained run for a model",
    )
    forecast_parser.add_argument("--ticker", required=True, help="Ticker symbol to forecast (e.g. AAPL)")
    forecast_parser.add_argument("--model-id", required=True, help="Model ID to use for selecting the latest run")
    forecast_parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon in trading days")
    forecast_parser.add_argument(
        "--artifacts-dir",
        default="artifacts/runs",
        help="Directory containing model run artifacts",
    )
    forecast_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print forecast response as JSON",
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


def _run_compare(args: argparse.Namespace) -> int:
    container = build_container()
    response = container.compare_models.execute(
        CompareModelsRequest(
            model_ids=args.model_ids,
            rank_by=args.rank_by,
            artifacts_dir=args.artifacts_dir,
        )
    )

    id_to_label = get_settings().model_defaults.id_to_label()
    rows: list[dict[str, object]] = []
    for row in response.rows:
        record: dict[str, object] = {
            "rank": row.rank,
            "model": id_to_label.get(row.model_id, row.model_id),
        }
        record.update(row.metrics)
        rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No comparable model runs were found.")
        return 0

    base_columns = ["rank", "model"]
    metric_columns = [column for column in response.rank_by if column in frame.columns]
    remaining_columns = [
        column for column in frame.columns if column not in base_columns and column not in metric_columns
    ]
    ordered_columns = base_columns + metric_columns + sorted(remaining_columns)

    print(frame[ordered_columns].to_string(index=False))
    return 0


def _run_forecast(_args: argparse.Namespace) -> int:
    try:
        container = build_container()
        response = container.forecast.execute(
            ForecastRequest(
                ticker=_args.ticker,
                model_id=_args.model_id,
                horizon_days=_args.horizon,
                artifacts_dir=_args.artifacts_dir,
            )
        )
    except ValueError as exc:
        print(f"Forecast validation error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Forecast artifact error: {exc}", file=sys.stderr)
        return 1
    except TypeError as exc:
        print(f"Forecast runtime error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Forecast unexpected error: {exc}", file=sys.stderr)
        return 1

    if _args.as_json:
        print(json.dumps(response.to_dict(), indent=2, sort_keys=True))
        return 0

    print(
        f"[{response.model_id}] ticker={response.ticker} "
        f"horizon_days={response.horizon_days} rows={len(response.predictions)}"
    )
    for row in response.predictions:
        date_value = row.get("date", "")
        pred_ret_value = row.get("pred_ret_1d")
        pred_close_value = row.get("pred_close")
        print(
            f"{date_value} "
            f"pred_ret_1d={pred_ret_value} "
            f"pred_close={pred_close_value}"
        )
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        return _run_train(args)

    if args.command == "compare":
        return _run_compare(args)

    if args.command == "forecast":
        return _run_forecast(args)

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


