#!/usr/bin/env python3
"""
Script 2 — Per-Stock Metrics: Single Cutoff + Walk-Forward
===========================================================
Part 1  Uses the application layer (build_container / TrainModelRequest) to
        train all five models with cutoff=2024-01-01, then computes MAE, RMSE,
        R² and Direction Accuracy for every model × ticker pair.

Part 2  Runs expanding-window walk-forward validation on the same 3-year data
        set, collects per-ticker metrics averaged across folds, and saves the
        walk-forward test predictions for Script 03.

Outputs
-------
artifacts/dissertation/per_stock_metrics.csv       — single-cutoff results
artifacts/dissertation/per_stock_metrics_wf.csv    — walk-forward results
artifacts/dissertation/wf_predictions/<model>.csv  — walk-forward predictions

Run from repo root:
    python scripts/dissertation/02_per_stock_metrics.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure finsight is importable even without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from finsight.application.dto import FetchMarketDataRequest, TrainModelRequest
from finsight.bootstrap.container import build_container
from finsight.domain.metrics import forecast_metrics
from finsight.infrastructure.features.feature_store import PandasFeatureStore
from finsight.infrastructure.features.policies import WalkForwardSplitPolicy
from finsight.infrastructure.ml.sklearn import (
    HistGradientBoostingModel,
    LinearSklearnModel,
    NaiveBaselineModel,
    SklearnModelRouter,
    XGBoostModel,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Experimental conditions (shared across all scripts)
# ═══════════════════════════════════════════════════════════════════════════════
TICKERS = ["AAPL", "JPM", "XOM", "KO", "TSLA"]
MODEL_IDS = ["naive_zero", "naive_mean", "ridge", "hist_gbdt", "xgboost"]
CUTOFF_DATE = "2024-01-01"
YEARS = 3
END_DATE = "2024-12-31"
ARTIFACTS_DIR = "artifacts/runs"
TARGET_COLUMN = "target_ret_1d"

# Walk-forward hyper-parameters
WF_MIN_TRAIN_DAYS = 504  # ≈ 2 years of trading days → test period ≈ 2024
WF_TEST_WINDOW = 21  # ≈ 1 month
WF_STEP_DAYS = 21  # monthly non-overlapping steps

# Output paths
OUTPUT_DIR = Path("artifacts/dissertation")
WF_PRED_DIR = OUTPUT_DIR / "wf_predictions"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _per_ticker_metrics(
    preds: pd.DataFrame,
    model_id: str,
) -> list[dict[str, object]]:
    """Compute MAE/RMSE/R²/direction_accuracy per ticker from a predictions DF."""
    rows: list[dict[str, object]] = []
    for ticker in TICKERS:
        tp = preds[preds["ticker"] == ticker]
        if tp.empty:
            continue
        m = forecast_metrics(y_true=tp["y_true"].tolist(), y_pred=tp["y_pred"].tolist())
        rows.append(
            {
                "ticker": ticker,
                "model_id": model_id,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "direction_accuracy": m["direction_accuracy"],
            }
        )
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WF_PRED_DIR.mkdir(parents=True, exist_ok=True)

    container = build_container()

    # ── Part 1: Single Cutoff ────────────────────────────────────────────────
    print("=" * 70)
    print("PART 1: Single-Cutoff Evaluation  (cutoff={})".format(CUTOFF_DATE))
    print("=" * 70)

    train_result = container.train_model.execute(
        TrainModelRequest(
            cutoff_date=CUTOFF_DATE,
            years=YEARS,
            end=END_DATE,
            model_types=MODEL_IDS,
            artifacts_dir=ARTIFACTS_DIR,
        )
    )

    sc_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        run_dir = train_result.run_dirs[model_id]
        preds = pd.read_csv(Path(run_dir) / "predictions.csv")
        preds["date"] = pd.to_datetime(preds["date"])
        sc_rows.extend(_per_ticker_metrics(preds, model_id))

    sc_df = pd.DataFrame(sc_rows)
    sc_df.to_csv(OUTPUT_DIR / "per_stock_metrics.csv", index=False)

    print("\nSingle-Cutoff Per-Stock Metrics:")
    print(sc_df.to_string(index=False))
    print(f"\nSaved → {OUTPUT_DIR / 'per_stock_metrics.csv'}")

    # ── Part 2: Walk-Forward ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(
        "PART 2: Walk-Forward Validation  "
        f"(min_train={WF_MIN_TRAIN_DAYS}, window={WF_TEST_WINDOW}, step={WF_STEP_DAYS})"
    )
    print("=" * 70)

    # Fetch data for all tickers (same date range as training)
    end_d = date.fromisoformat(END_DATE)
    start_d = end_d - timedelta(days=(YEARS * 365) - 1)

    print(f"\nFetching data: {start_d} → {end_d}")
    series_list = []
    for ticker in TICKERS:
        result = container.fetch_market_data.execute(
            FetchMarketDataRequest(
                ticker=ticker,
                start_date=start_d.isoformat(),
                end_date=end_d.isoformat(),
                interval="1d",
                include_summary=False,
            )
        )
        series_list.append(result.history)
        print(f"  {ticker}: {len(result.history.df)} rows")

    # Build feature dataset
    feature_store = PandasFeatureStore()
    feature_dataset = feature_store.build_feature_dataset(series_list)
    print(f"\nFeature dataset: {len(feature_dataset)} rows, "
          f"{len(feature_dataset.columns)} columns")

    # Walk-forward split
    split_policy = WalkForwardSplitPolicy(
        min_train_size=WF_MIN_TRAIN_DAYS,
        test_size=WF_TEST_WINDOW,
        step_size=WF_STEP_DAYS,
    )
    folds = split_policy.split_frame(feature_dataset)
    print(f"Walk-forward folds: {len(folds)}")
    for f in folds:
        print(f"  Fold {f.fold_index}: train {f.train_start}..{f.train_end} | "
              f"test {f.test_start}..{f.test_end}")

    # Build model router
    model_router = SklearnModelRouter(
        adapters=[
            NaiveBaselineModel(),
            LinearSklearnModel(),
            HistGradientBoostingModel(),
            XGBoostModel(),
        ]
    )

    wf_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        print(f"\n  Evaluating {model_id} across {len(folds)} folds …")
        fold_preds: list[pd.DataFrame] = []
        for fold in folds:
            eval_result = model_router.evaluate(
                train_dataset=fold.train_df,
                test_dataset=fold.test_df,
                model_type=model_id,
                target_column=TARGET_COLUMN,
            )
            fold_preds.append(eval_result.predictions)

        combined = pd.concat(fold_preds, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Save walk-forward predictions for Script 03
        combined.to_csv(WF_PRED_DIR / f"{model_id}.csv", index=False)

        wf_rows.extend(_per_ticker_metrics(combined, model_id))

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUTPUT_DIR / "per_stock_metrics_wf.csv", index=False)

    print("\nWalk-Forward Per-Stock Metrics:")
    print(wf_df.to_string(index=False))
    print(f"\nSaved → {OUTPUT_DIR / 'per_stock_metrics_wf.csv'}")
    print(f"Predictions → {WF_PRED_DIR}/")

    # ── Comparison summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QUICK COMPARISON: Mean RMSE by Model (Single Cutoff vs Walk-Forward)")
    print("=" * 70)
    sc_mean = sc_df.groupby("model_id")["rmse"].mean().rename("single_cutoff")
    wf_mean = wf_df.groupby("model_id")["rmse"].mean().rename("walk_forward")
    comp = pd.concat([sc_mean, wf_mean], axis=1).reindex(MODEL_IDS)
    comp["delta"] = comp["walk_forward"] - comp["single_cutoff"]
    print(comp.to_string())


if __name__ == "__main__":
    main()
