#!/usr/bin/env python3
"""
Script 2 — Per-Stock Metrics: Single Cutoff + Walk-Forward (Return & Price)
============================================================================
Part 1  Uses the application layer (build_container / TrainModelRequest) to
        train all five models with cutoff=2024-01-01, then computes MAE, RMSE,
        R² and Direction Accuracy for every model × ticker pair (return target).

Part 2  Runs expanding-window walk-forward validation on the same 3-year data
        set, collects per-ticker metrics averaged across folds, and saves the
        walk-forward test predictions for Script 03 (return target).

Part 3  Repeats Single Cutoff + Walk-Forward evaluation using a **price-level
        target** (next-day close) instead of next-day return.  This allows a
        direct comparison of return-based vs price-based prediction.

Outputs
-------
artifacts/dissertation/per_stock_metrics.csv              — return, single cutoff
artifacts/dissertation/per_stock_metrics_wf.csv           — return, walk-forward
artifacts/dissertation/wf_predictions/<model>.csv         — return WF predictions
artifacts/dissertation/per_stock_metrics_price.csv        — price, single cutoff
artifacts/dissertation/per_stock_metrics_price_wf.csv     — price, walk-forward
artifacts/dissertation/sc_predictions_price/<model>.csv   — price SC predictions
artifacts/dissertation/wf_predictions_price/<model>.csv   — price WF predictions

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
from finsight.infrastructure.features.feature_pipeline import (
    FEATURE_COLUMNS,
    add_features,
    add_target,
    to_panel_df,
)
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
PRICE_TARGET_COLUMN = "target_price_1d"

# Walk-forward hyper-parameters
WF_MIN_TRAIN_DAYS = 504  # ≈ 2 years of trading days → test period ≈ 2024
WF_TEST_WINDOW = 21  # ≈ 1 month
WF_STEP_DAYS = 21  # monthly non-overlapping steps

# Output paths
OUTPUT_DIR = Path("artifacts/dissertation")
WF_PRED_DIR = OUTPUT_DIR / "wf_predictions"
SC_PRED_PRICE_DIR = OUTPUT_DIR / "sc_predictions_price"
WF_PRED_PRICE_DIR = OUTPUT_DIR / "wf_predictions_price"


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


def _per_ticker_metrics_price(
    preds: pd.DataFrame,
    model_id: str,
) -> list[dict[str, object]]:
    """Compute metrics for price-level predictions.

    Direction accuracy is computed from the implied price movement direction:
    did the model correctly predict whether the price went up or down?
    Requires a ``close`` column (current-day close) in *preds*.
    """
    rows: list[dict[str, object]] = []
    for ticker in TICKERS:
        tp = preds[preds["ticker"] == ticker].copy()
        if tp.empty:
            continue
        m = forecast_metrics(y_true=tp["y_true"].tolist(), y_pred=tp["y_pred"].tolist())
        # Override direction accuracy: use price direction vs current close
        if "close" in tp.columns and tp["close"].notna().all():
            actual_up = tp["y_true"].values > tp["close"].values
            pred_up = tp["y_pred"].values > tp["close"].values
            dir_acc = float((actual_up == pred_up).mean())
        else:
            dir_acc = m["direction_accuracy"]
        rows.append(
            {
                "ticker": ticker,
                "model_id": model_id,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "direction_accuracy": dir_acc,
            }
        )
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    for d in (OUTPUT_DIR, WF_PRED_DIR, SC_PRED_PRICE_DIR, WF_PRED_PRICE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    container = build_container()

    # ── Part 1: Single Cutoff (Return) ───────────────────────────────────────
    print("=" * 70)
    print("PART 1: Single-Cutoff Evaluation — Return Target (cutoff={})".format(CUTOFF_DATE))
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

    print("\nSingle-Cutoff Per-Stock Metrics (Return):")
    print(sc_df.to_string(index=False))
    print(f"\nSaved → {OUTPUT_DIR / 'per_stock_metrics.csv'}")

    # ── Part 2: Walk-Forward (Return) ────────────────────────────────────────
    print("\n" + "=" * 70)
    print(
        "PART 2: Walk-Forward Validation — Return Target  "
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

    # Build feature dataset (return target)
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

    # Build model router (shared by Parts 2 & 3)
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

    print("\nWalk-Forward Per-Stock Metrics (Return):")
    print(wf_df.to_string(index=False))
    print(f"\nSaved → {OUTPUT_DIR / 'per_stock_metrics_wf.csv'}")
    print(f"Predictions → {WF_PRED_DIR}/")

    # ── Part 3: Price-Level Prediction ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 3: Price-Level Prediction (Single Cutoff + Walk-Forward)")
    print("=" * 70)

    # Build feature panel with price target (reuse fetched series_list)
    panel = to_panel_df(series_list)
    panel = add_features(panel)
    panel = add_target(panel)
    panel[PRICE_TARGET_COLUMN] = panel.groupby("ticker", sort=False)["close"].shift(-1)

    # Preserve close for direction accuracy computation
    close_lookup = panel[["date", "ticker", "close"]].copy()
    close_lookup["date"] = pd.to_datetime(close_lookup["date"])

    # Build clean feature frame with price target only (no return target,
    # no close — so the model router picks only the 15 feature columns)
    price_cols = ["date", "ticker"] + list(FEATURE_COLUMNS) + [PRICE_TARGET_COLUMN]
    price_df = panel[price_cols].dropna(
        subset=list(FEATURE_COLUMNS) + [PRICE_TARGET_COLUMN]
    )
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    price_df["date"] = pd.to_datetime(price_df["date"])

    print(f"\nPrice-level feature dataset: {len(price_df)} rows")

    # ── 3a: Single Cutoff (Price) ────────────────────────────────────────
    print(f"\n--- Price Single Cutoff (cutoff={CUTOFF_DATE}) ---")
    cutoff_ts = pd.Timestamp(CUTOFF_DATE)
    train_price = price_df[price_df["date"] < cutoff_ts].copy()
    test_price = price_df[price_df["date"] >= cutoff_ts].copy()
    print(f"  Train: {len(train_price)} rows, Test: {len(test_price)} rows")

    price_sc_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        print(f"  Training {model_id} (price target) …")
        eval_result = model_router.evaluate(
            train_dataset=train_price,
            test_dataset=test_price,
            model_type=model_id,
            target_column=PRICE_TARGET_COLUMN,
        )
        preds = eval_result.predictions
        preds["date"] = pd.to_datetime(preds["date"])
        preds = preds.merge(close_lookup, on=["date", "ticker"], how="left")
        preds.to_csv(SC_PRED_PRICE_DIR / f"{model_id}.csv", index=False)
        price_sc_rows.extend(_per_ticker_metrics_price(preds, model_id))

    price_sc_df = pd.DataFrame(price_sc_rows)
    price_sc_df.to_csv(OUTPUT_DIR / "per_stock_metrics_price.csv", index=False)

    print("\nPrice Single-Cutoff Per-Stock Metrics:")
    print(price_sc_df.to_string(index=False))
    print(f"\nSaved → {OUTPUT_DIR / 'per_stock_metrics_price.csv'}")

    # ── 3b: Walk-Forward (Price) ─────────────────────────────────────────
    print(f"\n--- Price Walk-Forward ---")
    folds_price = split_policy.split_frame(price_df)
    print(f"Walk-forward folds: {len(folds_price)}")

    price_wf_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        print(f"  Evaluating {model_id} (price target) across {len(folds_price)} folds …")
        fold_preds_p: list[pd.DataFrame] = []
        for fold in folds_price:
            eval_result = model_router.evaluate(
                train_dataset=fold.train_df,
                test_dataset=fold.test_df,
                model_type=model_id,
                target_column=PRICE_TARGET_COLUMN,
            )
            fold_preds_p.append(eval_result.predictions)

        combined_p = pd.concat(fold_preds_p, ignore_index=True)
        combined_p["date"] = pd.to_datetime(combined_p["date"])
        combined_p = combined_p.sort_values(["ticker", "date"]).reset_index(drop=True)
        combined_p = combined_p.merge(close_lookup, on=["date", "ticker"], how="left")
        combined_p.to_csv(WF_PRED_PRICE_DIR / f"{model_id}.csv", index=False)
        price_wf_rows.extend(_per_ticker_metrics_price(combined_p, model_id))

    price_wf_df = pd.DataFrame(price_wf_rows)
    price_wf_df.to_csv(OUTPUT_DIR / "per_stock_metrics_price_wf.csv", index=False)

    print("\nPrice Walk-Forward Per-Stock Metrics:")
    print(price_wf_df.to_string(index=False))
    print(f"\nSaved → {OUTPUT_DIR / 'per_stock_metrics_price_wf.csv'}")

    # ── Comparison summaries ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QUICK COMPARISON: Mean RMSE by Model")
    print("=" * 70)

    def _mean_rmse_table(df: pd.DataFrame, label: str) -> pd.Series:
        return df.groupby("model_id")["rmse"].mean().rename(label)

    comp = pd.concat(
        [
            _mean_rmse_table(sc_df, "ret_sc"),
            _mean_rmse_table(wf_df, "ret_wf"),
            _mean_rmse_table(price_sc_df, "price_sc"),
            _mean_rmse_table(price_wf_df, "price_wf"),
        ],
        axis=1,
    ).reindex(MODEL_IDS)
    print(comp.to_string())


if __name__ == "__main__":
    main()
