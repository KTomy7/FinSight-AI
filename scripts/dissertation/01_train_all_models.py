#!/usr/bin/env python3
"""
Script 1 — Train All Models (SC + WF × Return + Price)
=======================================================
Trains every model under all four experimental conditions and saves per-stock
metrics CSVs plus raw prediction files for Scripts 02-04.

Parts
-----
1. Single Cutoff, Return Target  — via ``TrainModelRequest`` (cutoff 2024-01-01)
2. Walk-Forward, Return Target   — expanding-window using building blocks
3. Single Cutoff, Price Target   — ``target_price_1d`` = next-day close
4. Walk-Forward, Price Target    — same walk-forward on price target

Outputs  (all under ``artifacts/dissertation/``)
-------------------------------------------------
per_stock_metrics_sc_return.csv   per_stock_metrics_wf_return.csv
per_stock_metrics_sc_price.csv    per_stock_metrics_wf_price.csv
predictions_sc_return/<model>.csv predictions_wf_return/<model>.csv
predictions_sc_price/<model>.csv  predictions_wf_price/<model>.csv

Run from repo root:
    python scripts/dissertation/01_train_all_models.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

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
# Experimental conditions
# ═══════════════════════════════════════════════════════════════════════════════
TICKERS = ["AAPL", "JPM", "XOM", "KO", "TSLA"]
MODEL_IDS = ["naive_zero", "naive_mean", "ridge", "hist_gbdt", "xgboost"]
CUTOFF_DATE = "2024-01-01"
YEARS = 3
END_DATE = "2024-12-31"
ARTIFACTS_DIR = "artifacts/runs"
TARGET_COLUMN = "target_ret_1d"
PRICE_TARGET_COLUMN = "target_price_1d"

WF_MIN_TRAIN_DAYS = 504   # ≈ 2 years of trading days
WF_TEST_WINDOW = 21       # ≈ 1 month
WF_STEP_DAYS = 21         # monthly non-overlapping steps

# Output paths
OUTPUT_DIR = Path("artifacts/dissertation")
PRED_DIRS = {
    ("sc", "return"): OUTPUT_DIR / "predictions_sc_return",
    ("wf", "return"): OUTPUT_DIR / "predictions_wf_return",
    ("sc", "price"):  OUTPUT_DIR / "predictions_sc_price",
    ("wf", "price"):  OUTPUT_DIR / "predictions_wf_price",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _per_ticker_metrics(
    preds: pd.DataFrame,
    model_id: str,
) -> list[dict[str, object]]:
    """MAE / RMSE / R² / Direction Accuracy per ticker (return target)."""
    rows: list[dict[str, object]] = []
    for ticker in TICKERS:
        tp = preds[preds["ticker"] == ticker]
        if tp.empty:
            continue
        m = forecast_metrics(y_true=tp["y_true"].tolist(), y_pred=tp["y_pred"].tolist())
        rows.append({
            "ticker": ticker, "model_id": model_id,
            "mae": m["mae"], "rmse": m["rmse"],
            "r2": m["r2"], "direction_accuracy": m["direction_accuracy"],
        })
    return rows


def _per_ticker_metrics_price(
    preds: pd.DataFrame,
    model_id: str,
) -> list[dict[str, object]]:
    """Metrics for price target; direction accuracy from implied movement."""
    rows: list[dict[str, object]] = []
    for ticker in TICKERS:
        tp = preds[preds["ticker"] == ticker].copy()
        if tp.empty:
            continue
        m = forecast_metrics(y_true=tp["y_true"].tolist(), y_pred=tp["y_pred"].tolist())
        if "close" in tp.columns and tp["close"].notna().all():
            actual_up = tp["y_true"].values > tp["close"].values
            pred_up = tp["y_pred"].values > tp["close"].values
            dir_acc = float((actual_up == pred_up).mean())
        else:
            dir_acc = m["direction_accuracy"]
        rows.append({
            "ticker": ticker, "model_id": model_id,
            "mae": m["mae"], "rmse": m["rmse"],
            "r2": m["r2"], "direction_accuracy": dir_acc,
        })
    return rows


def _save_metrics(rows: list[dict], strategy: str, target: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / f"per_stock_metrics_{strategy}_{target}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    print(df.to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    for d in PRED_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    container = build_container()

    # ── Part 1: Single Cutoff — Return ───────────────────────────────────────
    print("=" * 70)
    print(f"PART 1: Single Cutoff × Return (cutoff={CUTOFF_DATE})")
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

    sc_ret_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        run_dir = train_result.run_dirs[model_id]
        preds = pd.read_csv(Path(run_dir) / "predictions.csv")
        preds["date"] = pd.to_datetime(preds["date"])
        # Save to consistent prediction dir
        preds.to_csv(PRED_DIRS[("sc", "return")] / f"{model_id}.csv", index=False)
        sc_ret_rows.extend(_per_ticker_metrics(preds, model_id))

    _save_metrics(sc_ret_rows, "sc", "return")

    # ── Fetch data (shared by Parts 2-4) ─────────────────────────────────────
    end_d = date.fromisoformat(END_DATE)
    start_d = end_d - timedelta(days=(YEARS * 365) - 1)

    print(f"\nFetching market data: {start_d} → {end_d}")
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

    # ── Build feature datasets ───────────────────────────────────────────────
    # Return target dataset (via feature store)
    feature_store = PandasFeatureStore()
    feature_dataset = feature_store.build_feature_dataset(series_list)
    print(f"\nReturn feature dataset: {len(feature_dataset)} rows")

    # Price target dataset (manual panel build)
    panel = to_panel_df(series_list)
    panel = add_features(panel)
    panel = add_target(panel)
    panel[PRICE_TARGET_COLUMN] = panel.groupby("ticker", sort=False)["close"].shift(-1)
    close_lookup = panel[["date", "ticker", "close"]].copy()
    close_lookup["date"] = pd.to_datetime(close_lookup["date"])

    price_cols = ["date", "ticker"] + list(FEATURE_COLUMNS) + [PRICE_TARGET_COLUMN]
    price_df = panel[price_cols].dropna(
        subset=list(FEATURE_COLUMNS) + [PRICE_TARGET_COLUMN]
    ).sort_values(["ticker", "date"]).reset_index(drop=True)
    price_df["date"] = pd.to_datetime(price_df["date"])
    print(f"Price feature dataset:  {len(price_df)} rows")

    # Model router (shared by Parts 2-4)
    model_router = SklearnModelRouter(
        adapters=[
            NaiveBaselineModel(),
            LinearSklearnModel(),
            HistGradientBoostingModel(),
            XGBoostModel(),
        ]
    )

    # Walk-forward split policy
    split_policy = WalkForwardSplitPolicy(
        min_train_size=WF_MIN_TRAIN_DAYS,
        test_size=WF_TEST_WINDOW,
        step_size=WF_STEP_DAYS,
    )

    # ── Part 2: Walk-Forward — Return ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2: Walk-Forward × Return")
    print("=" * 70)

    folds = split_policy.split_frame(feature_dataset)
    print(f"Folds: {len(folds)}")
    for f in folds:
        print(f"  Fold {f.fold_index}: train {f.train_start}..{f.train_end} | "
              f"test {f.test_start}..{f.test_end}")

    wf_ret_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        print(f"\n  {model_id} across {len(folds)} folds …")
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
        combined.to_csv(PRED_DIRS[("wf", "return")] / f"{model_id}.csv", index=False)
        wf_ret_rows.extend(_per_ticker_metrics(combined, model_id))

    _save_metrics(wf_ret_rows, "wf", "return")

    # ── Part 3: Single Cutoff — Price ────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PART 3: Single Cutoff × Price (cutoff={CUTOFF_DATE})")
    print("=" * 70)

    cutoff_ts = pd.Timestamp(CUTOFF_DATE)
    train_price = price_df[price_df["date"] < cutoff_ts].copy()
    test_price = price_df[price_df["date"] >= cutoff_ts].copy()
    print(f"  Train: {len(train_price)} rows, Test: {len(test_price)} rows")

    sc_price_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        print(f"  {model_id} (price) …")
        eval_result = model_router.evaluate(
            train_dataset=train_price,
            test_dataset=test_price,
            model_type=model_id,
            target_column=PRICE_TARGET_COLUMN,
        )
        preds = eval_result.predictions
        preds["date"] = pd.to_datetime(preds["date"])
        preds = preds.merge(close_lookup, on=["date", "ticker"], how="left")
        preds.to_csv(PRED_DIRS[("sc", "price")] / f"{model_id}.csv", index=False)
        sc_price_rows.extend(_per_ticker_metrics_price(preds, model_id))

    _save_metrics(sc_price_rows, "sc", "price")

    # ── Part 4: Walk-Forward — Price ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 4: Walk-Forward × Price")
    print("=" * 70)

    folds_price = split_policy.split_frame(price_df)
    print(f"Folds: {len(folds_price)}")

    wf_price_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        print(f"  {model_id} (price) across {len(folds_price)} folds …")
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
        combined_p.to_csv(PRED_DIRS[("wf", "price")] / f"{model_id}.csv", index=False)
        wf_price_rows.extend(_per_ticker_metrics_price(combined_p, model_id))

    _save_metrics(wf_price_rows, "wf", "price")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Mean RMSE by Model across all 4 conditions")
    print("=" * 70)

    def _col(rows: list[dict], label: str) -> pd.Series:
        return pd.DataFrame(rows).groupby("model_id")["rmse"].mean().rename(label)

    comp = pd.concat([
        _col(sc_ret_rows, "sc_return"),
        _col(wf_ret_rows, "wf_return"),
        _col(sc_price_rows, "sc_price"),
        _col(wf_price_rows, "wf_price"),
    ], axis=1).reindex(MODEL_IDS)
    print(comp.to_string())
    print("\nDone ✓")


if __name__ == "__main__":
    main()
