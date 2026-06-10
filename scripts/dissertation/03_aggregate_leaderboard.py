#!/usr/bin/env python3
"""
Script 3 — Aggregate Leaderboard + Comparisons
================================================
Reads per-stock metrics CSVs from Script 01 and:

 1. Computes aggregate leaderboard per model (mean MAE, RMSE, R², Dir. Acc).
 2. Ranks by mean RMSE (ascending).
 3. Groups tickers by sector and prints mean RMSE per model per sector.
 4. Compares SC vs WF side-by-side for each target type.
 5. Compares return vs price model rankings.
 6. Saves a formatted text leaderboard (replaces ``finsight compare``).

Outputs  (all under ``artifacts/dissertation/``)
-------------------------------------------------
aggregate_leaderboard_sc_return.csv
aggregate_leaderboard_wf_return.csv
aggregate_leaderboard_sc_price.csv
aggregate_leaderboard_wf_price.csv
aggregate_comparison_return.csv           — SC vs WF (return)
aggregate_comparison_price.csv            — SC vs WF (price)
aggregate_comparison_return_vs_price.csv  — return vs price
leaderboard.txt                           — formatted text summary

Run from repo root:
    python scripts/dissertation/03_aggregate_leaderboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from finsight.config.settings import get_settings

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
DISS_DIR = Path("artifacts/dissertation")
MODEL_LABELS = get_settings().model_defaults.id_to_label()

STRATEGIES = ["sc", "wf"]
TARGETS = ["return", "price"]
STRATEGY_LABELS = {"sc": "Single Cutoff", "wf": "Walk-Forward"}
TARGET_LABELS = {"return": "Return", "price": "Price"}

SECTOR_MAP = {
    "KO": "Consumer Staples",
    "JPM": "Financial",
    "XOM": "Energy",
    "AAPL": "Technology",
    "TSLA": "Growth / Volatile",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _load(strategy: str, target: str) -> pd.DataFrame | None:
    path = DISS_DIR / f"per_stock_metrics_{strategy}_{target}.csv"
    return pd.read_csv(path) if path.exists() else None


def build_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("model_id")
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_direction_accuracy=("direction_accuracy", "mean"),
        )
        .sort_values("mean_rmse")
        .reset_index()
    )
    agg.insert(0, "rank", range(1, len(agg) + 1))
    agg.insert(2, "model_label", agg["model_id"].map(MODEL_LABELS))
    return agg


def build_sector_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    return (
        df.groupby(["sector", "model_id"])["rmse"]
        .mean()
        .unstack("model_id")
        .reindex(columns=df["model_id"].unique())
    )


def build_comparison(sc_agg: pd.DataFrame, wf_agg: pd.DataFrame) -> pd.DataFrame:
    comp = sc_agg[["model_id", "model_label", "mean_rmse", "mean_mae",
                    "mean_r2", "mean_direction_accuracy"]].copy()
    comp.columns = ["model_id", "model", "sc_rmse", "sc_mae", "sc_r2", "sc_dir_acc"]
    wf_cols = wf_agg[["model_id", "mean_rmse", "mean_mae",
                       "mean_r2", "mean_direction_accuracy"]].copy()
    wf_cols.columns = ["model_id", "wf_rmse", "wf_mae", "wf_r2", "wf_dir_acc"]
    comp = comp.merge(wf_cols, on="model_id", how="outer")
    comp["rmse_delta"] = comp["wf_rmse"] - comp["sc_rmse"]
    comp["r2_delta"] = comp["wf_r2"] - comp["sc_r2"]
    comp["dir_acc_delta"] = comp["wf_dir_acc"] - comp["sc_dir_acc"]
    return comp.sort_values("sc_rmse")


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def rank_stability(sc_agg: pd.DataFrame, wf_agg: pd.DataFrame) -> None:
    sc_rank = list(sc_agg["model_id"])
    wf_rank = list(wf_agg["model_id"])
    if sc_rank == wf_rank:
        print("\n✓ Model ranking is STABLE across both evaluation strategies.")
    else:
        print("\n⚠ Model ranking DIFFERS between strategies:")
        print(f"  Single cutoff : {sc_rank}")
        print(f"  Walk-forward  : {wf_rank}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # Load all per-stock metrics
    data: dict[tuple[str, str], pd.DataFrame] = {}
    aggs: dict[tuple[str, str], pd.DataFrame] = {}

    for strategy in STRATEGIES:
        for target in TARGETS:
            df = _load(strategy, target)
            if df is not None:
                data[(strategy, target)] = df

    if not data:
        print("ERROR: No per-stock metrics CSVs found. Run Script 01 first.")
        sys.exit(1)

    # ── Individual leaderboards ──────────────────────────────────────────────
    for (strategy, target), df in data.items():
        tag = f"{STRATEGY_LABELS[strategy]}, {TARGET_LABELS[target]}"
        out_name = f"aggregate_leaderboard_{strategy}_{target}.csv"

        agg = build_aggregate(df)
        aggs[(strategy, target)] = agg
        agg.to_csv(DISS_DIR / out_name, index=False)

        _section(f"{tag} — Aggregate Leaderboard (ranked by RMSE)")
        print(agg.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / out_name}")

        _section(f"{tag} — Sector Breakdown (Mean RMSE)")
        print(build_sector_table(df).to_string())

    # ── SC vs WF comparisons (per target type) ──────────────────────────────
    for target in TARGETS:
        sc_key = ("sc", target)
        wf_key = ("wf", target)
        if sc_key in aggs and wf_key in aggs:
            _section(f"{TARGET_LABELS[target]} — SC vs WF Comparison")
            comp = build_comparison(aggs[sc_key], aggs[wf_key])
            out = f"aggregate_comparison_{target}.csv"
            comp.to_csv(DISS_DIR / out, index=False)
            print(comp.to_string(index=False))
            print(f"\nSaved → {DISS_DIR / out}")
            rank_stability(aggs[sc_key], aggs[wf_key])

    # ── Return vs Price comparison ───────────────────────────────────────────
    ret_key = ("sc", "return")
    price_key = ("sc", "price")
    if ret_key in aggs and price_key in aggs:
        _section("RETURN vs PRICE — Single-Cutoff Comparison")
        ret_agg = aggs[ret_key]
        price_agg = aggs[price_key]

        ret_cols = ret_agg[["model_id", "model_label", "mean_r2",
                             "mean_direction_accuracy"]].copy()
        ret_cols.columns = ["model_id", "model", "ret_r2", "ret_dir_acc"]
        price_cols = price_agg[["model_id", "mean_r2",
                                 "mean_direction_accuracy"]].copy()
        price_cols.columns = ["model_id", "price_r2", "price_dir_acc"]
        rvp = ret_cols.merge(price_cols, on="model_id", how="outer")
        rvp["r2_delta"] = rvp["price_r2"] - rvp["ret_r2"]
        rvp["dir_acc_delta"] = rvp["price_dir_acc"] - rvp["ret_dir_acc"]
        rvp = rvp.sort_values("ret_r2", ascending=False)

        out = "aggregate_comparison_return_vs_price.csv"
        rvp.to_csv(DISS_DIR / out, index=False)
        print(rvp.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / out}")

        ret_rank = list(ret_agg["model_id"])
        price_rank = list(price_agg["model_id"])
        if ret_rank == price_rank:
            print("\n✓ Model ranking is IDENTICAL for return and price targets.")
        else:
            print("\n⚠ Model ranking DIFFERS between return and price targets:")
            print(f"  Return : {ret_rank}")
            print(f"  Price  : {price_rank}")

    # ── Text leaderboard (replaces finsight compare) ─────────────────────────
    _section("Generating leaderboard.txt")
    lines: list[str] = []
    lines.append("FinSight-AI — Dissertation Experiment Leaderboard")
    lines.append("=" * 60)
    for (strategy, target), agg in sorted(aggs.items()):
        tag = f"{STRATEGY_LABELS[strategy]}, {TARGET_LABELS[target]}"
        lines.append(f"\n{tag}")
        lines.append("-" * 60)
        lines.append(
            f"{'Rank':<5} {'Model':<30} {'MAE':>10} {'RMSE':>10} "
            f"{'R²':>8} {'DirAcc':>8}"
        )
        lines.append("-" * 60)
        for _, row in agg.iterrows():
            lines.append(
                f"{row['rank']:<5} {row['model_label']:<30} "
                f"{row['mean_mae']:>10.6f} {row['mean_rmse']:>10.6f} "
                f"{row['mean_r2']:>8.4f} {row['mean_direction_accuracy']:>7.1%}"
            )
    text = "\n".join(lines) + "\n"
    (DISS_DIR / "leaderboard.txt").write_text(text)
    print(text)
    print(f"Saved → {DISS_DIR / 'leaderboard.txt'}")


if __name__ == "__main__":
    main()
