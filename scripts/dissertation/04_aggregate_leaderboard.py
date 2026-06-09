#!/usr/bin/env python3
"""
Script 4 — Aggregate Leaderboard (Single Cutoff + Walk-Forward)
================================================================
Reads the per-stock metrics CSVs produced by Script 02 and:

 1. Computes aggregate statistics per model (mean MAE, RMSE, Direction Accuracy).
 2. Ranks by mean RMSE (ascending).
 3. Groups tickers by sector and prints mean RMSE per model per sector.
 4. Compares single-cutoff vs walk-forward aggregate results side-by-side.

Outputs
-------
artifacts/dissertation/aggregate_leaderboard.csv
artifacts/dissertation/aggregate_leaderboard_wf.csv
artifacts/dissertation/aggregate_comparison.csv

Run from repo root:
    python scripts/dissertation/04_aggregate_leaderboard.py
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

# Sector groupings
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
def build_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean MAE, RMSE, Direction Accuracy per model, ranked by RMSE asc."""
    agg = (
        df.groupby("model_id")
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_direction_accuracy=("direction_accuracy", "mean"),
        )
        .sort_values("mean_rmse")
        .reset_index()
    )
    agg.insert(0, "rank", range(1, len(agg) + 1))
    agg.insert(2, "model_label", agg["model_id"].map(MODEL_LABELS))
    return agg


def build_sector_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean RMSE per model per sector group."""
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    return (
        df.groupby(["sector", "model_id"])["rmse"]
        .mean()
        .unstack("model_id")
        .reindex(columns=df["model_id"].unique())
    )


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    sc_path = DISS_DIR / "per_stock_metrics.csv"
    wf_path = DISS_DIR / "per_stock_metrics_wf.csv"

    has_sc = sc_path.exists()
    has_wf = wf_path.exists()

    if not has_sc and not has_wf:
        print("ERROR: No per-stock metrics CSVs found. Run Script 02 first.")
        sys.exit(1)

    # ── Single Cutoff ────────────────────────────────────────────────────────
    sc_agg = None
    if has_sc:
        sc_df = pd.read_csv(sc_path)
        sc_agg = build_aggregate(sc_df)
        sc_agg.to_csv(DISS_DIR / "aggregate_leaderboard.csv", index=False)

        print_section("Single-Cutoff Aggregate Leaderboard (ranked by Mean RMSE)")
        print(sc_agg.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / 'aggregate_leaderboard.csv'}")

        print_section("Single-Cutoff Sector Breakdown (Mean RMSE)")
        sector_sc = build_sector_table(sc_df)
        print(sector_sc.to_string())

    # ── Walk-Forward ─────────────────────────────────────────────────────────
    wf_agg = None
    if has_wf:
        wf_df = pd.read_csv(wf_path)
        wf_agg = build_aggregate(wf_df)
        wf_agg.to_csv(DISS_DIR / "aggregate_leaderboard_wf.csv", index=False)

        print_section("Walk-Forward Aggregate Leaderboard (ranked by Mean RMSE)")
        print(wf_agg.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / 'aggregate_leaderboard_wf.csv'}")

        print_section("Walk-Forward Sector Breakdown (Mean RMSE)")
        sector_wf = build_sector_table(wf_df)
        print(sector_wf.to_string())

    # ── Side-by-side comparison ──────────────────────────────────────────────
    if sc_agg is not None and wf_agg is not None:
        print_section("Comparison: Single Cutoff vs Walk-Forward")

        comp = sc_agg[["model_id", "model_label", "mean_rmse", "mean_mae", "mean_direction_accuracy"]].copy()
        comp.columns = ["model_id", "model", "sc_rmse", "sc_mae", "sc_dir_acc"]
        wf_cols = wf_agg[["model_id", "mean_rmse", "mean_mae", "mean_direction_accuracy"]].copy()
        wf_cols.columns = ["model_id", "wf_rmse", "wf_mae", "wf_dir_acc"]
        comp = comp.merge(wf_cols, on="model_id", how="outer")
        comp["rmse_delta"] = comp["wf_rmse"] - comp["sc_rmse"]
        comp["dir_acc_delta"] = comp["wf_dir_acc"] - comp["sc_dir_acc"]
        comp = comp.sort_values("sc_rmse")
        comp.to_csv(DISS_DIR / "aggregate_comparison.csv", index=False)

        print(comp.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / 'aggregate_comparison.csv'}")

        # Rank stability check
        sc_rank = list(sc_agg["model_id"])
        wf_rank = list(wf_agg["model_id"])
        if sc_rank == wf_rank:
            print("\n✓ Model ranking is STABLE across both evaluation strategies.")
        else:
            print("\n⚠ Model ranking DIFFERS between strategies:")
            print(f"  Single cutoff : {sc_rank}")
            print(f"  Walk-forward  : {wf_rank}")


if __name__ == "__main__":
    main()
