#!/usr/bin/env python3
"""
Script 4 — Aggregate Leaderboard (Return & Price, SC + WF)
============================================================
Reads the per-stock metrics CSVs produced by Script 02 and:

 1. Computes aggregate statistics per model (mean MAE, RMSE, R², Dir. Accuracy).
 2. Ranks by mean RMSE (ascending).
 3. Groups tickers by sector and prints mean RMSE per model per sector.
 4. Compares single-cutoff vs walk-forward aggregate results side-by-side.
 5. Compares return-target vs price-target model rankings.

Outputs
-------
artifacts/dissertation/aggregate_leaderboard.csv            — return SC
artifacts/dissertation/aggregate_leaderboard_wf.csv         — return WF
artifacts/dissertation/aggregate_comparison.csv             — return SC vs WF
artifacts/dissertation/aggregate_leaderboard_price.csv      — price SC
artifacts/dissertation/aggregate_leaderboard_price_wf.csv   — price WF
artifacts/dissertation/aggregate_comparison_price.csv       — price SC vs WF
artifacts/dissertation/aggregate_return_vs_price.csv        — return vs price

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

SECTOR_MAP = {
    "KO": "Consumer Staples",
    "JPM": "Financial",
    "XOM": "Energy",
    "AAPL": "Technology",
    "TSLA": "Growth / Volatile",
}

# CSV paths — return
SC_RET = DISS_DIR / "per_stock_metrics.csv"
WF_RET = DISS_DIR / "per_stock_metrics_wf.csv"
# CSV paths — price
SC_PRICE = DISS_DIR / "per_stock_metrics_price.csv"
WF_PRICE = DISS_DIR / "per_stock_metrics_price_wf.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def build_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean MAE, RMSE, R², Direction Accuracy per model, ranked by RMSE asc."""
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
    """Mean RMSE per model per sector group."""
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    return (
        df.groupby(["sector", "model_id"])["rmse"]
        .mean()
        .unstack("model_id")
        .reindex(columns=df["model_id"].unique())
    )


def build_comparison(sc_agg: pd.DataFrame, wf_agg: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison of SC vs WF aggregates."""
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


def print_section(title: str) -> None:
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


def _process_target(
    sc_path: Path,
    wf_path: Path,
    label: str,
    sc_out: str,
    wf_out: str,
    comp_out: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Process one target type (return or price). Returns (sc_agg, wf_agg)."""
    has_sc = sc_path.exists()
    has_wf = wf_path.exists()
    sc_agg = wf_agg = None

    if has_sc:
        sc_df = pd.read_csv(sc_path)
        sc_agg = build_aggregate(sc_df)
        sc_agg.to_csv(DISS_DIR / sc_out, index=False)

        print_section(f"{label} — Single-Cutoff Aggregate Leaderboard (ranked by RMSE)")
        print(sc_agg.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / sc_out}")

        print_section(f"{label} — Single-Cutoff Sector Breakdown (Mean RMSE)")
        print(build_sector_table(sc_df).to_string())

    if has_wf:
        wf_df = pd.read_csv(wf_path)
        wf_agg = build_aggregate(wf_df)
        wf_agg.to_csv(DISS_DIR / wf_out, index=False)

        print_section(f"{label} — Walk-Forward Aggregate Leaderboard (ranked by RMSE)")
        print(wf_agg.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / wf_out}")

        print_section(f"{label} — Walk-Forward Sector Breakdown (Mean RMSE)")
        print(build_sector_table(wf_df).to_string())

    if sc_agg is not None and wf_agg is not None:
        print_section(f"{label} — Comparison: SC vs WF")
        comp = build_comparison(sc_agg, wf_agg)
        comp.to_csv(DISS_DIR / comp_out, index=False)
        print(comp.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / comp_out}")
        rank_stability(sc_agg, wf_agg)

    return sc_agg, wf_agg


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    any_found = any(p.exists() for p in (SC_RET, WF_RET, SC_PRICE, WF_PRICE))
    if not any_found:
        print("ERROR: No per-stock metrics CSVs found. Run Script 02 first.")
        sys.exit(1)

    # ── Return target ────────────────────────────────────────────────────────
    ret_sc_agg, ret_wf_agg = _process_target(
        SC_RET, WF_RET, "RETURN",
        "aggregate_leaderboard.csv",
        "aggregate_leaderboard_wf.csv",
        "aggregate_comparison.csv",
    )

    # ── Price target ─────────────────────────────────────────────────────────
    price_sc_agg, price_wf_agg = _process_target(
        SC_PRICE, WF_PRICE, "PRICE",
        "aggregate_leaderboard_price.csv",
        "aggregate_leaderboard_price_wf.csv",
        "aggregate_comparison_price.csv",
    )

    # ── Return vs Price comparison ───────────────────────────────────────────
    # Use single-cutoff aggregates for the comparison (most common baseline)
    if ret_sc_agg is not None and price_sc_agg is not None:
        print_section("RETURN vs PRICE — Single-Cutoff Comparison")

        ret_cols = ret_sc_agg[["model_id", "model_label", "mean_r2",
                                "mean_direction_accuracy"]].copy()
        ret_cols.columns = ["model_id", "model", "ret_r2", "ret_dir_acc"]
        price_cols = price_sc_agg[["model_id", "mean_r2",
                                    "mean_direction_accuracy"]].copy()
        price_cols.columns = ["model_id", "price_r2", "price_dir_acc"]
        rvp = ret_cols.merge(price_cols, on="model_id", how="outer")
        rvp["r2_delta"] = rvp["price_r2"] - rvp["ret_r2"]
        rvp["dir_acc_delta"] = rvp["price_dir_acc"] - rvp["ret_dir_acc"]
        rvp = rvp.sort_values("ret_r2", ascending=False)
        rvp.to_csv(DISS_DIR / "aggregate_return_vs_price.csv", index=False)

        print(rvp.to_string(index=False))
        print(f"\nSaved → {DISS_DIR / 'aggregate_return_vs_price.csv'}")

        ret_rank = list(ret_sc_agg["model_id"])
        price_rank = list(price_sc_agg["model_id"])
        if ret_rank == price_rank:
            print("\n✓ Model ranking is IDENTICAL for return and price targets.")
        else:
            print("\n⚠ Model ranking DIFFERS between return and price targets:")
            print(f"  Return : {ret_rank}")
            print(f"  Price  : {price_rank}")


if __name__ == "__main__":
    main()
