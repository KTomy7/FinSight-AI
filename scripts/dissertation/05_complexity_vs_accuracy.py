#!/usr/bin/env python3
"""
Script 5 — Model Complexity vs. Accuracy (Single Cutoff + Walk-Forward)
========================================================================
Reads per-stock metrics CSVs from Script 02 and:

 1. Assigns an ordinal complexity score to each model.
 2. Produces scatter plots of complexity (x) vs RMSE (y) per ticker,
    for **both** evaluation strategies (side-by-side).
 3. Computes Spearman rank correlation between complexity and RMSE to
    quantify the "model complexity vs. predictive gain" relationship.

Outputs
-------
artifacts/dissertation/plots/complexity_vs_rmse.png           — single cutoff
artifacts/dissertation/plots/complexity_vs_rmse_wf.png        — walk-forward
artifacts/dissertation/plots/complexity_vs_rmse_comparison.png — side-by-side

Run from repo root:
    python scripts/dissertation/05_complexity_vs_accuracy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from finsight.config.settings import get_settings

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
DISS_DIR = Path("artifacts/dissertation")
PLOT_DIR = DISS_DIR / "plots"
MODEL_LABELS = get_settings().model_defaults.id_to_label()

COMPLEXITY = {
    "naive_zero": 1,
    "naive_mean": 2,
    "ridge": 3,
    "hist_gbdt": 4,
    "xgboost": 5,
}

TICKER_MARKERS = {
    "AAPL": ("o", "#007AFF"),
    "JPM": ("s", "#1B5E20"),
    "XOM": ("D", "#E65100"),
    "KO": ("^", "#B71C1C"),
    "TSLA": ("v", "#6A1B9A"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["complexity"] = df["model_id"].map(COMPLEXITY)
    df["model_label"] = df["model_id"].map(MODEL_LABELS)
    return df


def _scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
) -> None:
    """Draw one scatter panel: complexity (x) vs RMSE (y), grouped by ticker."""
    for ticker, (marker, color) in TICKER_MARKERS.items():
        tdf = df[df["ticker"] == ticker]
        ax.scatter(
            tdf["complexity"],
            tdf["rmse"],
            marker=marker,
            color=color,
            s=80,
            label=ticker,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        # Annotate model labels
        for _, row in tdf.iterrows():
            ax.annotate(
                row["model_label"],
                (row["complexity"], row["rmse"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color=color,
                alpha=0.8,
            )

    ax.set_xlabel("Model Complexity Score", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(list(COMPLEXITY.values()))
    ax.set_xticklabels(
        [MODEL_LABELS.get(m, m) for m in COMPLEXITY],
        fontsize=8,
        rotation=25,
        ha="right",
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)


def _single_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Standalone scatter plot for one evaluation approach."""
    fig, ax = plt.subplots(figsize=(10, 7))
    _scatter(ax, df, title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _spearman(df: pd.DataFrame, label: str) -> None:
    """Print Spearman ρ between complexity and RMSE (across all tickers)."""
    rho, p = stats.spearmanr(df["complexity"], df["rmse"])
    print(f"\n  Spearman ρ ({label}): {rho:+.4f}  (p = {p:.4f})")

    # Also per-ticker
    for ticker in sorted(df["ticker"].unique()):
        tdf = df[df["ticker"] == ticker]
        if len(tdf) < 3:
            continue
        rho_t, p_t = stats.spearmanr(tdf["complexity"], tdf["rmse"])
        print(f"    {ticker}: ρ = {rho_t:+.4f}  (p = {p_t:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    sc_path = DISS_DIR / "per_stock_metrics.csv"
    wf_path = DISS_DIR / "per_stock_metrics_wf.csv"
    has_sc = sc_path.exists()
    has_wf = wf_path.exists()

    if not has_sc and not has_wf:
        print("ERROR: No per-stock metrics CSVs found. Run Script 02 first.")
        sys.exit(1)

    sc_df = _prepare(pd.read_csv(sc_path)) if has_sc else None
    wf_df = _prepare(pd.read_csv(wf_path)) if has_wf else None

    # ── Individual plots ─────────────────────────────────────────────────────
    if sc_df is not None:
        print("Single-cutoff scatter plot:")
        _single_plot(
            sc_df,
            PLOT_DIR / "complexity_vs_rmse.png",
            "Model Complexity vs. RMSE [Single Cutoff]",
        )
        _spearman(sc_df, "Single Cutoff — all tickers pooled")

    if wf_df is not None:
        print("\nWalk-forward scatter plot:")
        _single_plot(
            wf_df,
            PLOT_DIR / "complexity_vs_rmse_wf.png",
            "Model Complexity vs. RMSE [Walk-Forward]",
        )
        _spearman(wf_df, "Walk-Forward — all tickers pooled")

    # ── Side-by-side comparison ──────────────────────────────────────────────
    if sc_df is not None and wf_df is not None:
        print("\nSide-by-side comparison plot:")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        _scatter(ax1, sc_df, "Single Cutoff")
        _scatter(ax2, wf_df, "Walk-Forward")
        fig.suptitle(
            "Model Complexity vs. RMSE — Evaluation Strategy Comparison",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        out = PLOT_DIR / "complexity_vs_rmse_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
