#!/usr/bin/env python3
"""
Script 5 — Model Complexity vs. Accuracy (Return & Price, SC + WF)
====================================================================
Reads per-stock metrics CSVs from Script 02 and:

 1. Assigns an ordinal complexity score to each model.
 2. Produces scatter plots of complexity (x) vs RMSE (y) and complexity (x) vs
    R² (y) per ticker, for **both** evaluation strategies and **both** target
    types (return and price).
 3. Computes Spearman rank correlation between complexity and RMSE/R² to
    quantify the "model complexity vs. predictive gain" relationship.

Outputs
-------
artifacts/dissertation/plots/complexity_vs_rmse.png                  — return SC
artifacts/dissertation/plots/complexity_vs_rmse_wf.png               — return WF
artifacts/dissertation/plots/complexity_vs_rmse_comparison.png       — return side-by-side
artifacts/dissertation/plots/complexity_vs_r2.png                    — return SC (R²)
artifacts/dissertation/plots/complexity_vs_r2_wf.png                 — return WF (R²)
artifacts/dissertation/plots/complexity_vs_r2_comparison.png         — return side-by-side (R²)
artifacts/dissertation/plots/complexity_vs_rmse_price.png            — price SC
artifacts/dissertation/plots/complexity_vs_rmse_price_wf.png         — price WF
artifacts/dissertation/plots/complexity_vs_rmse_price_comparison.png — price side-by-side
artifacts/dissertation/plots/complexity_vs_r2_price.png              — price SC (R²)
artifacts/dissertation/plots/complexity_vs_r2_price_wf.png           — price WF (R²)
artifacts/dissertation/plots/complexity_vs_r2_price_comparison.png   — price side-by-side (R²)

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

# Metrics CSV paths
METRICS_FILES = {
    "ret_sc": DISS_DIR / "per_stock_metrics.csv",
    "ret_wf": DISS_DIR / "per_stock_metrics_wf.csv",
    "price_sc": DISS_DIR / "per_stock_metrics_price.csv",
    "price_wf": DISS_DIR / "per_stock_metrics_price_wf.csv",
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
    metric: str = "rmse",
    y_label: str = "RMSE",
) -> None:
    """Draw one scatter panel: complexity (x) vs metric (y), grouped by ticker."""
    for ticker, (marker, color) in TICKER_MARKERS.items():
        tdf = df[df["ticker"] == ticker]
        ax.scatter(
            tdf["complexity"],
            tdf[metric],
            marker=marker,
            color=color,
            s=80,
            label=ticker,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        for _, row in tdf.iterrows():
            ax.annotate(
                row["model_label"],
                (row["complexity"], row[metric]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color=color,
                alpha=0.8,
            )

    ax.set_xlabel("Model Complexity Score", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
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


def _single_plot(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    metric: str = "rmse",
    y_label: str = "RMSE",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    _scatter(ax, df, title, metric=metric, y_label=y_label)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _comparison_plot(
    sc_df: pd.DataFrame,
    wf_df: pd.DataFrame,
    out_path: Path,
    suptitle: str,
    metric: str = "rmse",
    y_label: str = "RMSE",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    _scatter(ax1, sc_df, "Single Cutoff", metric=metric, y_label=y_label)
    _scatter(ax2, wf_df, "Walk-Forward", metric=metric, y_label=y_label)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _spearman(df: pd.DataFrame, label: str, metric: str = "rmse") -> None:
    rho, p = stats.spearmanr(df["complexity"], df[metric])
    print(f"\n  Spearman ρ  complexity vs {metric}  ({label}): {rho:+.4f}  (p = {p:.4f})")
    for ticker in sorted(df["ticker"].unique()):
        tdf = df[df["ticker"] == ticker]
        if len(tdf) < 3:
            continue
        rho_t, p_t = stats.spearmanr(tdf["complexity"], tdf[metric])
        print(f"    {ticker}: ρ = {rho_t:+.4f}  (p = {p_t:.4f})")


def _generate_plots(
    sc_df: pd.DataFrame | None,
    wf_df: pd.DataFrame | None,
    target_label: str,
    suffix: str,
) -> None:
    """Generate RMSE and R² plots for one target type (return or price)."""

    for metric, y_label in [("rmse", "RMSE"), ("r2", "R²")]:
        metric_suffix = f"_{metric}" if metric != "rmse" else ""

        if sc_df is not None:
            print(f"\n{target_label} — SC scatter ({metric.upper()}):")
            _single_plot(
                sc_df,
                PLOT_DIR / f"complexity_vs_{metric}{suffix}.png",
                f"Complexity vs. {y_label} [{target_label}, Single Cutoff]",
                metric=metric,
                y_label=y_label,
            )
            _spearman(sc_df, f"{target_label} SC — pooled", metric=metric)

        if wf_df is not None:
            print(f"\n{target_label} — WF scatter ({metric.upper()}):")
            _single_plot(
                wf_df,
                PLOT_DIR / f"complexity_vs_{metric}{suffix}_wf.png",
                f"Complexity vs. {y_label} [{target_label}, Walk-Forward]",
                metric=metric,
                y_label=y_label,
            )
            _spearman(wf_df, f"{target_label} WF — pooled", metric=metric)

        if sc_df is not None and wf_df is not None:
            print(f"\n{target_label} — side-by-side ({metric.upper()}):")
            _comparison_plot(
                sc_df,
                wf_df,
                PLOT_DIR / f"complexity_vs_{metric}{suffix}_comparison.png",
                f"Complexity vs. {y_label} — {target_label} SC vs WF",
                metric=metric,
                y_label=y_label,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, pd.DataFrame | None] = {}
    for key, path in METRICS_FILES.items():
        loaded[key] = _prepare(pd.read_csv(path)) if path.exists() else None

    if all(v is None for v in loaded.values()):
        print("ERROR: No per-stock metrics CSVs found. Run Script 02 first.")
        sys.exit(1)

    # ── Return target plots ──────────────────────────────────────────────────
    _generate_plots(loaded["ret_sc"], loaded["ret_wf"], "Return", "")

    # ── Price target plots ───────────────────────────────────────────────────
    _generate_plots(loaded["price_sc"], loaded["price_wf"], "Price", "_price")

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
