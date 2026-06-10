#!/usr/bin/env python3
"""
Script 4 — Model Complexity vs. Accuracy
==========================================
Reads per-stock metrics CSVs from Script 01 and:

 1. Assigns an ordinal complexity score to each model (1–5).
 2. Scatter plots: complexity (x) vs RMSE/R² (y), per ticker.
 3. Spearman rank correlation (pooled + per-ticker).

All four combinations {sc, wf} × {return, price} are processed automatically.

Outputs  (all under ``artifacts/dissertation/plots/``)
------------------------------------------------------
complexity_vs_rmse_{strategy}_{target}.png
complexity_vs_r2_{strategy}_{target}.png
complexity_vs_rmse_comparison_{target}.png       — side-by-side SC vs WF
complexity_vs_r2_comparison_{target}.png         — side-by-side SC vs WF

Run from repo root:
    python scripts/dissertation/04_complexity_vs_accuracy.py
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

STRATEGIES = ["sc", "wf"]
TARGETS = ["return", "price"]
STRATEGY_LABELS = {"sc": "Single Cutoff", "wf": "Walk-Forward"}
TARGET_LABELS = {"return": "Return", "price": "Price"}

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
def _load(strategy: str, target: str) -> pd.DataFrame | None:
    path = DISS_DIR / f"per_stock_metrics_{strategy}_{target}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
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
    for ticker, (marker, color) in TICKER_MARKERS.items():
        tdf = df[df["ticker"] == ticker]
        ax.scatter(
            tdf["complexity"], tdf[metric],
            marker=marker, color=color, s=80, label=ticker,
            edgecolors="white", linewidths=0.5, zorder=3,
        )
        for _, row in tdf.iterrows():
            ax.annotate(
                row["model_label"],
                (row["complexity"], row[metric]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7, color=color, alpha=0.8,
            )

    ax.set_xlabel("Model Complexity Score", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(list(COMPLEXITY.values()))
    ax.set_xticklabels(
        [MODEL_LABELS.get(m, m) for m in COMPLEXITY],
        fontsize=8, rotation=25, ha="right",
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)


def _single_plot(df: pd.DataFrame, out: Path, title: str,
                 metric: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    _scatter(ax, df, title, metric=metric, y_label=y_label)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _comparison_plot(sc_df: pd.DataFrame, wf_df: pd.DataFrame, out: Path,
                     suptitle: str, metric: str, y_label: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    _scatter(ax1, sc_df, "Single Cutoff", metric=metric, y_label=y_label)
    _scatter(ax2, wf_df, "Walk-Forward", metric=metric, y_label=y_label)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _spearman(df: pd.DataFrame, label: str, metric: str = "rmse") -> None:
    rho, p = stats.spearmanr(df["complexity"], df[metric])
    print(f"\n  Spearman ρ  complexity vs {metric}  ({label}): "
          f"{rho:+.4f}  (p = {p:.4f})")
    for ticker in sorted(df["ticker"].unique()):
        tdf = df[df["ticker"] == ticker]
        if len(tdf) < 3:
            continue
        rho_t, p_t = stats.spearmanr(tdf["complexity"], tdf[metric])
        print(f"    {ticker}: ρ = {rho_t:+.4f}  (p = {p_t:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    for strategy in STRATEGIES:
        for target in TARGETS:
            df = _load(strategy, target)
            if df is not None:
                loaded[(strategy, target)] = df

    if not loaded:
        print("ERROR: No per-stock metrics CSVs found. Run Script 01 first.")
        sys.exit(1)

    metrics = [("rmse", "RMSE"), ("r2", "R²")]

    # ── Individual plots ─────────────────────────────────────────────────────
    for (strategy, target), df in loaded.items():
        stag = STRATEGY_LABELS[strategy]
        ttag = TARGET_LABELS[target]
        for metric, y_label in metrics:
            print(f"\n{stag}, {ttag} — {metric.upper()} scatter:")
            _single_plot(
                df,
                PLOT_DIR / f"complexity_vs_{metric}_{strategy}_{target}.png",
                f"Complexity vs. {y_label} [{stag}, {ttag}]",
                metric=metric,
                y_label=y_label,
            )
            _spearman(df, f"{stag} {ttag} — pooled", metric=metric)

    # ── Side-by-side SC vs WF (per target) ───────────────────────────────────
    for target in TARGETS:
        sc_key = ("sc", target)
        wf_key = ("wf", target)
        if sc_key in loaded and wf_key in loaded:
            ttag = TARGET_LABELS[target]
            for metric, y_label in metrics:
                print(f"\n{ttag} — SC vs WF comparison ({metric.upper()}):")
                _comparison_plot(
                    loaded[sc_key],
                    loaded[wf_key],
                    PLOT_DIR / f"complexity_vs_{metric}_comparison_{target}.png",
                    f"Complexity vs. {y_label} — {ttag} SC vs WF",
                    metric=metric,
                    y_label=y_label,
                )

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
