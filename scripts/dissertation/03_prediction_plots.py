#!/usr/bin/env python3
"""
Script 3 — Prediction vs. Actual Plots (Single Cutoff + Walk-Forward)
======================================================================
After training has been run (Scripts 01 & 02), loads backtest predictions and
produces:

  Per-ticker figures  — 5 subplots (one per model) with predicted vs actual
                        daily return on the test set.
  Per-model figures   — all 5 tickers on one figure.

Each plot type is generated for **both** the single-cutoff and walk-forward
approaches so the reader can visually compare the two evaluation strategies.

Outputs
-------
artifacts/dissertation/plots/<TICKER>_prediction_vs_actual.png
artifacts/dissertation/plots/<TICKER>_prediction_vs_actual_wf.png
artifacts/dissertation/plots/<MODEL_ID>_all_tickers.png
artifacts/dissertation/plots/<MODEL_ID>_all_tickers_wf.png

Run from repo root:
    python scripts/dissertation/03_prediction_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from finsight.config.settings import get_settings
from finsight.domain.metrics import forecast_metrics
from finsight.infrastructure.ml.registry import LocalFileModelRegistry

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
TICKERS = ["AAPL", "JPM", "XOM", "KO", "TSLA"]
MODEL_IDS = ["naive_zero", "naive_mean", "ridge", "hist_gbdt", "xgboost"]
ARTIFACTS_DIR = "artifacts/runs"
DISS_DIR = Path("artifacts/dissertation")
PLOT_DIR = DISS_DIR / "plots"
WF_PRED_DIR = DISS_DIR / "wf_predictions"

MODEL_LABELS = get_settings().model_defaults.id_to_label()

COLORS = {
    "actual": "#333333",
    "predicted": "#2196F3",
}
TICKER_COLORS = {
    "AAPL": "#007AFF",
    "JPM": "#1B5E20",
    "XOM": "#E65100",
    "KO": "#B71C1C",
    "TSLA": "#6A1B9A",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_single_cutoff_predictions() -> dict[str, pd.DataFrame]:
    """Load predictions from artifacts/runs/ for each model_id."""
    registry = LocalFileModelRegistry()
    preds_by_model: dict[str, pd.DataFrame] = {}
    for model_id in MODEL_IDS:
        run_id = registry.latest_run_id(artifact_root=ARTIFACTS_DIR, model_id=model_id)
        path = Path(ARTIFACTS_DIR) / run_id / "predictions.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        preds_by_model[model_id] = df
    return preds_by_model


def load_walk_forward_predictions() -> dict[str, pd.DataFrame]:
    """Load walk-forward predictions saved by Script 02."""
    preds_by_model: dict[str, pd.DataFrame] = {}
    for model_id in MODEL_IDS:
        path = WF_PRED_DIR / f"{model_id}.csv"
        if not path.exists():
            print(f"  ⚠ Walk-forward predictions not found: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        preds_by_model[model_id] = df
    return preds_by_model


# ═══════════════════════════════════════════════════════════════════════════════
# Plot builders
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_ticker(
    preds_by_model: dict[str, pd.DataFrame],
    suffix: str = "",
    title_extra: str = "",
) -> None:
    """
    For each ticker, create a figure with 5 subplots (one per model).
    Each subplot: predicted return vs actual return over the test dates.
    """
    for ticker in TICKERS:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        axes_flat = axes.flatten()
        fig.suptitle(
            f"{ticker} — Predicted vs Actual Daily Return{title_extra}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        for idx, model_id in enumerate(MODEL_IDS):
            ax = axes_flat[idx]
            df = preds_by_model.get(model_id)
            if df is None:
                ax.set_title(f"{MODEL_LABELS.get(model_id, model_id)} (no data)")
                continue
            tdf = df[df["ticker"] == ticker].sort_values("date")
            if tdf.empty:
                ax.set_title(f"{MODEL_LABELS.get(model_id, model_id)} (no data)")
                continue

            ax.plot(tdf["date"], tdf["y_true"], color=COLORS["actual"],
                    linewidth=0.8, alpha=0.7, label="Actual")
            ax.plot(tdf["date"], tdf["y_pred"], color=COLORS["predicted"],
                    linewidth=0.8, alpha=0.7, label="Predicted")
            ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
            # Annotate metrics
            m = forecast_metrics(tdf["y_true"].tolist(), tdf["y_pred"].tolist())
            metrics_text = (f"MAE={m['mae']:.5f}  RMSE={m['rmse']:.5f}\n"
                            f"R²={m['r2']:.4f}  DirAcc={m['direction_accuracy']:.1%}")
            ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
            ax.set_title(f"{MODEL_LABELS.get(model_id, model_id)} — {ticker}",
                         fontsize=11)
            ax.set_ylabel("Daily Return", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        # Hide unused subplot (6th cell in 3x2 grid)
        axes_flat[5].set_visible(False)

        fig.autofmt_xdate(rotation=30)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = PLOT_DIR / f"{ticker}_prediction_vs_actual{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def plot_per_model(
    preds_by_model: dict[str, pd.DataFrame],
    suffix: str = "",
    title_extra: str = "",
) -> None:
    """
    For each model, create a figure with 5 subplots (one per ticker).
    """
    for model_id in MODEL_IDS:
        df = preds_by_model.get(model_id)
        if df is None:
            continue

        label = MODEL_LABELS.get(model_id, model_id)
        fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
        fig.suptitle(
            f"{label} — All Tickers{title_extra}",
            fontsize=15,
            fontweight="bold",
            y=0.99,
        )

        for idx, ticker in enumerate(TICKERS):
            ax = axes[idx]
            tdf = df[df["ticker"] == ticker].sort_values("date")
            if tdf.empty:
                ax.set_title(f"{ticker} (no data)")
                continue

            tc = TICKER_COLORS.get(ticker, "#333")
            ax.plot(tdf["date"], tdf["y_true"], color=COLORS["actual"],
                    linewidth=0.8, alpha=0.6, label="Actual")
            ax.plot(tdf["date"], tdf["y_pred"], color=tc,
                    linewidth=0.8, alpha=0.7, label=f"Predicted ({ticker})")
            ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
            # Annotate metrics
            m = forecast_metrics(tdf["y_true"].tolist(), tdf["y_pred"].tolist())
            metrics_text = (f"MAE={m['mae']:.5f}  RMSE={m['rmse']:.5f}\n"
                            f"R²={m['r2']:.4f}  DirAcc={m['direction_accuracy']:.1%}")
            ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
            ax.set_title(f"{label} — {ticker}", fontsize=11)
            ax.set_ylabel("Daily Return", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        fig.autofmt_xdate(rotation=30)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = PLOT_DIR / f"{model_id}_all_tickers{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Single Cutoff ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading single-cutoff predictions …")
    print("=" * 60)
    sc_preds = load_single_cutoff_predictions()

    print("\nPer-ticker plots (single cutoff):")
    plot_per_ticker(sc_preds, suffix="", title_extra=" [Single Cutoff]")

    print("\nPer-model plots (single cutoff):")
    plot_per_model(sc_preds, suffix="", title_extra=" [Single Cutoff]")

    # ── Walk-Forward ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading walk-forward predictions …")
    print("=" * 60)
    wf_preds = load_walk_forward_predictions()

    if wf_preds:
        print("\nPer-ticker plots (walk-forward):")
        plot_per_ticker(wf_preds, suffix="_wf", title_extra=" [Walk-Forward]")

        print("\nPer-model plots (walk-forward):")
        plot_per_model(wf_preds, suffix="_wf", title_extra=" [Walk-Forward]")
    else:
        print("  ⚠ No walk-forward predictions found. Run Script 02 first.")

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
