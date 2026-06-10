#!/usr/bin/env python3
"""
Script 2 — Prediction vs. Actual Plots
========================================
Loads prediction CSVs produced by Script 01 and generates:

  Per-ticker figures  — 5 subplots (one per model), predicted vs actual.
  Per-model figures   — 5 subplots (one per ticker).

Generated for all four combinations:
  {sc, wf} × {return, price}

Outputs  (all under ``artifacts/dissertation/plots/``)
------------------------------------------------------
<TICKER>_prediction_vs_actual_{strategy}_{target}.png
<MODEL_ID>_all_tickers_{strategy}_{target}.png

Run from repo root:
    python scripts/dissertation/02_prediction_plots.py
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

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
TICKERS = ["AAPL", "JPM", "XOM", "KO", "TSLA"]
MODEL_IDS = ["naive_zero", "naive_mean", "ridge", "hist_gbdt", "xgboost"]
DISS_DIR = Path("artifacts/dissertation")
PLOT_DIR = DISS_DIR / "plots"
MODEL_LABELS = get_settings().model_defaults.id_to_label()

STRATEGIES = ["sc", "wf"]
TARGETS = ["return", "price"]
STRATEGY_LABELS = {"sc": "Single Cutoff", "wf": "Walk-Forward"}
TARGET_LABELS = {"return": "Return", "price": "Price"}

COLORS = {"actual": "#333333", "predicted": "#2196F3"}
TICKER_COLORS = {
    "AAPL": "#007AFF", "JPM": "#1B5E20", "XOM": "#E65100",
    "KO": "#B71C1C", "TSLA": "#6A1B9A",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_predictions(strategy: str, target: str) -> dict[str, pd.DataFrame]:
    pred_dir = DISS_DIR / f"predictions_{strategy}_{target}"
    preds: dict[str, pd.DataFrame] = {}
    for model_id in MODEL_IDS:
        path = pred_dir / f"{model_id}.csv"
        if not path.exists():
            print(f"  ⚠ Missing: {path}")
            continue
        preds[model_id] = pd.read_csv(path, parse_dates=["date"])
    return preds


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _subplot_metrics(tdf: pd.DataFrame, is_price: bool = False) -> dict[str, float]:
    m = forecast_metrics(tdf["y_true"].tolist(), tdf["y_pred"].tolist())
    if is_price and "close" in tdf.columns and tdf["close"].notna().all():
        actual_up = tdf["y_true"].values > tdf["close"].values
        pred_up = tdf["y_pred"].values > tdf["close"].values
        m["direction_accuracy"] = float((actual_up == pred_up).mean())
    return m


def _metrics_text(m: dict[str, float], is_price: bool = False) -> str:
    if is_price:
        return (f"MAE=${m['mae']:.2f}  RMSE=${m['rmse']:.2f}\n"
                f"R²={m['r2']:.4f}  DirAcc={m['direction_accuracy']:.1%}")
    return (f"MAE={m['mae']:.5f}  RMSE={m['rmse']:.5f}\n"
            f"R²={m['r2']:.4f}  DirAcc={m['direction_accuracy']:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot builders
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_ticker(
    preds_by_model: dict[str, pd.DataFrame],
    strategy: str,
    target: str,
) -> None:
    is_price = target == "price"
    y_label = "Close Price ($)" if is_price else "Daily Return"
    strat_label = STRATEGY_LABELS[strategy]
    tgt_label = TARGET_LABELS[target]
    suffix = f"_{strategy}_{target}"

    for ticker in TICKERS:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        axes_flat = axes.flatten()
        fig.suptitle(
            f"{ticker} — Predicted vs Actual [{strat_label}, {tgt_label}]",
            fontsize=15, fontweight="bold", y=0.98,
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
            if not is_price:
                ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")

            m = _subplot_metrics(tdf, is_price=is_price)
            ax.text(0.02, 0.97, _metrics_text(m, is_price=is_price),
                    transform=ax.transAxes, fontsize=7, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
            ax.set_title(f"{MODEL_LABELS.get(model_id, model_id)} — {ticker}", fontsize=11)
            ax.set_ylabel(y_label, fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        axes_flat[5].set_visible(False)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = PLOT_DIR / f"{ticker}_prediction_vs_actual{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def plot_per_model(
    preds_by_model: dict[str, pd.DataFrame],
    strategy: str,
    target: str,
) -> None:
    is_price = target == "price"
    y_label = "Close Price ($)" if is_price else "Daily Return"
    strat_label = STRATEGY_LABELS[strategy]
    tgt_label = TARGET_LABELS[target]
    suffix = f"_{strategy}_{target}"

    for model_id in MODEL_IDS:
        df = preds_by_model.get(model_id)
        if df is None:
            continue

        label = MODEL_LABELS.get(model_id, model_id)
        fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
        fig.suptitle(
            f"{label} — All Tickers [{strat_label}, {tgt_label}]",
            fontsize=15, fontweight="bold", y=0.99,
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
            if not is_price:
                ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")

            m = _subplot_metrics(tdf, is_price=is_price)
            ax.text(0.02, 0.97, _metrics_text(m, is_price=is_price),
                    transform=ax.transAxes, fontsize=7, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
            ax.set_title(f"{label} — {ticker}", fontsize=11)
            ax.set_ylabel(y_label, fontsize=9)
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

    for strategy in STRATEGIES:
        for target in TARGETS:
            tag = f"{STRATEGY_LABELS[strategy]}, {TARGET_LABELS[target]}"
            print(f"\n{'=' * 60}")
            print(f"Loading predictions: {tag}")
            print("=" * 60)

            preds = load_predictions(strategy, target)
            if not preds:
                print(f"  ⚠ No predictions found for {tag}. Run Script 01 first.")
                continue

            print(f"\nPer-ticker plots ({tag}):")
            plot_per_ticker(preds, strategy, target)
            print(f"\nPer-model plots ({tag}):")
            plot_per_model(preds, strategy, target)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
