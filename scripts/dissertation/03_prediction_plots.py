#!/usr/bin/env python3
"""
Script 3 — Prediction vs. Actual Plots (Return & Price, SC + WF)
=================================================================
After training has been run (Scripts 01 & 02), loads backtest predictions and
produces:

  Per-ticker figures  — 5 subplots (one per model) with predicted vs actual
                        on the test set.
  Per-model figures   — all 5 tickers on one figure.

Each plot type is generated for **both** the single-cutoff and walk-forward
approaches, and for **both** return and price-level targets.

Outputs
-------
artifacts/dissertation/plots/<TICKER>_prediction_vs_actual.png
artifacts/dissertation/plots/<TICKER>_prediction_vs_actual_wf.png
artifacts/dissertation/plots/<TICKER>_prediction_vs_actual_price.png
artifacts/dissertation/plots/<TICKER>_prediction_vs_actual_price_wf.png
artifacts/dissertation/plots/<MODEL_ID>_all_tickers.png
artifacts/dissertation/plots/<MODEL_ID>_all_tickers_wf.png
artifacts/dissertation/plots/<MODEL_ID>_all_tickers_price.png
artifacts/dissertation/plots/<MODEL_ID>_all_tickers_price_wf.png

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
SC_PRED_PRICE_DIR = DISS_DIR / "sc_predictions_price"
WF_PRED_PRICE_DIR = DISS_DIR / "wf_predictions_price"

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
    """Load return predictions from artifacts/runs/ for each model_id."""
    registry = LocalFileModelRegistry()
    preds_by_model: dict[str, pd.DataFrame] = {}
    for model_id in MODEL_IDS:
        run_id = registry.latest_run_id(artifact_root=ARTIFACTS_DIR, model_id=model_id)
        path = Path(ARTIFACTS_DIR) / run_id / "predictions.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        preds_by_model[model_id] = df
    return preds_by_model


def _load_from_dir(pred_dir: Path) -> dict[str, pd.DataFrame]:
    """Load predictions CSVs from a directory (one per model)."""
    preds: dict[str, pd.DataFrame] = {}
    for model_id in MODEL_IDS:
        path = pred_dir / f"{model_id}.csv"
        if not path.exists():
            print(f"  ⚠ Predictions not found: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        preds[model_id] = df
    return preds


def load_walk_forward_predictions() -> dict[str, pd.DataFrame]:
    return _load_from_dir(WF_PRED_DIR)


def load_price_sc_predictions() -> dict[str, pd.DataFrame]:
    return _load_from_dir(SC_PRED_PRICE_DIR)


def load_price_wf_predictions() -> dict[str, pd.DataFrame]:
    return _load_from_dir(WF_PRED_PRICE_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _subplot_metrics(tdf: pd.DataFrame, is_price: bool = False) -> dict[str, float]:
    """Compute metrics for one subplot's data slice."""
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
    suffix: str = "",
    title_extra: str = "",
    y_label: str = "Daily Return",
    is_price: bool = False,
) -> None:
    """
    For each ticker, create a figure with 5 subplots (one per model).
    Each subplot: predicted value vs actual value over the test dates.
    """
    value_kind = "Next-Day Close Price" if is_price else "Daily Return"
    for ticker in TICKERS:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        axes_flat = axes.flatten()
        fig.suptitle(
            f"{ticker} — Predicted vs Actual {value_kind}{title_extra}",
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
            if not is_price:
                ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
            # Annotate metrics
            m = _subplot_metrics(tdf, is_price=is_price)
            ax.text(0.02, 0.97, _metrics_text(m, is_price=is_price),
                    transform=ax.transAxes, fontsize=7, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
            ax.set_title(f"{MODEL_LABELS.get(model_id, model_id)} — {ticker}",
                         fontsize=11)
            ax.set_ylabel(y_label, fontsize=9)
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
    y_label: str = "Daily Return",
    is_price: bool = False,
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
            if not is_price:
                ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
            # Annotate metrics
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

    plot_kwargs_return = dict(y_label="Daily Return", is_price=False)
    plot_kwargs_price = dict(y_label="Close Price ($)", is_price=True)

    # ── Return — Single Cutoff ───────────────────────────────────────────────
    print("=" * 60)
    print("Loading single-cutoff predictions (return) …")
    print("=" * 60)
    sc_preds = load_single_cutoff_predictions()

    print("\nPer-ticker plots (return, single cutoff):")
    plot_per_ticker(sc_preds, suffix="", title_extra=" [Single Cutoff]", **plot_kwargs_return)
    print("\nPer-model plots (return, single cutoff):")
    plot_per_model(sc_preds, suffix="", title_extra=" [Single Cutoff]", **plot_kwargs_return)

    # ── Return — Walk-Forward ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading walk-forward predictions (return) …")
    print("=" * 60)
    wf_preds = load_walk_forward_predictions()
    if wf_preds:
        print("\nPer-ticker plots (return, walk-forward):")
        plot_per_ticker(wf_preds, suffix="_wf", title_extra=" [Walk-Forward]", **plot_kwargs_return)
        print("\nPer-model plots (return, walk-forward):")
        plot_per_model(wf_preds, suffix="_wf", title_extra=" [Walk-Forward]", **plot_kwargs_return)
    else:
        print("  ⚠ No walk-forward return predictions found. Run Script 02 first.")

    # ── Price — Single Cutoff ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading single-cutoff predictions (price) …")
    print("=" * 60)
    price_sc = load_price_sc_predictions()
    if price_sc:
        print("\nPer-ticker plots (price, single cutoff):")
        plot_per_ticker(price_sc, suffix="_price", title_extra=" [Single Cutoff]", **plot_kwargs_price)
        print("\nPer-model plots (price, single cutoff):")
        plot_per_model(price_sc, suffix="_price", title_extra=" [Single Cutoff]", **plot_kwargs_price)
    else:
        print("  ⚠ No price single-cutoff predictions found. Run Script 02 first.")

    # ── Price — Walk-Forward ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading walk-forward predictions (price) …")
    print("=" * 60)
    price_wf = load_price_wf_predictions()
    if price_wf:
        print("\nPer-ticker plots (price, walk-forward):")
        plot_per_ticker(price_wf, suffix="_price_wf", title_extra=" [Walk-Forward]", **plot_kwargs_price)
        print("\nPer-model plots (price, walk-forward):")
        plot_per_model(price_wf, suffix="_price_wf", title_extra=" [Walk-Forward]", **plot_kwargs_price)
    else:
        print("  ⚠ No price walk-forward predictions found. Run Script 02 first.")

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
