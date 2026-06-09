# Dissertation — Chapter 5 Experimental Scripts

Scripts for reproducing the experimental results in *Chapter 5: Theoretical and
Experimental Results* of the FinSight-AI dissertation on AI-Driven Financial
Market Trend Prediction.

## Evaluation strategies

Every script produces results for **two** evaluation strategies so they can be
compared directly:

| Strategy | Description |
|---|---|
| **Single Cutoff** | Fixed train/test split at `2024-01-01`. Train on 2021–2023, test on 2024. |
| **Walk-Forward** | Expanding-window validation with `min_train = 504 days` (~2 yr), `test_window = 21 days` (~1 mo), `step = 21 days`. Test period naturally covers ≈ 2024, making it directly comparable with the single cutoff. |

## Target types

Each evaluation strategy is run with **two** prediction targets:

| Target | Column | Description |
|---|---|---|
| **Return** | `target_ret_1d` | Next-day daily return `(close_t+1 / close_t) − 1`. |
| **Price** | `target_price_1d` | Next-day close price `close_t+1`. Same features, different prediction objective. |

This 2 × 2 design (strategy × target) lets you compare how model rankings and
metrics change between return-based and price-based prediction.

## Experimental conditions (fixed)

| Parameter | Value |
|---|---|
| Training window | `--years 3` |
| Cutoff date | `2024-01-01` |
| End date | `2024-12-31` |
| Models | `naive_zero`, `naive_mean`, `ridge`, `hist_gbdt`, `xgboost` |
| Tickers | AAPL, JPM, XOM, KO, TSLA |
| Metrics | MAE, RMSE, R², Direction Accuracy |

> **Note on direction accuracy for price targets:** since raw price values are
> always positive, direction accuracy is computed from the *implied price
> movement* — i.e. whether the model correctly predicted if the price went up
> or down relative to the current-day close.

## Prerequisites

```bash
# From the repository root
pip install -e .          # or: pip install -r requirements.txt
pip install scipy         # needed by Script 05 (Spearman correlation)
```

## Execution order

Run all scripts **from the repository root directory**.

```bash
# Step 1 — Train all models (single cutoff, return target) and print leaderboard
bash scripts/dissertation/01_run_all_training.sh

# Step 2 — Compute per-stock metrics (SC + WF, return + price targets)
python scripts/dissertation/02_per_stock_metrics.py

# Step 3 — Generate prediction-vs-actual plots (all 4 combinations)
python scripts/dissertation/03_prediction_plots.py

# Step 4 — Aggregate leaderboard tables + comparisons
python scripts/dissertation/04_aggregate_leaderboard.py

# Step 5 — Complexity-vs-accuracy scatter plots + Spearman ρ
python scripts/dissertation/05_complexity_vs_accuracy.py
```

> **Note:** Script 01 runs the CLI-based training and saves artifacts to
> `artifacts/runs/`. Script 02 re-trains via the Python API (same conditions)
> and additionally runs walk-forward validation and price-level prediction.
> Scripts 03–05 only read artifacts and CSV files produced by the earlier scripts.

## Output artifacts

```
artifacts/
├── runs/                                         # Model run artifacts (Script 01 & 02)
│   └── <timestamp>__<model_id>/
│       ├── model.joblib
│       ├── metrics.json
│       ├── manifest.json
│       └── predictions.csv
└── dissertation/
    ├── leaderboard.txt                           # CLI compare output (Script 01)
    │
    │  # ── Return target ──────────────────────
    ├── per_stock_metrics.csv                     # Per-ticker — return SC (Script 02)
    ├── per_stock_metrics_wf.csv                  # Per-ticker — return WF (Script 02)
    ├── aggregate_leaderboard.csv                 # Aggregate  — return SC (Script 04)
    ├── aggregate_leaderboard_wf.csv              # Aggregate  — return WF (Script 04)
    ├── aggregate_comparison.csv                  # SC vs WF   — return    (Script 04)
    ├── wf_predictions/                           # WF test predictions — return
    │   └── <model_id>.csv
    │
    │  # ── Price target ───────────────────────
    ├── per_stock_metrics_price.csv               # Per-ticker — price SC  (Script 02)
    ├── per_stock_metrics_price_wf.csv            # Per-ticker — price WF  (Script 02)
    ├── aggregate_leaderboard_price.csv           # Aggregate  — price SC  (Script 04)
    ├── aggregate_leaderboard_price_wf.csv        # Aggregate  — price WF  (Script 04)
    ├── aggregate_comparison_price.csv            # SC vs WF   — price     (Script 04)
    ├── sc_predictions_price/                     # SC test predictions — price
    │   └── <model_id>.csv
    ├── wf_predictions_price/                     # WF test predictions — price
    │   └── <model_id>.csv
    │
    │  # ── Cross-target comparison ────────────
    ├── aggregate_return_vs_price.csv             # Return vs price R²/DirAcc (Script 04)
    │
    │  # ── Plots ──────────────────────────────
    └── plots/
        │  # Return prediction plots
        ├── <TICKER>_prediction_vs_actual.png
        ├── <TICKER>_prediction_vs_actual_wf.png
        ├── <MODEL_ID>_all_tickers.png
        ├── <MODEL_ID>_all_tickers_wf.png
        │  # Price prediction plots
        ├── <TICKER>_prediction_vs_actual_price.png
        ├── <TICKER>_prediction_vs_actual_price_wf.png
        ├── <MODEL_ID>_all_tickers_price.png
        ├── <MODEL_ID>_all_tickers_price_wf.png
        │  # Complexity scatter — return
        ├── complexity_vs_rmse.png
        ├── complexity_vs_rmse_wf.png
        ├── complexity_vs_rmse_comparison.png
        ├── complexity_vs_r2.png
        ├── complexity_vs_r2_wf.png
        ├── complexity_vs_r2_comparison.png
        │  # Complexity scatter — price
        ├── complexity_vs_rmse_price.png
        ├── complexity_vs_rmse_price_wf.png
        ├── complexity_vs_rmse_price_comparison.png
        ├── complexity_vs_r2_price.png
        ├── complexity_vs_r2_price_wf.png
        └── complexity_vs_r2_price_comparison.png
```
