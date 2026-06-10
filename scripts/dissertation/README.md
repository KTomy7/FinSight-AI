# Dissertation — Chapter 5 Experimental Scripts

Scripts for reproducing the experimental results in *Chapter 5: Theoretical and
Experimental Results* of the FinSight-AI dissertation on AI-Driven Financial
Market Trend Prediction.

## Experimental design

### Evaluation strategies

| Strategy | Description |
|---|---|
| **Single Cutoff (sc)** | Fixed train/test split at `2024-01-01`. Train on 2021–2023, test on 2024. |
| **Walk-Forward (wf)** | Expanding-window validation with `min_train = 504 days` (~2 yr), `test_window = 21 days` (~1 mo), `step = 21 days`. Test period naturally covers ≈ 2024. |

### Target types

| Target | Column | Description |
|---|---|---|
| **Return** | `target_ret_1d` | Next-day daily return `(close_t+1 / close_t) − 1`. |
| **Price** | `target_price_1d` | Next-day close price `close_t+1`. Same 15 features, different prediction objective. |

This 2 × 2 design produces four experimental conditions:

| | Return | Price |
|---|---|---|
| **Single Cutoff** | `_sc_return` | `_sc_price` |
| **Walk-Forward** | `_wf_return` | `_wf_price` |

All output file names follow this naming convention: `{name}_{strategy}_{target}.{ext}`

### Fixed parameters

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
pip install scipy         # needed by Script 04 (Spearman correlation)
```

## Execution order

Run all scripts **from the repository root directory**.

```bash
# Script 01 — Train all models under all 4 conditions, save metrics + predictions
python scripts/dissertation/01_train_all_models.py

# Script 02 — Generate prediction-vs-actual plots (all 4 combinations)
python scripts/dissertation/02_prediction_plots.py

# Script 03 — Aggregate leaderboard tables, comparisons, text leaderboard
python scripts/dissertation/03_aggregate_leaderboard.py

# Script 04 — Complexity-vs-accuracy scatter plots + Spearman ρ
python scripts/dissertation/04_complexity_vs_accuracy.py
```

> Script 01 does all training and evaluation. Scripts 02–04 only read CSV and
> prediction files produced by Script 01.

## Output artifacts

```
artifacts/
├── runs/                                             # Model artifacts (from training)
│   └── <timestamp>__<model_id>/
│       ├── model.joblib
│       ├── metrics.json
│       ├── manifest.json
│       └── predictions.csv
└── dissertation/
    │
    │  # ── Per-stock metrics (Script 01) ──────────
    ├── per_stock_metrics_sc_return.csv
    ├── per_stock_metrics_wf_return.csv
    ├── per_stock_metrics_sc_price.csv
    ├── per_stock_metrics_wf_price.csv
    │
    │  # ── Predictions (Script 01) ────────────────
    ├── predictions_sc_return/<model_id>.csv
    ├── predictions_wf_return/<model_id>.csv
    ├── predictions_sc_price/<model_id>.csv
    ├── predictions_wf_price/<model_id>.csv
    │
    │  # ── Aggregate tables (Script 03) ───────────
    ├── aggregate_leaderboard_sc_return.csv
    ├── aggregate_leaderboard_wf_return.csv
    ├── aggregate_leaderboard_sc_price.csv
    ├── aggregate_leaderboard_wf_price.csv
    ├── aggregate_comparison_return.csv               — SC vs WF (return)
    ├── aggregate_comparison_price.csv                — SC vs WF (price)
    ├── aggregate_comparison_return_vs_price.csv      — return vs price
    ├── leaderboard.txt                               — formatted text summary
    │
    │  # ── Plots (Scripts 02 & 04) ────────────────
    └── plots/
        │  # Prediction plots (Script 02)
        ├── <TICKER>_prediction_vs_actual_{sc,wf}_{return,price}.png
        ├── <MODEL_ID>_all_tickers_{sc,wf}_{return,price}.png
        │  # Complexity scatter (Script 04)
        ├── complexity_vs_rmse_{sc,wf}_{return,price}.png
        ├── complexity_vs_r2_{sc,wf}_{return,price}.png
        ├── complexity_vs_rmse_comparison_{return,price}.png
        └── complexity_vs_r2_comparison_{return,price}.png
```
