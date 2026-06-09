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

## Experimental conditions (fixed)

| Parameter | Value |
|---|---|
| Training window | `--years 3` |
| Cutoff date | `2024-01-01` |
| End date | `2024-12-31` |
| Models | `naive_zero`, `naive_mean`, `ridge`, `hist_gbdt`, `xgboost` |
| Tickers | AAPL, JPM, XOM, KO, TSLA |
| Metrics | MAE, RMSE, Direction Accuracy |

## Prerequisites

```bash
# From the repository root
pip install -e .          # or: pip install -r requirements.txt
pip install scipy         # needed by Script 05 (Spearman correlation)
```

## Execution order

Run all scripts **from the repository root directory**.

```bash
# Step 1 — Train all models (single cutoff) and print leaderboard
bash scripts/dissertation/01_run_all_training.sh

# Step 2 — Compute per-stock metrics (single cutoff + walk-forward)
python scripts/dissertation/02_per_stock_metrics.py

# Step 3 — Generate prediction-vs-actual plots (both strategies)
python scripts/dissertation/03_prediction_plots.py

# Step 4 — Aggregate leaderboard tables + comparison (both strategies)
python scripts/dissertation/04_aggregate_leaderboard.py

# Step 5 — Complexity-vs-accuracy scatter plots + Spearman ρ (both strategies)
python scripts/dissertation/05_complexity_vs_accuracy.py
```

> **Note:** Script 01 runs the CLI-based training and saves artifacts to
> `artifacts/runs/`. Script 02 re-trains via the Python API (same conditions)
> and additionally runs walk-forward validation. Scripts 03–05 only read
> artifacts and CSV files produced by the earlier scripts.

## Output artifacts

```
artifacts/
├── runs/                                   # Model run artifacts (Script 01 & 02)
│   └── <timestamp>__<model_id>/
│       ├── model.joblib
│       ├── metrics.json
│       ├── manifest.json
│       └── predictions.csv
└── dissertation/
    ├── leaderboard.txt                     # CLI compare output (Script 01)
    ├── per_stock_metrics.csv               # Per-ticker metrics — single cutoff (Script 02)
    ├── per_stock_metrics_wf.csv            # Per-ticker metrics — walk-forward (Script 02)
    ├── aggregate_leaderboard.csv           # Aggregate leaderboard — single cutoff (Script 04)
    ├── aggregate_leaderboard_wf.csv        # Aggregate leaderboard — walk-forward (Script 04)
    ├── aggregate_comparison.csv            # Side-by-side comparison table (Script 04)
    ├── wf_predictions/                     # Walk-forward test predictions (Script 02)
    │   └── <model_id>.csv
    └── plots/                              # All figures (Scripts 03 & 05)
        ├── <TICKER>_prediction_vs_actual.png
        ├── <TICKER>_prediction_vs_actual_wf.png
        ├── <MODEL_ID>_all_tickers.png
        ├── <MODEL_ID>_all_tickers_wf.png
        ├── complexity_vs_rmse.png
        ├── complexity_vs_rmse_wf.png
        └── complexity_vs_rmse_comparison.png
```
