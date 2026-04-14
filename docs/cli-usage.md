# CLI Usage Guide

The FinSight CLI provides three main commands for training models, comparing runs, and generating forecasts.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure tickers and models** (optional):
   Edit `config/config.yaml` to customize ticker catalog and model defaults.

---

## Commands

### `train` — Train and evaluate models

Trains baseline models on fixed tickers using historical market data.

**Syntax:**
```bash
finsight train --cutoff CUTOFF_DATE [OPTIONS]
```

**Required Arguments:**
- `--cutoff CUTOFF_DATE` — Global time-split cutoff date (ISO format: `YYYY-MM-DD`). Data before this date is training; after is testing.

**Optional Arguments:**
- `--years N` — Lookback window in years (default: `2`)
- `--end END_DATE` — Inclusive end date for data fetch (default: today)
- `--model-types TYPE [TYPE ...]` — Model IDs to train (default: all training-enabled models)
- `--artifacts-dir DIR` — Directory for run artifacts (default: `artifacts/runs`)

**Example:**
```bash
finsight train --cutoff 2025-06-01 --years 3 --model-types naive_zero ridge
```

**Output:**
```
[naive_zero] run_dir=artifacts/runs/2026-04-14T120000Z__naive_zero
  MAE=0.015234 RMSE=0.018567 DirectionAcc=0.5432
[ridge] run_dir=artifacts/runs/2026-04-14T120000Z__ridge
  MAE=0.012456 RMSE=0.015678 DirectionAcc=0.6234
```

---

### `compare` — Compare trained model runs

Ranks trained models by performance metrics and prints a leaderboard.

**Syntax:**
```bash
finsight compare [OPTIONS]
```

**Optional Arguments:**
- `--model-ids ID [ID ...]` — Model IDs to compare (default: all training-enabled models)
- `--rank-by METRIC [METRIC ...]` — Metrics for ranking (default: `mae rmse direction_accuracy`)
- `--artifacts-dir DIR` — Directory containing model artifacts (default: `artifacts/runs`)

**Available metrics:**
- `mae` — Mean Absolute Error (lower is better)
- `rmse` — Root Mean Squared Error (lower is better)
- `direction_accuracy` — Proportion of correct direction predictions (higher is better)

**Example:**
```bash
finsight compare --model-ids naive_zero ridge --rank-by mae direction_accuracy
```

**Output:**
```
rank                model        mae   rmse  direction_accuracy
   1       Naive (Zero)  0.015234 0.0186              0.5432
   2  Ridge Regression  0.012456 0.0157              0.6234
```

---

### `forecast` — Generate price forecasts

Forecasts future stock prices using the latest trained run of a specified model.

**Syntax:**
```bash
finsight forecast --ticker TICKER --model-id MODEL_ID --horizon DAYS [OPTIONS]
```

**Required Arguments:**
- `--ticker TICKER` — Stock ticker symbol (e.g., `AAPL`, `MSFT`)
- `--model-id MODEL_ID` — Model ID to use for forecasting (e.g., `ridge`, `naive_zero`)
- `--horizon DAYS` — Forecast horizon in trading days (must be positive integer)

**Optional Arguments:**
- `--artifacts-dir DIR` — Directory containing model artifacts (default: `artifacts/runs`)
- `--json` — Output forecast as JSON (default: human-readable text)

#### Text Output (default)

**Example:**
```bash
finsight forecast --ticker AAPL --model-id ridge --horizon 5
```

**Output:**
```
[ridge] ticker=AAPL horizon_days=5 rows=5
2026-04-15 pred_ret_1d=0.0125 pred_close=152.45
2026-04-16 pred_ret_1d=-0.0087 pred_close=151.13
2026-04-17 pred_ret_1d=0.0234 pred_close=154.68
2026-04-20 pred_ret_1d=0.0056 pred_close=155.54
2026-04-21 pred_ret_1d=-0.0045 pred_close=154.84
```

Fields:
- `DATE` — Forecast date (ISO format: `YYYY-MM-DD`)
- `pred_ret_1d` — Predicted 1-day return (decimal, e.g., 0.0125 = +1.25%)
- `pred_close` — Predicted closing price

#### JSON Output

**Example:**
```bash
finsight forecast --ticker AAPL --model-id ridge --horizon 2 --json
```

**Output:**
```json
{
  "generated_at": "2026-04-14T12:00:00Z",
  "horizon_days": 2,
  "model_id": "ridge",
  "predictions": [
    {
      "date": "2026-04-15",
      "pred_close": 152.45,
      "pred_ret_1d": 0.0125
    },
    {
      "date": "2026-04-16",
      "pred_close": 151.13,
      "pred_ret_1d": -0.0087
    }
  ],
  "ticker": "AAPL"
}
```

---

## Error Handling

All commands print errors to `stderr` on failure. Command-line usage and argument parsing errors exit with code `2`; validation errors, missing artifacts, and runtime failures exit with code `1`.

### Common Error Scenarios

#### Missing Required Arguments
```bash
$ finsight forecast --ticker AAPL
usage: finsight forecast [-h] --ticker TICKER --model-id MODEL_ID --horizon HORIZON ...
finsight forecast: error: the following arguments are required: --model-id, --horizon
```

#### Invalid Validation Inputs
```bash
$ finsight forecast --ticker "" --model-id ridge --horizon 5
Forecast validation error: ticker must be a non-empty string.
```

```bash
$ finsight forecast --ticker AAPL --model-id ridge --horizon 0
Forecast validation error: horizon_days must be a positive integer.
```

#### No Trained Runs Found
```bash
$ finsight forecast --ticker AAPL --model-id unknown_model --horizon 5
Forecast artifact error: No runs found for model_id 'unknown_model' under artifact root: artifacts/runs
```

**Solution:** Train the model first with `finsight train --cutoff <date>`.

#### Corrupt or Incompatible Artifacts
```bash
$ finsight forecast --ticker AAPL --model-id ridge --horizon 5
Forecast runtime error: Loaded model artifact does not implement predict(...).
```

**Solution:** Check that artifacts were saved correctly; retrain if necessary.

---

## Workflow Example

A typical workflow from training to forecasting:

```bash
# 1. Train models with a cutoff date
finsight train --cutoff 2025-06-01 --years 2 --model-types naive_zero ridge

# 2. Compare the trained models to see which performed best
finsight compare --model-ids naive_zero ridge --rank-by mae direction_accuracy

# 3. Generate forecasts using the best model
finsight forecast --ticker AAPL --model-id ridge --horizon 30

# 4. Export forecast as JSON for downstream processing
finsight forecast --ticker AAPL --model-id ridge --horizon 30 --json > forecast_aapl.json
```

---

## Exit Codes

- `0` — Success
- `1` — Validation error, missing artifact, or runtime failure
- `2` — Command-line usage or argument parsing error (for example, missing required arguments)

---

## Notes

- All dates must be in ISO format: `YYYY-MM-DD`.
- Forecast dates skip weekends (business days only).
- Predicted returns compound iteratively; each forecast feeds into the next day's feature engineering.
- Use `--json` for scripting and automation; text output is optimized for human readability.

