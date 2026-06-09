# FinSight-AI Web Streamlit Architecture

This document explains the code layers that support the Streamlit application and maps every active Streamlit page to the application use cases it invokes.

## Purpose

The Streamlit app is a thin presentation layer. It gathers user input, delegates work to use cases through the bootstrap container, and renders results with presenters. All business logic stays outside the UI layer.

## End-to-end flow

```text
Streamlit page
  -> presenter formatting helpers
  -> application use case
  -> domain ports and entities
  -> infrastructure implementations
  -> persisted artifacts / fetched market data / model output
  -> back to presenter
  -> Streamlit widgets
```

## Layer-by-layer reference

### 1) `src/finsight/domain/`

The domain layer holds the core business vocabulary and the contracts that describe what the application needs.

#### `entities.py`
- `OHLCVSeries` — historical market data for one ticker and date range.
- `StockSummary` — lightweight summary information for a ticker.
- `ModelEvaluationResult` — evaluation output returned by model implementations.

#### `value_objects.py`
- `Ticker` — validated, normalized ticker symbol.
- `Period` — Yahoo Finance-style period value.
- `Interval` — Yahoo Finance-style interval value.
- `DateRange` — inclusive date range used for market-data requests.

#### `ports.py`
- `MarketDataPort` — fetch OHLCV series and summaries.
- `FeatureStorePort` — build feature datasets and split them for training or inference.
- `ModelPort` — evaluate a model type and expose supported model types.
- `ModelRegistryPort` — persist and reload run artifacts.
- `RunRegistryPort` — store and load best-run registry state.

#### `metrics.py`
- Canonical metric names: `mae`, `rmse`, `r2`, `direction_accuracy`.
- Metric functions: MAE, RMSE, R², direction accuracy, and the combined `forecast_metrics(...)` helper.

**Role in the app:**
The domain layer never talks to Streamlit, files, or external APIs directly. It defines the language that the application and infrastructure use.

---

### 2) `src/finsight/application/`

The application layer contains use cases, DTOs, and contracts. It orchestrates the domain and infrastructure.

#### `dto.py`
The shared request/result objects used by both CLI and Streamlit flows.

Relevant DTOs for the Streamlit pages:
- `FetchMarketDataRequest` / `FetchMarketDataResult`
- `TrainModelRequest` / `TrainModelResult`
- `BacktestRequest` / `BacktestReport` / `BacktestResult`
- `CompareModelsRequest` / `CompareModelsResult`
- `ForecastRequest` / `ForecastResult`
- Supporting DTOs: `DatasetSpec`, `FeatureSpec`, `ModelComparisonRow`, `BacktestFoldSummary`, `RunSummary`, `RegistrySnapshot`, `ModelRunArtifacts`

#### `use_cases/`
- `fetch_market_data.py` — fetches market data for a ticker and optional summary.
- `train_model.py` — runs single-split training, evaluates models, saves artifacts, and records runs.
- `backtest.py` — runs walk-forward evaluation across one or more models.
- `compare_models.py` — loads saved runs and builds a deterministic leaderboard.
- `forecast.py` — loads the latest trained run for a model and produces forward price forecasts.

#### `contracts/run_manifest.py`
- `build_run_manifest(...)` and `validate_run_manifest(...)` define the schema for `manifest.json` written during training.

**Role in the app:**
The Streamlit pages never implement model logic themselves. They create DTOs, call the relevant use case, and display the result.

---

### 3) `src/finsight/infrastructure/`

The infrastructure layer provides concrete implementations of the domain ports.

#### `market_data/`
- `yfinance_provider.py` — Yahoo Finance-backed `MarketDataPort` implementation.

#### `features/`
- `feature_store.py` — builds training and inference datasets, splits data, and exposes feature metadata.
- `feature_pipeline.py` — feature-engineering pipeline helpers.
- `policies.py` — split policies used by the feature store and evaluation flows.

#### `ml/`
- `sklearn/` — model adapters for the supported model types.
  - `baseline.py`
  - `linear.py`
  - `tree.py`
  - `xgboost.py`
  - `router.py` — selects the concrete model adapter by model ID.
- `registry.py` — local filesystem persistence for model artifacts, metrics, manifests, and predictions.
- `run_registry.py` — stores the best run per model in the run registry.
- `model_ranker.py` — deterministic ranking logic used by model comparison and registry decisions.

**Role in the app:**
Infrastructure is where pandas, scikit-learn, yfinance, and filesystem I/O live. It is wired into the app through the bootstrap container.

---

### 4) `src/finsight/adapters/web_streamlit/`

This is the Streamlit adapter layer. It turns use-case results into interactive UI.

#### `app.py`
- Streamlit entry point.
- Configures the page.
- Renders the sidebar.
- Dispatches to the selected page handler.

#### `views/`
Active pages:
- `home.py`
- `predict.py`
- `backtest.py`
- `train_model.py`
- `compare.py`

Routing and wiring:
- `layout.py` renders the sidebar menu.
- `__init__.py` exposes `PAGE_HANDLERS` and page aliases.

#### `presenters.py`
Pure formatting helpers that convert use-case DTOs into pandas DataFrames for display.
- `ForecastPresenter`
- `ComparisonPresenter`
- `TrainPresenter`
- `BacktestPresenter`

#### `ticker_options.py`
Helper for building labeled ticker dropdown options from config.

**Role in the app:**
This layer contains Streamlit widgets, tables, charts, and error messages. It should not contain business rules.

---

### 5) `src/finsight/bootstrap/`

The bootstrap layer is the composition root.

#### `container.py`
- `AppContainer` groups the fully constructed use cases.
- `build_container()` instantiates infrastructure implementations and injects them into the use cases.

Constructed use cases:
- `fetch_market_data`
- `train_model`
- `backtest`
- `compare_models`
- `forecast`

**Role in the app:**
The Streamlit adapter gets all dependencies from the container rather than constructing them directly.

---

### 6) `src/finsight/config/`

Configuration is loaded from `config/config.yaml` via typed settings in `settings.py`.

Typical configuration surfaces used by the Streamlit app:
- model catalog and labels
- default model IDs
- ticker catalog and labels
- default horizon and interval values
- cache TTL settings

**Role in the app:**
Config drives dropdown values, defaults, and supported model selection.

---

### 7) `src/finsight/cli/`

The CLI is a secondary adapter that reuses the same application layer as Streamlit.

**Role in the app:**
It is not part of the Streamlit UI, but it confirms that the use cases are UI-agnostic and can be reused outside the web app.

---

## Streamlit pages and the use cases they use

### `Home`
**File:** `src/finsight/adapters/web_streamlit/views/home.py`

- Purpose: landing page and product overview.
- Use case calls: none.
- Notes: purely informational, with a banner image and navigation hints.

---

### `Predict`
**File:** `src/finsight/adapters/web_streamlit/views/predict.py`

- Purpose: generate a forward price forecast for one ticker and one trained model.
- Presenter: `ForecastPresenter`
- Primary use case: `Forecast`
- Direct DTOs used:
  - `ForecastRequest`
  - `ForecastResult`

#### Internal call chain
1. The page collects ticker, model, and forecast horizon.
2. It calls `build_container().forecast` through a cached Streamlit resource.
3. `Forecast.execute(...)` loads the latest saved run for the selected model.
4. `FetchMarketData.execute(...)` retrieves current market history.
5. The feature store builds inference features.
6. The model predicts forward one day at a time.
7. The presenter formats the result for the table and line chart.

#### Dependencies used inside `Forecast`
- `FetchMarketData`
- `FeatureStorePort`
- `ModelRegistryPort`

---

### `Backtest`
**File:** `src/finsight/adapters/web_streamlit/views/backtest.py`

- Purpose: run walk-forward evaluation across selected models.
- Presenter: `BacktestPresenter`
- Primary use case: `Backtest`
- Direct DTOs used:
  - `BacktestRequest`
  - `BacktestReport`
  - `BacktestResult`

#### Internal call chain
1. The page collects model IDs and walk-forward settings.
2. It calls `build_container().backtest` through a cached Streamlit resource.
3. `Backtest.execute(...)` fetches historical data for the configured ticker basket.
4. The feature store builds a training dataset.
5. `WalkForwardSplitPolicy` produces the folds.
6. The model is evaluated on each fold.
7. The presenter converts the report into a summary table and fold-level details.

#### Dependencies used inside `Backtest`
- `FetchMarketData`
- `FeatureStorePort`
- `ModelPort`
- `WalkForwardSplitPolicy`

---

### `Train Model`
**File:** `src/finsight/adapters/web_streamlit/views/train_model.py`

- Purpose: run single-split training, persist artifacts, and display per-ticker evaluation visuals.
- Presenter: `TrainPresenter`
- Primary use case: `TrainModel`
- Direct DTOs used:
  - `TrainModelRequest`
  - `TrainModelResult`
  - `FetchMarketDataRequest` (for chart reconstruction)

#### Internal call chain
1. The page gathers selected models and ticker display filters.
2. It calls `container.train_model.execute(...)`.
3. `TrainModel.execute(...)` fetches market data for the training basket.
4. The feature store builds the training dataset and splits it at the cutoff date.
5. The model evaluates each selected model type.
6. The model registry persists the run directory, model, metrics, manifest, and predictions.
7. The run registry records the completed run.
8. The view loads `predictions.csv` and `manifest.json` for each run.
9. The page fetches market history again when needed to reconstruct ticker-level backtest charts.
10. The presenter formats metrics and per-ticker display data.

#### Dependencies used inside `TrainModel`
- `FetchMarketData`
- `FeatureStorePort`
- `ModelPort`
- `ModelRegistryPort`
- `RunRegistryPort`
- `build_run_manifest(...)`

#### Important note
The current `Train Model` page is a single-split evaluation page. It is not the same as the dedicated walk-forward `Backtest` page.

---

### `Compare Models`
**File:** `src/finsight/adapters/web_streamlit/views/compare.py`

- Purpose: build a deterministic leaderboard from saved runs.
- Presenter: `ComparisonPresenter`
- Primary use case: `CompareModels`
- Direct DTOs used:
  - `CompareModelsRequest`
  - `CompareModelsResult`

#### Internal call chain
1. The page collects model IDs and ranking metrics.
2. It calls `build_container().compare_models` through a cached Streamlit resource.
3. `CompareModels.execute(...)` loads run artifacts from disk.
4. When enabled, it also loads the run registry to prefer the best run per model.
5. `ModelRanker` computes the deterministic ranking key.
6. The presenter formats the leaderboard for the table.

#### Dependencies used inside `CompareModels`
- `ModelRegistryPort`
- `RunRegistryPort`
- `ModelRanker`

---

## Page routing summary

The app sidebar in `views/layout.py` exposes these active pages:

- `Home`
- `Predict`
- `Backtest`
- `Train Model`
- `Compare Models`

The routing map in `views/__init__.py` points those labels to the corresponding render functions.

There is also a backward-compatibility alias:
- `render_train_backtest` currently points to the same implementation as `train_model.render`.

---

## Use-case reference table

| Use case | Used by page(s) | Main responsibility | Main dependencies |
|---|---|---|---|
| `FetchMarketData` | indirectly by `Predict`, `Backtest`, `Train Model`; directly inside `Train Model` view for chart reconstruction | Fetch OHLCV history and optional stock summary | `MarketDataPort` |
| `TrainModel` | `Train Model` | Train selected models on a fixed ticker basket, save artifacts, register the run | `FetchMarketData`, `FeatureStorePort`, `ModelPort`, `ModelRegistryPort`, `RunRegistryPort` |
| `Backtest` | `Backtest` | Evaluate models across walk-forward folds | `FetchMarketData`, `FeatureStorePort`, `ModelPort`, `WalkForwardSplitPolicy` |
| `CompareModels` | `Compare Models` | Load saved runs and produce a ranked leaderboard | `ModelRegistryPort`, `RunRegistryPort`, `ModelRanker` |
| `Forecast` | `Predict` | Load latest run artifacts and generate future price forecasts | `FetchMarketData`, `FeatureStorePort`, `ModelRegistryPort` |

---

## Why the separation matters

- The UI layer stays simple and testable.
- Use cases can be reused from both the Streamlit app and the CLI.
- Infrastructure can be swapped without rewriting page logic.
- Presenters isolate display formatting from business logic.

---

## If you add a new page

1. Add a render function under `src/finsight/adapters/web_streamlit/views/`.
2. Add any formatting helpers to `presenters.py`.
3. Reuse or add a DTO in `src/finsight/application/dto.py`.
4. Add or reuse a use case in `src/finsight/application/use_cases/`.
5. Wire the use case into `bootstrap/container.py`.
6. Register the page label in `views/layout.py` and `views/__init__.py`.

---

## If you add a new use case

1. Define request/result DTOs in `src/finsight/application/dto.py`.
2. Implement the orchestration logic in `src/finsight/application/use_cases/`.
3. Depend only on domain ports and DTOs.
4. Add the dependency to `bootstrap/container.py`.
5. Create or update presenters and Streamlit pages if the new use case should appear in the UI.

