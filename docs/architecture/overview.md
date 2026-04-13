# Architecture Overview

## Goal

FinSight-AI is a stock-market forecasting tool. It fetches historical OHLCV data, engineers
features, trains baseline ML models, and surfaces results through a Streamlit web UI and a
CLI. The architecture keeps ML logic decoupled from the UI and makes it straightforward to
add new data sources, feature pipelines, or model implementations.

---

## Layer Map

```
src/finsight/
├── domain/          # Pure business rules — no I/O, no framework imports
├── application/     # Use cases, DTOs, and contracts that orchestrate the domain
├── infrastructure/  # Concrete adapters for I/O (market data, features, ML, storage)
├── adapters/        # Entry-point adapters (Streamlit web UI, future REST)
├── cli/             # Command-line entry point
├── bootstrap/       # Dependency-injection wiring
└── config/          # Typed settings loaded from config/config.yaml
```

---

## Layer Responsibilities

### `domain/`

Contains the core vocabulary of the application.

| Module | Contents |
|---|---|
| `entities.py` | `OHLCVSeries`, `StockSummary` — rich data objects |
| `value_objects.py` | `Ticker`, `DateRange`, `Interval`, `Period` — validated immutable values |
| `ports.py` | `Protocol` interfaces: `MarketDataPort`, `FeatureStorePort`, `ModelPort`, `ModelRegistryPort` |
| `metrics.py` | Pure metric functions (MAE, RMSE, direction accuracy) and canonical key constants |

**Rules:**
- No imports from `application`, `infrastructure`, or `adapters`.
- No I/O, no framework imports.
- Port protocols live here because they express what the domain *needs*, not what provides it.

---

### `application/`

Orchestrates domain objects and ports to fulfil business use cases.

| Module | Contents |
|---|---|
| `use_cases/train_model.py` | `TrainModel` — builds features, evaluates models, writes run manifests |
| `use_cases/fetch_market_data.py` | `FetchMarketData` — fetches and summarises OHLCV data |
| `use_cases/compare_models.py` | `CompareModels` — ranks model runs into a deterministic leaderboard |
| `dto.py` | All request/response data transfer objects (see [DTO conventions](../conventions/naming-and-contracts.md)) |
| `contracts/run_manifest.py` | `build_run_manifest`, `validate_run_manifest` — structured training-run records |

**Rules:**
- Use cases depend only on domain ports (injected via constructor).
- DTOs are frozen dataclasses. DTOs used for adapter/persistence serialization provide `to_dict` / `from_dict` helpers; others may expose domain entities directly.
- No Streamlit, no yfinance, no sklearn imports.

---

### `infrastructure/`

Provides concrete implementations of the domain ports.

| Sub-package | Port implemented |
|---|---|
| `market_data/yfinance_provider.py` | `MarketDataPort` |
| `features/feature_store.py` | `FeatureStorePort` |
| `ml/sklearn/baseline.py` | `ModelPort` |
| `ml/registry.py` | `ModelRegistryPort` |

**Rules:**
- Each implementation satisfies exactly one port protocol.
- Framework-specific code (pandas, scikit-learn, yfinance) is confined here.
- No Streamlit imports.

---

### `adapters/web_streamlit/`

Thin presentation layer. Converts use case results into Streamlit widgets.

| Module | Contents |
|---|---|
| `app.py` | Page routing, Streamlit app entry point |
| `views/home.py` | Home / dashboard view |
| `views/predict.py` | Forecast / prediction view |
| `views/compare.py` | Side-by-side model comparison view |
| `presenters.py` | Converts domain/DTO objects to display-ready dicts |
| `ticker_options.py` | Helper to build ticker dropdown options from config |

**Rules:**
- Views call use cases via the container; they do not construct infrastructure objects directly.
- Plotting and `st.*` calls belong only in this layer.
- Business logic (feature engineering, metric computation) must not live here.

---

### `cli/`

CLI entry point. Parses arguments and delegates to `TrainModel` and `CompareModels` via the container.

---

### `bootstrap/`

Wires all layers together.

`container.py` exposes `build_container()` (cached via `@lru_cache`) which returns
an `AppContainer` with fully-constructed use case instances. The container is the
only place that imports both `infrastructure` and `application` simultaneously.

---

### `config/`

`settings.py` defines typed dataclasses (`Settings`, `ModelDefaults`, `TickerCatalogSettings`,
etc.) and loads them from `config/config.yaml` using PyYAML (`yaml`) plus manual parsing.
Always access settings through `get_settings()` or the container; never read the YAML directly.

---

## Key Data Flows

### Training a model

```
CLI / Streamlit view
  └─ TrainModel.execute(TrainModelRequest)
       ├─ FetchMarketData → MarketDataPort.fetch_ohlcv  (yfinance)
       ├─ FeatureStorePort.build_feature_dataset        (pandas)
       ├─ FeatureStorePort.split_train_test
       ├─ ModelPort.evaluate                            (NumPy/pandas baseline; scikit-learn optional)
       ├─ build_run_manifest + validate_run_manifest
       ├─ ModelRegistryPort.save_manifest               (local filesystem)
       └─ returns TrainModelResult
```

### Comparing trained models

```
CLI / Streamlit view
  └─ CompareModels.execute(CompareModelsRequest)
       ├─ ModelRegistryPort.latest_run_id              (locates latest run for each model)
       ├─ ModelRegistryPort.load_run_artifacts          (loads metrics and manifest data)
       ├─ deterministic metric ranking + tie-breaks
       └─ returns CompareModelsResult                   (table-ready leaderboard rows)
```

### Fetching market data

```
Streamlit view
  └─ FetchMarketData.execute(FetchMarketDataRequest)
       └─ MarketDataPort.fetch_ohlcv + get_summary      (yfinance)
            └─ returns FetchMarketDataResult
```

---

## Extension Points

### Adding a new model

1. Implement `ModelPort` in `infrastructure/ml/<framework>/<name>.py`.
2. Add an entry to `model_defaults.catalog` in `config/config.yaml`.
3. Register the implementation in `bootstrap/container.py`.
4. No changes to the domain or application layers are required.

### Adding a new data source

1. Implement `MarketDataPort` in `infrastructure/market_data/<name>.py`.
2. Swap the binding in `bootstrap/container.py`.

### Adding a new use case

1. Define request/response DTOs in `application/dto.py`.
2. Implement the use case class in `application/use_cases/<name>.py`, depending only on
   domain ports.
3. Add the use case to `AppContainer` in `bootstrap/container.py`.

---

## Where to Capture Architecture Decisions

Place new architecture decision records (ADRs) or migration notes in `docs/architecture/`.
Name files descriptively (e.g., `lstm-model-integration.md`, `dto-migration.md`).
Update them in the same PR that implements the described change.
