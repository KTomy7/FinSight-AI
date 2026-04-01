# Naming and Contracts

Conventions for naming, typing, and extending contracts across the codebase.

---

## DTOs (Data Transfer Objects)

### Location

All use-case request and response DTOs live in a single module:

```
src/finsight/application/dto.py
```

Import DTOs from there — never from individual use-case modules:

```python
# correct
from finsight.application.dto import TrainModelRequest, TrainModelResult

# wrong — this path is not exported
from finsight.application.use_cases.train_model import TrainModelRequest
```

### Current DTO catalogue

| DTO | Direction | Use case |
|---|---|---|
| `FetchMarketDataRequest` | request | `FetchMarketData` |
| `FetchMarketDataResult` | result | `FetchMarketData` |
| `DatasetSpec` | nested result | `TrainModel` |
| `FeatureSpec` | nested result | `TrainModel` |
| `TrainModelRequest` | request | `TrainModel` |
| `TrainModelResult` | result | `TrainModel` |
| `ForecastResult` | result | future forecast use case |
| `BacktestResult` | result | future backtest use case |

### DTO rules

- DTOs are **frozen dataclasses** (`@dataclass(frozen=True, slots=True)`).
- Every DTO exposes `to_dict() -> dict[str, Any]` and a `from_dict(cls, payload)` classmethod.
- Field types use built-in Python types or domain entities — no infrastructure types.
- Prefer `str | None` for optional dates (ISO `"YYYY-MM-DD"` format).

### Adding a new DTO

1. Add the dataclass to `src/finsight/application/dto.py`.
2. Follow the frozen-dataclass pattern with `to_dict` / `from_dict`.
3. Update this file's catalogue table.
4. Import the DTO in any use case or adapter that needs it.

---

## Metric Keys

Forecasting metric names are defined as module-level constants in `src/finsight/domain/metrics.py`:

```python
METRIC_MAE               = "mae"
METRIC_RMSE              = "rmse"
METRIC_DIRECTION_ACCURACY = "direction_accuracy"

SUPPORTED_METRIC_NAMES = (METRIC_MAE, METRIC_RMSE, METRIC_DIRECTION_ACCURACY)
```

Always reference these constants instead of raw strings when storing or displaying metrics.
`SUPPORTED_METRIC_NAMES` defines the canonical display order.

---

## Port Protocols

Domain ports (`src/finsight/domain/ports.py`) use `typing.Protocol` with
`@runtime_checkable`. The four ports are:

| Protocol | Responsibility |
|---|---|
| `MarketDataPort` | Fetch OHLCV series and stock summary |
| `FeatureStorePort` | Build feature datasets, split train/test, inspect metadata |
| `ModelPort` | Evaluate a model type; expose supported model types |
| `ModelRegistryPort` | Persist run artifacts (metrics, manifest, predictions) |

### Rules for ports

- Ports are defined in `domain/`; implementations live in `infrastructure/`.
- A port method raises `NotImplementedError` by default (Protocol body convention).
- Infrastructure implementations must satisfy the protocol structurally —
  no explicit inheritance required.
- Constructor-inject ports into use cases; never instantiate infrastructure classes
  inside use-case or domain code.

### Adding a new port

1. Define the `Protocol` class in `src/finsight/domain/ports.py`.
2. Implement it in the appropriate `infrastructure/` sub-package.
3. Inject it through `bootstrap/container.py`.

---

## Run Manifests

Every completed training run writes a `manifest.json` to its run directory.
The contract is defined in `src/finsight/application/contracts/run_manifest.py`.

Required manifest keys:

```
run_id, model_id, feature_columns, target, split_policy,
dates, params, artifact_paths, created_at
```

Use `build_run_manifest(...)` to construct and `validate_run_manifest(manifest)`
to verify a manifest before persisting it.

---

## Configuration and Settings

Settings are typed dataclasses loaded from `config/config.yaml`:

- `Settings.model_defaults.catalog` — tuple of `ModelCatalogEntry` (id, label, supports_training, supports_prediction)
- `Settings.model_defaults.default_model_id` — default model to use when none is specified
- `Settings.ticker_catalog.entries` — tuple of `TickerCatalogEntry` (symbol, company_name)
- `Settings.ticker_catalog.symbols()` — convenience method returning just the symbol strings

Always retrieve settings through `get_settings()` from `finsight.config.settings`.

---

## Naming Conventions

| Concept | Convention | Example |
|---|---|---|
| Use-case class | `PascalCase` noun phrase | `TrainModel`, `FetchMarketData` |
| Use-case method | `execute` | `train_model.execute(request)` |
| DTO request | `<Action>Request` | `TrainModelRequest` |
| DTO result | `<Action>Result` | `TrainModelResult` |
| Port protocol | `<Role>Port` | `ModelPort`, `MarketDataPort` |
| Infrastructure impl | descriptive name | `YFinanceMarketDataProvider`, `PandasFeatureStore` |
| Metric key constant | `METRIC_<NAME>` | `METRIC_MAE`, `METRIC_RMSE` |
| Run artifact dir | `artifacts/runs/<run_id>/` | set via `artifacts_dir` in request |
