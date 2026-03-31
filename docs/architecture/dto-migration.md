# DTO Migration Guide

## Why this change

ML use-case request and response contracts are now centralized in `src/finsight/application/dto.py`.
This gives one place to define typed, serializable contracts and makes future use-cases
(e.g., forecasting and backtesting) consistent.

## New DTO module

Use DTOs from:

- `finsight.application.dto`

The module currently includes:

- `DatasetSpec`
- `FeatureSpec`
- `FetchMarketDataRequest`
- `FetchMarketDataResult`
- `TrainModelRequest`
- `TrainModelResult`
- `ForecastResult`
- `BacktestResult`

All DTOs provide `to_dict()` and `from_dict()` methods for adapter/persistence serialization.

## Migration path for existing TrainModelRequest/Response imports

### Preferred imports (new)

```python
from finsight.application.dto import TrainModelRequest, TrainModelResult
```

### Legacy imports (removed)


This path was temporary during migration and is no longer exported by
`finsight.application.use_cases.train_model`.

Use `finsight.application.dto` for all request/response DTO imports.

## Suggested rollout

1. Update adapters/tests to import DTOs from `finsight.application.dto`.
2. Keep legacy import compatibility for one release cycle.
3. Remove legacy import usage once the codebase no longer depends on it. (completed)

