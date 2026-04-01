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

### Preferred imports (canonical)

```python
from finsight.application.dto import TrainModelRequest, TrainModelResult
```

### Legacy imports (removed — migration complete)

The temporary re-export from `finsight.application.use_cases.train_model` has been
removed. Any code that previously imported DTOs from the use-case module must now
import from `finsight.application.dto`.

## Rollout status

1. ✅ DTOs centralised in `finsight.application.dto`.
2. ✅ All adapters and tests updated to import from `finsight.application.dto`.
3. ✅ Legacy re-export removed from `finsight.application.use_cases.train_model`.

