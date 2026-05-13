# DTO Migration Guide

## Why this change

ML use-case request and response contracts are now centralized in `src/finsight/application/dto.py`.
This gives one place to define typed, serializable contracts and keeps forecasting,
comparison, and backtesting flows consistent.

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
- `CompareModelsResult`
- `ForecastResult`
- `BacktestResult`

DTOs that are used for adapter/persistence serialization provide `to_dict()` and `from_dict()` methods.

The current Streamlit `Train & Backtest` view builds chart-ready summaries from `TrainModelResult` plus persisted run artifacts via `TrainPresenter`; it does not consume `BacktestResult` directly. `BacktestResult` remains available as the shared DTO for adapters that need a serializable backtest summary.

## Migration path for existing TrainModelRequest/TrainModelResult imports

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
4. ✅ Backtest/prediction presentation code now consumes the shared DTOs instead of ad hoc response shapes.

