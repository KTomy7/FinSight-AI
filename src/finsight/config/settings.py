from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True, slots=True)
class StockDataSettings:
    default_symbol: str = "AAPL"
    default_lookback_days: int = 365
    default_interval: str = "1d"


@dataclass(frozen=True, slots=True)
class PreprocessingSettings:
    lag_window: int = 5
    test_size: float = 0.2
    shuffle: bool = False


@dataclass(frozen=True, slots=True)
class CacheSettings:
    resource_ttl_seconds: int = 3600
    data_ttl_seconds: int = 900


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    id: str
    label: str
    supports_training: bool = True
    supports_prediction: bool = True


DEFAULT_MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(id="naive_zero", label="Naive (Zero)", supports_training=True, supports_prediction=True),
    ModelCatalogEntry(id="naive_mean", label="Naive (Mean)", supports_training=True, supports_prediction=True),
    ModelCatalogEntry(id="ridge", label="Ridge Regression", supports_training=False, supports_prediction=False),
)


@dataclass(frozen=True, slots=True)
class ModelDefaults:
    catalog: tuple[ModelCatalogEntry, ...] = DEFAULT_MODEL_CATALOG
    default_model_id: str = "naive_zero"
    horizon_min: int = 7
    horizon_max: int = 90
    default_horizon: int = 30

    def model_ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.catalog)

    def labels(self) -> tuple[str, ...]:
        return tuple(entry.label for entry in self.catalog)

    def id_to_label(self) -> dict[str, str]:
        return {entry.id: entry.label for entry in self.catalog}

    def label_to_id(self) -> dict[str, str]:
        return {entry.label: entry.id for entry in self.catalog}

    def training_model_ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.catalog if entry.supports_training)

    def prediction_model_ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.catalog if entry.supports_prediction)


@dataclass(frozen=True, slots=True)
class TickerCatalogEntry:
    symbol: str
    company_name: str


DEFAULT_TICKER_CATALOG: tuple[TickerCatalogEntry, ...] = (
    TickerCatalogEntry(symbol="AAPL", company_name="Apple Inc."),
    TickerCatalogEntry(symbol="JPM", company_name="JPMorgan Chase & Co."),
    TickerCatalogEntry(symbol="XOM", company_name="Exxon Mobil Corporation"),
    TickerCatalogEntry(symbol="KO", company_name="The Coca-Cola Company"),
    TickerCatalogEntry(symbol="TSLA", company_name="Tesla, Inc."),
)


@dataclass(frozen=True, slots=True)
class TickerCatalogSettings:
    entries: tuple[TickerCatalogEntry, ...] = DEFAULT_TICKER_CATALOG

    def symbols(self) -> tuple[str, ...]:
        return tuple(entry.symbol for entry in self.entries)


@dataclass(frozen=True, slots=True)
class Settings:
    stock_data: StockDataSettings = StockDataSettings()
    preprocessing: PreprocessingSettings = PreprocessingSettings()
    cache: CacheSettings = CacheSettings()
    model_defaults: ModelDefaults = ModelDefaults()
    ticker_catalog: TickerCatalogSettings = TickerCatalogSettings()


def _default_config_path() -> Path:
    # .../src/finsight/config/settings.py -> parents[3] == repository root.
    return Path(__file__).resolve().parents[3] / "config" / "config.yaml"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_str(value: Any, default: str) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized else default


def _as_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default

    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default

    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _normalize_symbol(value: Any) -> str:
    return _as_str(value, "").upper()


def _parse_model_catalog(value: Any, default: tuple[ModelCatalogEntry, ...]) -> tuple[ModelCatalogEntry, ...]:
    if not isinstance(value, (list, tuple)):
        return default

    parsed_entries: list[ModelCatalogEntry] = []
    for raw_entry in value:
        entry_raw = _as_mapping(raw_entry)
        model_id = _as_str(entry_raw.get("id"), "")
        label = _as_str(entry_raw.get("label"), "")
        if not model_id:
            continue
        parsed_entries.append(
            ModelCatalogEntry(
                id=model_id,
                label=label or model_id,
                supports_training=_as_bool(entry_raw.get("supports_training"), True),
                supports_prediction=_as_bool(entry_raw.get("supports_prediction"), True),
            )
        )

    return tuple(parsed_entries)


def _parse_ticker_catalog(value: Any, default: tuple[TickerCatalogEntry, ...]) -> tuple[TickerCatalogEntry, ...]:
    if not isinstance(value, (list, tuple)):
        return default

    parsed_entries: list[TickerCatalogEntry] = []
    for index, raw_entry in enumerate(value):
        entry_raw = _as_mapping(raw_entry)
        symbol = _normalize_symbol(entry_raw.get("symbol"))
        company_name = _as_str(entry_raw.get("company_name"), "")
        if not symbol:
            raise ValueError(f"ticker_catalog[{index}].symbol must be a non-empty string.")
        if not company_name:
            raise ValueError(f"ticker_catalog[{index}].company_name must be a non-empty string.")
        parsed_entries.append(TickerCatalogEntry(symbol=symbol, company_name=company_name))

    return tuple(parsed_entries)


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> Settings:
    path = config_path or _default_config_path()

    raw: Mapping[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
            raw = _as_mapping(loaded)
    except Exception:
        raw = {}

    stock_raw = _as_mapping(raw.get("stock_data"))
    preprocess_raw = _as_mapping(raw.get("preprocessing"))
    cache_raw = _as_mapping(raw.get("cache"))
    model_raw = _as_mapping(raw.get("model_defaults"))
    ticker_catalog_raw = raw.get("ticker_catalog")

    stock_settings = StockDataSettings(
        default_symbol=_as_str(stock_raw.get("default_symbol"), "AAPL"),
        default_lookback_days=_as_int(stock_raw.get("default_lookback_days"), 365, minimum=1),
        default_interval=_as_str(stock_raw.get("default_interval"), "1d"),
    )

    preprocessing_settings = PreprocessingSettings(
        lag_window=_as_int(preprocess_raw.get("lag_window"), 5, minimum=1),
        test_size=_as_float(preprocess_raw.get("test_size"), 0.2, minimum=0.01, maximum=0.99),
        shuffle=_as_bool(preprocess_raw.get("shuffle"), False),
    )

    cache_settings = CacheSettings(
        resource_ttl_seconds=_as_int(cache_raw.get("resource_ttl_seconds"), 3600, minimum=1),
        data_ttl_seconds=_as_int(cache_raw.get("data_ttl_seconds"), 900, minimum=1),
    )

    if "catalog" in model_raw:
        # When a catalog is explicitly provided (even if empty/invalid), do not
        # fall back to DEFAULT_MODEL_CATALOG so that validation can detect it.
        catalog = _parse_model_catalog(model_raw.get("catalog"), ())
    else:
        catalog = DEFAULT_MODEL_CATALOG
    if not catalog:
        raise ValueError("model_defaults.catalog must contain at least one model entry.")

    model_ids = tuple(entry.id for entry in catalog)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_defaults.catalog contains duplicate model ids.")

    labels = tuple(entry.label for entry in catalog)
    if len(set(labels)) != len(labels):
        raise ValueError("model_defaults.catalog contains duplicate model labels.")

    default_model_id = _as_str(model_raw.get("default_model_id"), model_ids[0])
    if default_model_id not in model_ids:
        raise ValueError(
            f"Unknown model_defaults.default_model_id '{default_model_id}'. Supported model ids: {model_ids}."
        )

    training_model_ids = tuple(entry.id for entry in catalog if entry.supports_training)
    if not training_model_ids:
        raise ValueError(
            "model_defaults.catalog must enable training for at least one model "
            "(set supports_training=true)."
        )

    horizon_min = _as_int(model_raw.get("horizon_min"), 7, minimum=1)
    horizon_max = _as_int(model_raw.get("horizon_max"), 90, minimum=horizon_min)
    default_horizon = _as_int(model_raw.get("default_horizon"), 30, minimum=horizon_min, maximum=horizon_max)

    model_settings = ModelDefaults(
        catalog=catalog,
        default_model_id=default_model_id,
        horizon_min=horizon_min,
        horizon_max=horizon_max,
        default_horizon=default_horizon,
    )

    if "ticker_catalog" in raw:
        # Reject invalid explicit catalogs instead of silently defaulting.
        ticker_catalog_entries = _parse_ticker_catalog(ticker_catalog_raw, ())
    else:
        ticker_catalog_entries = DEFAULT_TICKER_CATALOG

    if not ticker_catalog_entries:
        raise ValueError("ticker_catalog must contain at least one ticker entry.")

    ticker_symbols = tuple(entry.symbol for entry in ticker_catalog_entries)
    if len(set(ticker_symbols)) != len(ticker_symbols):
        raise ValueError("ticker_catalog contains duplicate symbols.")

    ticker_catalog_settings = TickerCatalogSettings(entries=ticker_catalog_entries)

    return Settings(
        stock_data=stock_settings,
        preprocessing=preprocessing_settings,
        cache=cache_settings,
        model_defaults=model_settings,
        ticker_catalog=ticker_catalog_settings,
    )


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()

