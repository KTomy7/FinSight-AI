from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True, slots=True)
class StockDataSettings:
    default_symbol: str = "AAPL"
    default_period: str = "1y"
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
class ModelDefaults:
    options: tuple[str, ...] = (
        "Linear Regression",
        "Random Forest",
        "LSTM (Neural Network)",
    )
    default_model: str = "Linear Regression"
    horizon_min: int = 7
    horizon_max: int = 90
    default_horizon: int = 30


@dataclass(frozen=True, slots=True)
class Settings:
    stock_data: StockDataSettings = StockDataSettings()
    preprocessing: PreprocessingSettings = PreprocessingSettings()
    cache: CacheSettings = CacheSettings()
    model_defaults: ModelDefaults = ModelDefaults()


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


def _as_tuple_of_str(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        normalized = tuple(_as_str(item, "") for item in value)
        filtered = tuple(item for item in normalized if item)
        if filtered:
            return filtered
    return default


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

    stock_settings = StockDataSettings(
        default_symbol=_as_str(stock_raw.get("default_symbol"), "AAPL"),
        default_period=_as_str(stock_raw.get("default_period"), "1y"),
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

    options = _as_tuple_of_str(
        model_raw.get("options"),
        (
            "Linear Regression",
            "Random Forest",
            "LSTM (Neural Network)",
        ),
    )
    default_model = _as_str(model_raw.get("default_model"), options[0])
    if default_model not in options:
        default_model = options[0]

    horizon_min = _as_int(model_raw.get("horizon_min"), 7, minimum=1)
    horizon_max = _as_int(model_raw.get("horizon_max"), 90, minimum=horizon_min)
    default_horizon = _as_int(model_raw.get("default_horizon"), 30, minimum=horizon_min, maximum=horizon_max)

    model_settings = ModelDefaults(
        options=options,
        default_model=default_model,
        horizon_min=horizon_min,
        horizon_max=horizon_max,
        default_horizon=default_horizon,
    )

    return Settings(
        stock_data=stock_settings,
        preprocessing=preprocessing_settings,
        cache=cache_settings,
        model_defaults=model_settings,
    )


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()

