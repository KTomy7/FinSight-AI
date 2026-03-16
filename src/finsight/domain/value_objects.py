from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Ticker:
    value: str

    def __post_init__(self) -> None:
        normalized = (self.value or "").strip().upper()
        if not normalized:
            raise ValueError("Ticker cannot be empty.")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Period:
    """Yahoo Finance-style period (e.g. '1y', '6mo', '5y', 'max')."""
    value: str = "1y"

    def __post_init__(self) -> None:
        normalized = (self.value or "").strip()
        if not normalized:
            raise ValueError("Period cannot be empty.")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Interval:
    """Yahoo Finance-style interval (e.g. '1d', '1h', '1wk')."""
    value: str = "1d"

    def __post_init__(self) -> None:
        normalized = (self.value or "").strip()
        if not normalized:
            raise ValueError("Interval cannot be empty.")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value

