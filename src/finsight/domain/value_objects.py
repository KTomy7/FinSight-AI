from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Union, cast

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


@dataclass(frozen=True, slots=True)
class DateRange:
    """Inclusive calendar date range [start, end].

    The constructor accepts any of:

    * ``datetime.date``     — stored as-is.
    * ``datetime.datetime`` — the time component is stripped via ``.date()``.
    * ``str``               — parsed from an ISO-8601 date string (``"YYYY-MM-DD"``).

    After construction both ``start`` and ``end`` are always ``datetime.date``
    instances (never ``datetime.datetime`` or ``str``).
    """

    start: Union[datetime.date, datetime.datetime, str]
    end: Union[datetime.date, datetime.datetime, str]

    def __post_init__(self) -> None:
        start: datetime.date | datetime.datetime | str = self.start
        end: datetime.date | datetime.datetime | str = self.end

        # Parse ISO strings
        if isinstance(start, str):
            try:
                start = datetime.date.fromisoformat(start)
            except ValueError as exc:
                raise ValueError(
                    f"DateRange.start is not a valid ISO date: {self.start!r}"
                ) from exc
        if isinstance(end, str):
            try:
                end = datetime.date.fromisoformat(end)
            except ValueError as exc:
                raise ValueError(
                    f"DateRange.end is not a valid ISO date: {self.end!r}"
                ) from exc

        # Normalise datetime.datetime → datetime.date (strip the time component)
        if isinstance(start, datetime.datetime):
            start = start.date()
        if isinstance(end, datetime.datetime):
            end = end.date()

        if not isinstance(start, datetime.date):
            raise TypeError(
                f"DateRange.start must be a date, datetime, or ISO string, "
                f"got {type(start).__name__}"
            )
        if not isinstance(end, datetime.date):
            raise TypeError(
                f"DateRange.end must be a date, datetime, or ISO string, "
                f"got {type(end).__name__}"
            )
        if start > end:
            raise ValueError(
                f"DateRange start ({start}) must be <= end ({end})"
            )

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def end_exclusive(self) -> datetime.date:
        """Return end + 1 day for APIs that treat the end boundary as exclusive."""
        # cast: __post_init__ guarantees self.end is datetime.date at runtime.
        return cast(datetime.date, self.end) + datetime.timedelta(days=1)

    def __str__(self) -> str:
        # cast: __post_init__ guarantees both fields are datetime.date at runtime.
        return (
            f"{cast(datetime.date, self.start).isoformat()} to "
            f"{cast(datetime.date, self.end).isoformat()}"
        )


