from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True, slots=True)
class TimeSplitPolicy:
    cutoff_date: date = field(init=False)
    date_col: str = "date"
    inclusive_test: bool = True
    _cutoff_ts: pd.Timestamp = field(init=False, repr=False)

    def __init__(
        self,
        cutoff_date: date | datetime | str,
        date_col: str = "date",
        inclusive_test: bool = True,
    ) -> None:
        cutoff = self._normalize_cutoff(cutoff_date)
        object.__setattr__(self, "cutoff_date", cutoff)
        object.__setattr__(self, "date_col", date_col)
        object.__setattr__(self, "inclusive_test", inclusive_test)
        object.__setattr__(self, "_cutoff_ts", pd.Timestamp(cutoff))

    def split_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.date_col not in df.columns:
            raise ValueError(
                f"Missing required date column '{self.date_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        parsed_dates = pd.to_datetime(df[self.date_col], errors="coerce")
        invalid_mask = parsed_dates.isna()
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())
            raise ValueError(
                f"Column '{self.date_col}' contains {invalid_count} invalid date value(s)."
            )

        frame = df.copy()
        frame[self.date_col] = parsed_dates

        if self.inclusive_test:
            train_mask = parsed_dates < self._cutoff_ts
            test_mask = parsed_dates >= self._cutoff_ts
        else:
            train_mask = parsed_dates <= self._cutoff_ts
            test_mask = parsed_dates > self._cutoff_ts

        train_df = frame.loc[train_mask].copy()
        test_df = frame.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            min_date = parsed_dates.min().date().isoformat()
            max_date = parsed_dates.max().date().isoformat()
            raise ValueError(
                "Time split produced an empty partition. "
                f"cutoff_date={self.cutoff_date.isoformat()}, "
                f"dataset_min_date={min_date}, dataset_max_date={max_date}."
            )

        sort_cols = ["ticker", self.date_col] if {"ticker", self.date_col}.issubset(frame.columns) else [self.date_col]
        train_df = train_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        test_df = test_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        return train_df, test_df

    @staticmethod
    def _normalize_cutoff(cutoff_date: date | datetime | str) -> date:
        if isinstance(cutoff_date, datetime):
            return cutoff_date.date()
        if isinstance(cutoff_date, date):
            return cutoff_date
        if isinstance(cutoff_date, str):
            try:
                return date.fromisoformat(cutoff_date)
            except ValueError as exc:
                raise ValueError(
                    "cutoff_date string must be ISO format YYYY-MM-DD"
                ) from exc
        raise TypeError("cutoff_date must be a datetime.date, datetime.datetime, or ISO string")

