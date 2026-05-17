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
        if df.empty:
            raise ValueError(
                "Input DataFrame is empty. Cannot perform time split on an empty dataset."
            )

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
            min_ts = parsed_dates.min()
            max_ts = parsed_dates.max()
            # min/max should never be NaT here (invalid dates were already checked),
            # but handle defensively to provide clear error message.
            if pd.isna(min_ts) or pd.isna(max_ts):
                raise ValueError(
                    f"Time split produced an empty partition and dataset has no valid dates. "
                    f"cutoff_date={self.cutoff_date.isoformat()}."
                )
            min_date = min_ts.date().isoformat()
            max_date = max_ts.date().isoformat()
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


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass(frozen=True, slots=True)
class WalkForwardSplitPolicy:
    min_train_size: int
    test_size: int
    step_size: int
    date_col: str = "date"
    max_folds: int | None = None

    def __post_init__(self) -> None:
        if self.min_train_size <= 0:
            raise ValueError("min_train_size must be a positive integer.")
        if self.test_size <= 0:
            raise ValueError("test_size must be a positive integer.")
        if self.step_size <= 0:
            raise ValueError("step_size must be a positive integer.")
        if self.max_folds is not None and self.max_folds <= 0:
            raise ValueError("max_folds must be None or a positive integer.")

    def split_frame(self, df: pd.DataFrame) -> list[WalkForwardFold]:
        if df.empty:
            raise ValueError(
                "Input DataFrame is empty. Cannot perform walk-forward split on an empty dataset."
            )
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
        unique_dates = sorted(parsed_dates.dt.normalize().unique())
        n_dates = len(unique_dates)

        min_required = self.min_train_size + self.test_size
        if n_dates < min_required:
            raise ValueError(
                "Walk-forward split cannot produce any fold. "
                f"Need at least {min_required} unique dates, found {n_dates}."
            )

        sort_cols = ["ticker", self.date_col] if {"ticker", self.date_col}.issubset(frame.columns) else [self.date_col]
        folds: list[WalkForwardFold] = []
        train_cutoff_idx = self.min_train_size

        while train_cutoff_idx + self.test_size <= n_dates:
            train_end_ts = pd.Timestamp(unique_dates[train_cutoff_idx - 1]).normalize()
            test_start_ts = pd.Timestamp(unique_dates[train_cutoff_idx]).normalize()
            test_end_ts = pd.Timestamp(unique_dates[train_cutoff_idx + self.test_size - 1]).normalize()

            train_mask = frame[self.date_col].dt.normalize() <= train_end_ts
            test_mask = (
                (frame[self.date_col].dt.normalize() >= test_start_ts)
                & (frame[self.date_col].dt.normalize() <= test_end_ts)
            )

            train_df = frame.loc[train_mask].copy()
            test_df = frame.loc[test_mask].copy()
            if train_df.empty or test_df.empty:
                raise ValueError(
                    "Walk-forward split produced an empty partition. "
                    f"train_end={train_end_ts.date().isoformat()}, "
                    f"test_start={test_start_ts.date().isoformat()}, "
                    f"test_end={test_end_ts.date().isoformat()}."
                )

            train_df = train_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
            test_df = test_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

            folds.append(
                WalkForwardFold(
                    fold_index=len(folds) + 1,
                    train_start=train_df[self.date_col].min().date(),
                    train_end=train_end_ts.date(),
                    test_start=test_start_ts.date(),
                    test_end=test_end_ts.date(),
                    train_df=train_df,
                    test_df=test_df,
                )
            )

            if self.max_folds is not None and len(folds) >= self.max_folds:
                break
            train_cutoff_idx += self.step_size

        if not folds:
            raise ValueError("Walk-forward split cannot produce any fold with the configured parameters.")
        return folds


