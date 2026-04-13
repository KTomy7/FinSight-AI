from __future__ import annotations

from typing import Sequence

import pandas as pd

from finsight.domain.entities import OHLCVSeries
from finsight.domain.ports import FeatureStorePort
from finsight.infrastructure.features.feature_pipeline import (
    FEATURE_COLUMNS,
    build_feature_dataset,
    build_inference_feature_dataset,
)
from finsight.infrastructure.features.policies import TimeSplitPolicy


class PandasFeatureStore(FeatureStorePort):
    def build_feature_dataset(self, series_list: Sequence[OHLCVSeries]) -> pd.DataFrame:
        return build_feature_dataset(list(series_list))

    def build_inference_feature_dataset(self, series_list: Sequence[OHLCVSeries]) -> pd.DataFrame:
        return build_inference_feature_dataset(list(series_list))

    def split_train_test(
        self,
        dataset: object,
        *,
        cutoff_date: str,
        date_col: str = "date",
        inclusive_test: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = self._require_dataframe(dataset, arg_name="dataset")
        policy = TimeSplitPolicy(
            cutoff_date=cutoff_date,
            date_col=date_col,
            inclusive_test=inclusive_test,
        )
        return policy.split_frame(frame)

    def frame_date_range(self, dataset: object, *, date_col: str = "date") -> tuple[str, str]:
        frame = self._require_dataframe(dataset, arg_name="dataset")
        parsed = pd.to_datetime(frame[date_col], errors="coerce")
        min_ts = parsed.min()
        max_ts = parsed.max()
        if pd.isna(min_ts) or pd.isna(max_ts):
            raise ValueError(f"DataFrame column '{date_col}' does not contain valid dates.")
        return min_ts.date().isoformat(), max_ts.date().isoformat()

    def row_count(self, dataset: object) -> int:
        frame = self._require_dataframe(dataset, arg_name="dataset")
        return int(len(frame))

    def feature_columns(self) -> tuple[str, ...]:
        return tuple(FEATURE_COLUMNS)

    @staticmethod
    def _require_dataframe(dataset: object, *, arg_name: str) -> pd.DataFrame:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"{arg_name} must be a pandas DataFrame.")
        return dataset

