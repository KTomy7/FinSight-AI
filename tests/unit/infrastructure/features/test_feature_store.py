import pandas as pd
import pytest

from finsight.infrastructure.features.feature_store import PandasFeatureStore


def test_frame_date_range_rejects_invalid_dates() -> None:
    store = PandasFeatureStore()
    frame = pd.DataFrame({"date": [None, None]})

    with pytest.raises(ValueError, match="does not contain valid dates"):
        store.frame_date_range(frame)


def test_row_count_rejects_non_dataframe_input() -> None:
    store = PandasFeatureStore()

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        store.row_count(dataset=[{"date": "2024-01-01"}])



