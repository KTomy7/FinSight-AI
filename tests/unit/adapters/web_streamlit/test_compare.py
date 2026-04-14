from __future__ import annotations

import pandas as pd

from finsight.adapters.web_streamlit.presenters import ComparisonPresenter
from finsight.application.dto import CompareModelsResult, ModelComparisonRow


def test_build_comparison_frame_orders_columns_and_applies_labels() -> None:
    """Test that comparison frame formatting works correctly (via presenter)."""
    result = CompareModelsResult(
        rows=[
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="2026-04-12T120000Z__ridge",
                metrics={"mae": 0.09, "rmse": 0.18, "direction_accuracy": 0.83, "extra": 7},
                sort_key=(0.09, 0.18, -0.83, "ridge", "2026-04-12T120000Z__ridge"),
            )
        ],
        rank_by=["mae", "rmse", "direction_accuracy"],
        metric_directions={"mae": "asc", "rmse": "asc", "direction_accuracy": "desc"},
    )

    frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup={"ridge": "Ridge Regression"})

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["rank", "model", "model_id", "run_id", "mae", "rmse", "direction_accuracy", "extra"]
    assert frame.iloc[0]["model"] == "Ridge Regression"
    assert frame.iloc[0]["rank"] == 1
    assert frame.iloc[0]["extra"] == 7

