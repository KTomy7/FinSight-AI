from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from finsight.application.dto import LoadModelRunRequest
from finsight.application.use_cases.load_model_run import LoadModelRun
from finsight.infrastructure.ml.registry import LocalModelRegistry


def test_load_model_run_returns_artifact_manifest_metrics_and_predict_metadata(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    artifacts_dir = tmp_path / "runs"
    model_run_id = "2026-04-01T101500Z__naive_zero"

    registry.save_run(
        artifact_root=str(artifacts_dir),
        model_run_id=model_run_id,
        model_artifact={
            "model_type": "naive_zero",
            "predict_metadata": {
                "model_type": "naive_zero",
                "target_column": "target_ret_1d",
            },
        },
        manifest={
            "run_id": model_run_id,
            "model_id": "naive_zero",
            "feature_columns": ["ret_1d"],
            "target": "target_ret_1d",
            "split_policy": {
                "name": "time_split",
                "cutoff_date": "2025-06-01",
                "date_col": "date",
                "inclusive_test": True,
            },
            "dates": {
                "requested_start": "2024-04-01",
                "requested_end": "2026-04-01",
                "train_min": "2024-04-01",
                "train_max": "2025-05-31",
                "test_min": "2025-06-01",
                "test_max": "2026-04-01",
            },
            "params": {"tickers": ["AAPL"], "interval": "1d", "years": 2, "horizon": "1d"},
            "artifact_paths": {},
            "created_at": "2026-04-01T10:15:00Z",
        },
        metrics={"mae": 0.01, "rmse": 0.02, "n_train": 100, "n_test": 20},
        predictions=pd.DataFrame(
            [
                {"date": "2026-03-30", "ticker": "AAPL", "y_true": 0.01, "y_pred": 0.0},
            ]
        ),
    )

    use_case = LoadModelRun(model_registry=registry)
    result = use_case.execute(LoadModelRunRequest(model_run_id=model_run_id, artifacts_dir=str(artifacts_dir)))

    assert result.model_run_id == model_run_id
    assert result.manifest["model_id"] == "naive_zero"
    assert result.metrics["n_test"] == 20
    assert result.predict_metadata is not None
    assert result.predict_metadata["target_column"] == "target_ret_1d"
    model_artifact = cast(dict[str, object], result.model_artifact)
    assert model_artifact["model_type"] == "naive_zero"


