import json
from pathlib import Path

import pandas as pd

from finsight.infrastructure.ml.registry import LocalModelRegistry


def test_save_and_load_run_by_model_run_id_round_trip(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-01T101500Z__naive_zero"

    model_artifact = {
        "model_type": "naive_zero",
        "predict_metadata": {"target": "target_ret_1d", "horizon": "1d"},
    }
    manifest = {
        "run_id": model_run_id,
        "model_id": "naive_zero",
        "feature_columns": ["ret_1d", "mom_20d"],
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
    }
    metrics = {"mae": 0.01, "rmse": 0.02, "direction_accuracy": 0.6, "n_train": 100, "n_test": 20}

    predictions = pd.DataFrame(
        [
            {"date": "2026-03-30", "ticker": "AAPL", "y_true": 0.01, "y_pred": 0.0},
            {"date": "2026-03-31", "ticker": "AAPL", "y_true": -0.01, "y_pred": 0.0},
        ]
    )

    run_dir = registry.save_run(
        artifact_root=str(artifact_root),
        model_run_id=model_run_id,
        model_artifact=model_artifact,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
    )

    loaded = registry.load_run(artifact_root=str(artifact_root), model_run_id=model_run_id)

    assert Path(run_dir).name == model_run_id
    assert loaded["model_run_id"] == model_run_id
    assert loaded["manifest"]["run_id"] == model_run_id
    assert loaded["metrics"]["n_test"] == 20
    assert loaded["model_artifact"]["predict_metadata"]["target"] == "target_ret_1d"


def test_save_run_persists_manifest_and_metrics_consistently(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-01T123000Z__naive_mean"

    manifest = {"run_id": model_run_id, "model_id": "naive_mean"}
    metrics = {"mae": 0.03, "rmse": 0.04}

    run_dir = Path(
        registry.save_run(
            artifact_root=str(artifact_root),
            model_run_id=model_run_id,
            model_artifact={"weights": [0.0]},
            manifest=manifest,
            metrics=metrics,
        )
    )

    saved_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert saved_manifest == manifest
    assert saved_metrics == metrics

