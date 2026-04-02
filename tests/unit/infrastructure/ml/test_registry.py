import json
import os
from pathlib import Path

import pandas as pd
import pytest

from finsight.infrastructure.ml.registry import FileSystemModelRegistry


def test_save_and_load_round_trip_by_model_run_id(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-02T101500Z__naive_zero"

    run_dir = registry.create_run_dir(artifact_root=str(artifact_root), run_id=model_run_id)
    metrics = {"mae": 0.01, "n_train": 50, "n_test": 20}
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
            "requested_start": "2024-01-01",
            "requested_end": "2026-03-17",
            "train_min": "2024-01-01",
            "train_max": "2025-05-31",
            "test_min": "2025-06-01",
            "test_max": "2026-03-17",
        },
        "params": {"tickers": ["AAPL"], "interval": "1d", "years": 2, "horizon": "1d"},
        "artifact_paths": {
            "run_dir": run_dir,
            "metrics": str(Path(run_dir) / "metrics.json"),
            "manifest": str(Path(run_dir) / "manifest.json"),
            "predictions": str(Path(run_dir) / "predictions.csv"),
        },
        "created_at": "2026-04-02T10:15:00Z",
    }
    predictions = pd.DataFrame(
        [
            {"date": "2026-03-15", "ticker": "AAPL", "y_true": 0.01, "y_pred": 0.0},
            {"date": "2026-03-16", "ticker": "AAPL", "y_true": -0.02, "y_pred": 0.0},
        ]
    )

    registry.save_metrics(run_dir=run_dir, metrics=metrics)
    registry.save_manifest(run_dir=run_dir, manifest=manifest)
    registry.save_predictions(run_dir=run_dir, predictions=predictions)

    loaded = registry.load_run(artifact_root=str(artifact_root), model_run_id=model_run_id)

    assert loaded.model_run_id == model_run_id
    assert loaded.run_dir == run_dir
    assert loaded.metrics == metrics
    assert loaded.manifest["model_id"] == "naive_zero"
    assert loaded.manifest["run_id"] == model_run_id


def test_persists_metrics_manifest_predictions_consistently(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-02T102000Z__naive_mean"
    run_dir = registry.create_run_dir(artifact_root=str(artifact_root), run_id=model_run_id)

    metrics = {"rmse": 0.5, "n_train": 100, "n_test": 30}
    manifest = {
        "run_id": model_run_id,
        "model_id": "naive_mean",
        "feature_columns": ["ret_1d"],
        "target": "target_ret_1d",
        "split_policy": {
            "name": "time_split",
            "cutoff_date": "2025-06-01",
            "date_col": "date",
            "inclusive_test": True,
        },
        "dates": {
            "requested_start": "2024-01-01",
            "requested_end": "2026-03-17",
            "train_min": "2024-01-01",
            "train_max": "2025-05-31",
            "test_min": "2025-06-01",
            "test_max": "2026-03-17",
        },
        "params": {"tickers": ["AAPL", "MSFT"], "interval": "1d", "years": 2, "horizon": "1d"},
        "artifact_paths": {
            "run_dir": run_dir,
            "metrics": str(Path(run_dir) / "metrics.json"),
            "manifest": str(Path(run_dir) / "manifest.json"),
            "predictions": str(Path(run_dir) / "predictions.csv"),
        },
        "created_at": "2026-04-02T10:20:00Z",
    }

    registry.save_metrics(run_dir=run_dir, metrics=metrics)
    registry.save_manifest(run_dir=run_dir, manifest=manifest)
    registry.save_predictions(
        run_dir=run_dir,
        predictions=[{"date": "2026-03-17", "ticker": "AAPL", "y_true": 0.2, "y_pred": 0.1}],
    )

    metrics_payload = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    predictions_payload = pd.read_csv(Path(run_dir) / "predictions.csv")

    assert metrics_payload == metrics
    assert manifest_payload["artifact_paths"]["run_dir"] == run_dir
    assert list(predictions_payload.columns) == ["date", "ticker", "y_true", "y_pred"]

    loaded = registry.load_run(artifact_root=str(artifact_root), model_run_id=model_run_id)
    assert loaded.metrics["rmse"] == 0.5
    assert loaded.manifest["model_id"] == "naive_mean"


def test_save_predictions_from_list_of_dicts_writes_expected_csv_structure(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()
    run_dir = registry.create_run_dir(
        artifact_root=str(tmp_path / "runs"),
        run_id="2026-04-02T104500Z__naive_zero",
    )

    registry.save_predictions(
        run_dir=run_dir,
        predictions=[
            {"date": "2026-03-17", "ticker": "AAPL", "y_true": 0.2, "y_pred": 0.1},
            {
                "date": "2026-03-18",
                "ticker": "MSFT",
                "y_true": -0.3,
                "y_pred": -0.1,
                "model_id": "naive_zero",
            },
        ],
    )

    predictions_payload = pd.read_csv(Path(run_dir) / "predictions.csv")

    assert list(predictions_payload.columns) == ["date", "ticker", "y_true", "y_pred", "model_id"]
    assert predictions_payload.loc[0, "date"] == "2026-03-17"
    assert predictions_payload.loc[0, "ticker"] == "AAPL"
    assert pd.isna(predictions_payload.loc[0, "model_id"])
    assert predictions_payload.loc[1, "model_id"] == "naive_zero"


def test_load_missing_run_id_raises_clear_error(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()

    with pytest.raises(FileNotFoundError, match="was not found"):
        registry.load_run(artifact_root=str(tmp_path / "runs"), model_run_id="missing_run")


def test_load_missing_run_id_does_not_create_artifact_root(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()
    artifact_root = tmp_path / "runs"

    with pytest.raises(FileNotFoundError, match="was not found"):
        registry.load_run(artifact_root=str(artifact_root), model_run_id="missing_run")

    assert not artifact_root.exists()


def test_load_run_rejects_run_dir_resolving_outside_artifact_root(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir(parents=True, exist_ok=True)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    (outside_dir / "metrics.json").write_text(json.dumps({"mae": 0.1}), encoding="utf-8")
    (outside_dir / "manifest.json").write_text(json.dumps({"model_id": "naive_zero"}), encoding="utf-8")
    (outside_dir / "predictions.csv").write_text("date,ticker,y_true,y_pred\n", encoding="utf-8")

    run_id = "2026-04-02T114500Z__symlink"
    symlink_path = artifact_root / run_id
    try:
        os.symlink(str(outside_dir), str(symlink_path), target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation not available in this environment: {exc}")

    with pytest.raises(FileNotFoundError, match="was not found"):
        registry.load_run(artifact_root=str(artifact_root), model_run_id=run_id)


def test_load_run_rejects_non_scalar_metric_values(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-02T120000Z__bad_metrics"
    run_dir = Path(registry.create_run_dir(artifact_root=str(artifact_root), run_id=model_run_id))

    (run_dir / "metrics.json").write_text(
        json.dumps({"mae": 0.1, "nested": {"value": 1}}),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(json.dumps({"model_id": "naive_zero"}), encoding="utf-8")
    (run_dir / "predictions.csv").write_text("date,ticker,y_true,y_pred\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Metric 'nested' has unsupported type"):
        registry.load_run(artifact_root=str(artifact_root), model_run_id=model_run_id)


@pytest.mark.parametrize(
    "invalid_run_id,expected_error",
    [
        ("", "non-empty string"),
        (" run", "leading or trailing whitespace"),
        ("run ", "leading or trailing whitespace"),
        ("   ", "leading or trailing whitespace"),
        ("a" * 256, "<= 255"),
        ("bad/run", "may only contain"),
        ("bad\\run", "may only contain"),
        ("bad*run", "may only contain"),
    ],
)
def test_create_run_dir_rejects_invalid_run_id(
    tmp_path: Path,
    invalid_run_id: str,
    expected_error: str,
) -> None:
    registry = FileSystemModelRegistry()

    with pytest.raises(ValueError, match=expected_error):
        registry.create_run_dir(artifact_root=str(tmp_path / "runs"), run_id=invalid_run_id)


def test_create_run_dir_rejects_non_string_run_id_type(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()

    with pytest.raises(TypeError, match="run_id must be a string"):
        registry.create_run_dir(artifact_root=str(tmp_path / "runs"), run_id=123)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_filename", ["nested/model.bin", "nested\\model.bin"])
def test_save_model_artifact_rejects_filename_with_path_separators(
    tmp_path: Path,
    invalid_filename: str,
) -> None:
    registry = FileSystemModelRegistry()
    run_dir = registry.create_run_dir(
        artifact_root=str(tmp_path / "runs"),
        run_id="2026-04-02T113000Z__naive_zero",
    )

    with pytest.raises(ValueError, match="must not include directory separators"):
        registry.save_model_artifact(
            run_dir=run_dir,
            model_artifact=b"artifact-bytes",
            filename=invalid_filename,
        )


def test_save_model_artifact_rejects_non_string_filename_type(tmp_path: Path) -> None:
    registry = FileSystemModelRegistry()
    run_dir = registry.create_run_dir(
        artifact_root=str(tmp_path / "runs"),
        run_id="2026-04-02T121500Z__naive_zero",
    )

    with pytest.raises(TypeError, match="filename must be a string"):
        registry.save_model_artifact(
            run_dir=run_dir,
            model_artifact=b"artifact-bytes",
            filename=123,  # type: ignore[arg-type]
        )

