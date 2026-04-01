import json
from pathlib import Path

import pandas as pd
import pytest

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


def test_save_run_rejects_existing_model_artifact(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-01T130000Z__naive_mean"

    registry.save_run(
        artifact_root=str(artifact_root),
        model_run_id=model_run_id,
        model_artifact={"weights": [0.1]},
        manifest={"run_id": model_run_id},
        metrics={"mae": 0.03},
    )

    with pytest.raises(FileExistsError, match="Run already exists"):
        registry.save_run(
            artifact_root=str(artifact_root),
            model_run_id=model_run_id,
            model_artifact={"weights": [0.2]},
            manifest={"run_id": model_run_id},
            metrics={"mae": 0.01},
        )


@pytest.mark.parametrize(
    "missing_name, expected_message",
    [
        ("model.pkl", "Missing model artifact"),
        ("manifest.json", "Missing manifest"),
        ("metrics.json", "Missing metrics"),
    ],
)
def test_load_run_reports_missing_required_artifacts(
    tmp_path: Path,
    missing_name: str,
    expected_message: str,
) -> None:
    registry = LocalModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-01T140000Z__naive_zero"

    run_dir = artifact_root / model_run_id
    run_dir.mkdir(parents=True)
    (run_dir / "model.pkl").write_bytes(b"not-empty")
    (run_dir / "manifest.json").write_text('{"run_id": "x"}', encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"mae": 0.2}', encoding="utf-8")
    (run_dir / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=expected_message):
        registry.load_run(artifact_root=str(artifact_root), model_run_id=model_run_id)


def test_load_run_reports_missing_run_directory(tmp_path: Path) -> None:
    registry = LocalModelRegistry()

    with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
        registry.load_run(artifact_root=str(tmp_path / "runs"), model_run_id="2026-04-01T150000Z__naive")


def test_load_run_treats_non_parseable_predictions_as_empty_dataframe(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-01T151500Z__naive"

    run_dir = Path(
        registry.save_run(
            artifact_root=str(artifact_root),
            model_run_id=model_run_id,
            model_artifact={"weights": [0.0]},
            manifest={"run_id": model_run_id},
            metrics={"mae": 0.03},
        )
    )
    # A non-empty file containing only whitespace triggers EmptyDataError in read_csv.
    (run_dir / "predictions.csv").write_text("\n", encoding="utf-8")

    loaded = registry.load_run(artifact_root=str(artifact_root), model_run_id=model_run_id)

    assert "predictions" in loaded
    assert isinstance(loaded["predictions"], pd.DataFrame)
    assert loaded["predictions"].empty


def test_load_run_without_predictions_file_omits_predictions_key(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-04-01T152000Z__naive"

    registry.save_run(
        artifact_root=str(artifact_root),
        model_run_id=model_run_id,
        model_artifact={"weights": [0.0]},
        manifest={"run_id": model_run_id},
        metrics={"mae": 0.03},
    )

    loaded = registry.load_run(artifact_root=str(artifact_root), model_run_id=model_run_id)
    assert "predictions" not in loaded


def test_save_predictions_handles_empty_list_payload(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    registry.save_predictions(run_dir=str(run_dir), predictions=[])

    assert (run_dir / "predictions.csv").read_text(encoding="utf-8") == ""


def test_save_predictions_merges_fieldnames_from_all_rows(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    registry.save_predictions(
        run_dir=str(run_dir),
        predictions=[
            {"date": "2024-01-01", "ticker": "AAPL"},
            {"date": "2024-01-02", "ticker": "AAPL", "confidence": 0.91},
        ],
    )

    persisted = pd.read_csv(run_dir / "predictions.csv")
    assert "confidence" in persisted.columns
    assert persisted.loc[1, "confidence"] == pytest.approx(0.91)


def test_resolve_run_dir_rejects_model_run_id_with_traversal() -> None:
    with pytest.raises(ValueError, match="Invalid model_run_id"):
        LocalModelRegistry._resolve_run_dir(artifact_root="C:/tmp/runs", model_run_id="../escape")


def test_read_json_rejects_non_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('["not", "an", "object"]', encoding="utf-8")

    with pytest.raises(TypeError, match="Expected JSON object"):
        LocalModelRegistry._read_json(path)


def test_read_pickle_rejects_paths_outside_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"not-a-real-pickle")

    with pytest.raises(ValueError, match="Refusing to load pickle outside artifact_root"):
        LocalModelRegistry._read_pickle(outside, artifact_root=str(artifact_root))


def test_read_pickle_loads_when_artifact_root_not_enforced(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pkl"
    payload = {"version": 1}
    LocalModelRegistry._write_pickle(path, payload)

    loaded = LocalModelRegistry._read_pickle(path)
    assert loaded == payload


def test_save_metrics_and_manifest_write_json_files(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    registry.save_metrics(run_dir=str(run_dir), metrics={"mae": 0.1})
    registry.save_manifest(run_dir=str(run_dir), manifest={"run_id": "abc"})

    assert json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) == {"mae": 0.1}
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) == {"run_id": "abc"}


