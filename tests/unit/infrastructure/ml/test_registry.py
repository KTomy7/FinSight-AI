from pathlib import Path

import pandas as pd
import pytest

from finsight.application.contracts import build_run_manifest
from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_RMSE
from finsight.infrastructure.ml.registry import LocalFileModelRegistry
from finsight.infrastructure.ml.sklearn import NaiveBaselineModel


def test_local_file_model_registry_round_trip_loads_model_and_metadata(tmp_path: Path) -> None:
    registry = LocalFileModelRegistry()
    artifact_root = tmp_path / "runs"
    run_id = "2026-04-02T120000Z__naive_zero"

    run_dir = Path(registry.create_run_dir(artifact_root=str(artifact_root), run_id=run_id))

    model = NaiveBaselineModel()
    metrics = {
        METRIC_MAE: 0.1,
        METRIC_RMSE: 0.2,
        METRIC_DIRECTION_ACCURACY: 0.75,
        "n_train": 42,
        "n_test": 13,
    }
    manifest = build_run_manifest(
        run_id=run_dir.name,
        model_id="naive_zero",
        feature_columns=["ret_1d", "volume_zscore"],
        target="target_ret_1d",
        split_policy={
            "name": "time_split",
            "cutoff_date": "2026-03-01",
            "date_col": "date",
            "inclusive_test": True,
        },
        dates={
            "requested_start": "2025-04-02",
            "requested_end": "2026-04-02",
            "train_min": "2025-04-02",
            "train_max": "2026-02-28",
            "test_min": "2026-03-01",
            "test_max": "2026-04-02",
        },
        params={"tickers": ["AAPL", "JPM"], "interval": "1d", "years": 1, "horizon": "1d"},
        artifact_paths={
            "run_dir": str(run_dir),
            "metrics": str(run_dir / "metrics.json"),
            "manifest": str(run_dir / "manifest.json"),
            "predictions": str(run_dir / "predictions.csv"),
        },
        created_at="2026-04-02T12:00:00Z",
    )

    registry.save_model(run_dir=str(run_dir), model=model)
    registry.save_metrics(run_dir=str(run_dir), metrics=metrics)
    registry.save_manifest(run_dir=str(run_dir), manifest=manifest)
    registry.save_predictions(
        run_dir=str(run_dir),
        predictions=pd.DataFrame(
            [
                {"date": "2026-03-01", "ticker": "AAPL", "y_true": 0.1, "y_pred": 0.0},
                {"date": "2026-03-02", "ticker": "JPM", "y_true": -0.1, "y_pred": 0.0},
            ]
        ),
    )

    loaded_model = registry.load_model(artifact_root=str(artifact_root), run_id=run_dir.name)
    loaded_metrics = registry.load_metrics(artifact_root=str(artifact_root), run_id=run_dir.name)
    loaded_manifest = registry.load_manifest(artifact_root=str(artifact_root), run_id=run_dir.name)
    loaded_bundle = registry.load_run_artifacts(artifact_root=str(artifact_root), run_id=run_dir.name)

    assert isinstance(loaded_model, NaiveBaselineModel)
    assert loaded_metrics[METRIC_MAE] == metrics[METRIC_MAE]
    assert loaded_metrics[METRIC_DIRECTION_ACCURACY] == metrics[METRIC_DIRECTION_ACCURACY]
    assert loaded_manifest["run_id"] == run_dir.name
    assert loaded_manifest["model_id"] == "naive_zero"
    assert loaded_bundle.run_id == run_dir.name
    assert loaded_bundle.run_dir == str(run_dir)
    assert isinstance(loaded_bundle.model, NaiveBaselineModel)
    assert loaded_bundle.metrics[METRIC_RMSE] == metrics[METRIC_RMSE]
    assert loaded_bundle.manifest["artifact_paths"]["manifest"].endswith("manifest.json")
    assert Path(loaded_bundle.model_path).exists()
    assert Path(loaded_bundle.metrics_path).exists()
    assert Path(loaded_bundle.manifest_path).exists()
    assert Path(loaded_bundle.predictions_path).exists()


def test_save_predictions_writes_empty_file_for_empty_rows(tmp_path: Path) -> None:
    registry = LocalFileModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    registry.save_predictions(run_dir=str(run_dir), predictions=[])

    assert (run_dir / "predictions.csv").read_text(encoding="utf-8") == ""


def test_save_predictions_merges_fieldnames_from_all_rows(tmp_path: Path) -> None:
    registry = LocalFileModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    registry.save_predictions(
        run_dir=str(run_dir),
        predictions=[
            {"date": "2026-03-01", "ticker": "AAPL"},
            {"date": "2026-03-02", "ticker": "AAPL", "y_pred": 0.1},
        ],
    )

    loaded = pd.read_csv(run_dir / "predictions.csv")
    assert {"date", "ticker", "y_pred"}.issubset(set(loaded.columns))


def test_load_metrics_raises_when_metrics_json_is_missing(tmp_path: Path) -> None:
    registry = LocalFileModelRegistry()

    with pytest.raises(FileNotFoundError, match="Missing model registry artifact"):
        registry.load_metrics(artifact_root=str(tmp_path), run_id="missing-run")


def test_load_metrics_rejects_non_object_json_payload(tmp_path: Path) -> None:
    registry = LocalFileModelRegistry()
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="must contain a JSON object"):
        registry.load_metrics(artifact_root=str(tmp_path / "runs"), run_id="run-a")


def test_load_run_artifacts_raises_for_missing_directory(tmp_path: Path) -> None:
    registry = LocalFileModelRegistry()

    with pytest.raises(FileNotFoundError, match="Model run directory does not exist"):
        registry.load_run_artifacts(artifact_root=str(tmp_path), run_id="missing-run")


def test_load_run_artifacts_raises_for_non_directory_path(tmp_path: Path) -> None:
    registry = LocalFileModelRegistry()
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    non_directory = artifact_root / "run-a"
    non_directory.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="Model run path is not a directory"):
        registry.load_run_artifacts(artifact_root=str(artifact_root), run_id="run-a")


