from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from finsight.application.dto import RunSummary
from finsight.infrastructure.ml.run_registry import LocalFileRunRegistry


@pytest.fixture
def temp_artifact_dir(tmp_path: Path) -> str:
    """Fixture providing a temporary artifact directory."""
    return str(tmp_path / "artifacts")


class TestLocalFileRunRegistry:
    """Test LocalFileRunRegistry implementation."""

    def test_init_with_supported_model_ids(self) -> None:
        """Test initialization with supported model IDs."""
        model_ids = ("ridge", "xgboost", "naive_zero")
        registry = LocalFileRunRegistry(supported_model_ids=model_ids)
        assert registry.supported_model_ids == model_ids

    def test_init_without_supported_model_ids(self) -> None:
        """Test initialization without supported model IDs (dynamic growth)."""
        registry = LocalFileRunRegistry()
        assert registry.supported_model_ids is None

    def test_load_registry_missing_file_returns_none(self, temp_artifact_dir: str) -> None:
        """When registry file doesn't exist, load_registry returns None."""
        registry = LocalFileRunRegistry()
        result = registry.load_registry(artifact_root=temp_artifact_dir)
        assert result is None

    def test_load_registry_corrupt_json_returns_none(self, temp_artifact_dir: str) -> None:
        """When registry JSON is corrupt, load_registry returns None."""
        Path(temp_artifact_dir).mkdir(parents=True, exist_ok=True)
        registry_path = Path(temp_artifact_dir) / "registry.json"
        registry_path.write_text("{ invalid json }")

        registry = LocalFileRunRegistry()
        result = registry.load_registry(artifact_root=temp_artifact_dir)
        assert result is None

    def test_load_registry_with_valid_file(self, temp_artifact_dir: str) -> None:
        """When registry file is valid, load_registry returns RegistrySnapshot."""
        Path(temp_artifact_dir).mkdir(parents=True, exist_ok=True)
        registry_data = {
            "updated_at": "2026-05-16T12:00:00Z",
            "best_ridge": {
                "run_id": "2026-05-16T120000Z__ridge",
                "created_at": "2026-05-16T12:00:00Z",
                "metrics": {"mae": 0.123, "rmse": 0.45},
            },
            "best_xgboost": None,
        }
        registry_path = Path(temp_artifact_dir) / "registry.json"
        registry_path.write_text(json.dumps(registry_data))

        registry = LocalFileRunRegistry()
        result = registry.load_registry(artifact_root=temp_artifact_dir)

        assert result is not None
        assert result.updated_at == "2026-05-16T12:00:00Z"
        assert "ridge" in result.best_by_model
        assert result.best_by_model["ridge"]["run_id"] == "2026-05-16T120000Z__ridge"
        assert "xgboost" not in result.best_by_model  # null entries not included

    def test_record_completed_run_first_run_creates_file(self, temp_artifact_dir: str) -> None:
        """First training run creates registry.json with the new run."""
        registry = LocalFileRunRegistry(
            supported_model_ids=("ridge", "xgboost", "naive_zero")
        )

        run_summary = RunSummary(
            run_id="2026-05-16T120000Z__ridge",
            model_id="ridge",
            created_at="2026-05-16T12:00:00Z",
            metrics={"mae": 0.123, "rmse": 0.45, "direction_accuracy": 0.78},
        )

        registry.record_completed_run(artifact_root=temp_artifact_dir, run_summary=run_summary)

        # Verify file was created
        registry_path = Path(temp_artifact_dir) / "registry.json"
        assert registry_path.exists()

        # Verify content
        with registry_path.open("r") as f:
            data = json.load(f)

        assert data["best_ridge"]["run_id"] == "2026-05-16T120000Z__ridge"
        assert data["best_ridge"]["metrics"]["mae"] == 0.123
        assert "best_xgboost" in data

    def test_record_completed_run_better_run_updates_registry(self, temp_artifact_dir: str) -> None:
        """When new run is better, registry is updated."""
        Path(temp_artifact_dir).mkdir(parents=True, exist_ok=True)

        # Create initial registry with worse run
        initial_data = {
            "updated_at": "2026-05-16T110000Z",
            "best_ridge": {
                "run_id": "2026-05-16T110000Z__ridge",
                "created_at": "2026-05-16T11:00:00Z",
                "metrics": {"mae": 0.200, "rmse": 0.50},
            },
        }
        registry_path = Path(temp_artifact_dir) / "registry.json"
        registry_path.write_text(json.dumps(initial_data))

        # Record a better run
        registry = LocalFileRunRegistry()
        new_run = RunSummary(
            run_id="2026-05-16T120000Z__ridge",
            model_id="ridge",
            created_at="2026-05-16T12:00:00Z",
            metrics={"mae": 0.100, "rmse": 0.30},  # Better MAE
        )

        registry.record_completed_run(artifact_root=temp_artifact_dir, run_summary=new_run)

        # Verify registry was updated
        with registry_path.open("r") as f:
            data = json.load(f)

        assert data["best_ridge"]["run_id"] == "2026-05-16T120000Z__ridge"
        assert data["best_ridge"]["metrics"]["mae"] == 0.100

    def test_record_completed_run_worse_run_does_not_update(self, temp_artifact_dir: str) -> None:
        """When new run is worse, registry is not updated."""
        Path(temp_artifact_dir).mkdir(parents=True, exist_ok=True)

        # Create initial registry with good run
        initial_data = {
            "updated_at": "2026-05-16T110000Z",
            "best_ridge": {
                "run_id": "2026-05-16T110000Z__ridge",
                "created_at": "2026-05-16T11:00:00Z",
                "metrics": {"mae": 0.100, "rmse": 0.30},
            },
        }
        registry_path = Path(temp_artifact_dir) / "registry.json"
        registry_path.write_text(json.dumps(initial_data))

        # Record a worse run
        registry = LocalFileRunRegistry()
        worse_run = RunSummary(
            run_id="2026-05-16T120000Z__ridge",
            model_id="ridge",
            created_at="2026-05-16T12:00:00Z",
            metrics={"mae": 0.200, "rmse": 0.50},  # Worse MAE
        )

        registry.record_completed_run(artifact_root=temp_artifact_dir, run_summary=worse_run)

        # Verify registry was NOT updated
        with registry_path.open("r") as f:
            data = json.load(f)

        assert data["best_ridge"]["run_id"] == "2026-05-16T110000Z__ridge"
        assert data["best_ridge"]["metrics"]["mae"] == 0.100

    def test_record_completed_run_missing_metric_does_not_update(self, temp_artifact_dir: str) -> None:
        """When new run is missing a metric, it is not considered better."""
        Path(temp_artifact_dir).mkdir(parents=True, exist_ok=True)

        initial_data = {
            "updated_at": "2026-05-16T110000Z",
            "best_ridge": {
                "run_id": "2026-05-16T110000Z__ridge",
                "created_at": "2026-05-16T11:00:00Z",
                "metrics": {"mae": 0.100, "rmse": 0.30},
            },
        }
        registry_path = Path(temp_artifact_dir) / "registry.json"
        registry_path.write_text(json.dumps(initial_data))

        # Record run missing mae metric
        registry = LocalFileRunRegistry()
        incomplete_run = RunSummary(
            run_id="2026-05-16T120000Z__ridge",
            model_id="ridge",
            created_at="2026-05-16T12:00:00Z",
            metrics={"rmse": 0.05},  # Missing mae
        )

        registry.record_completed_run(artifact_root=temp_artifact_dir, run_summary=incomplete_run)

        # Verify registry was NOT updated
        with registry_path.open("r") as f:
            data = json.load(f)

        assert data["best_ridge"]["run_id"] == "2026-05-16T110000Z__ridge"

    def test_record_completed_run_multiple_models(self, temp_artifact_dir: str) -> None:
        """Registry correctly tracks multiple models independently."""
        Path(temp_artifact_dir).mkdir(parents=True, exist_ok=True)

        registry = LocalFileRunRegistry(
            supported_model_ids=("ridge", "xgboost")
        )

        # Record ridge run
        ridge_run = RunSummary(
            run_id="2026-05-16T110000Z__ridge",
            model_id="ridge",
            created_at="2026-05-16T11:00:00Z",
            metrics={"mae": 0.150},
        )
        registry.record_completed_run(artifact_root=temp_artifact_dir, run_summary=ridge_run)

        # Record xgboost run
        xgboost_run = RunSummary(
            run_id="2026-05-16T110000Z__xgboost",
            model_id="xgboost",
            created_at="2026-05-16T11:00:00Z",
            metrics={"mae": 0.120},
        )
        registry.record_completed_run(artifact_root=temp_artifact_dir, run_summary=xgboost_run)

        # Verify both models in registry
        registry_path = Path(temp_artifact_dir) / "registry.json"
        with registry_path.open("r") as f:
            data = json.load(f)

        assert data["best_ridge"]["run_id"] == "2026-05-16T110000Z__ridge"
        assert data["best_xgboost"]["run_id"] == "2026-05-16T110000Z__xgboost"

    def test_save_registry_atomic_creates_file_atomically(self, temp_artifact_dir: str) -> None:
        """Registry file is written atomically (not partially written)."""
        registry = LocalFileRunRegistry()

        best_by_model = {
            "ridge": {
                "run_id": "2026-05-16T120000Z__ridge",
                "created_at": "2026-05-16T12:00:00Z",
                "metrics": {"mae": 0.123},
            }
        }

        registry._save_registry_atomic(artifact_root=temp_artifact_dir, best_by_model=best_by_model)

        # Verify file exists and is valid JSON
        registry_path = Path(temp_artifact_dir) / "registry.json"
        assert registry_path.exists()

        with registry_path.open("r") as f:
            data = json.load(f)

        assert "best_ridge" in data
        assert data["updated_at"].endswith("Z")

    def test_registry_filename_constant(self) -> None:
        """Verify the registry filename constant is set correctly."""
        assert LocalFileRunRegistry.REGISTRY_FILENAME == "registry.json"


__all__ = []

