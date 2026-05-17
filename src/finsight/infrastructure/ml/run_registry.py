from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finsight.application.dto import RegistrySnapshot, RunSummary
from finsight.domain.ports import RunRegistryPort
from finsight.infrastructure.ml.model_ranker import ModelRanker


class LocalFileRunRegistry(RunRegistryPort):
    """
    Local file-based implementation of the run registry.

    Stores best run per model in artifacts/registry.json.
    Registry format:
    {
      "updated_at": "2026-05-16T12:00:00Z",
      "best_ridge": {
        "run_id": "...",
        "created_at": "...",
        "metrics": {"mae": 0.123, ...}
      },
      "best_xgboost": null,  # null if model hasn't run yet
      ...
    }
    """

    REGISTRY_FILENAME = "registry.json"

    def __init__(self, *, supported_model_ids: tuple[str, ...] | None = None) -> None:
        """
        Initialize the registry service.

        Args:
            supported_model_ids: Tuple of model IDs to initialize in registry.
                                If None, registry will grow dynamically.
        """
        self.supported_model_ids = supported_model_ids

    def load_registry(self, *, artifact_root: str) -> RegistrySnapshot | None:
        """
        Load the registry from disk.

        Returns: RegistrySnapshot with current best runs, or None if file missing/corrupt.
        """
        registry_path = Path(artifact_root) / self.REGISTRY_FILENAME

        if not registry_path.exists():
            return None

        try:
            with registry_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return None

            updated_at = data.get("updated_at", "")
            best_by_model = {}

            # Extract all best_<model_id> entries
            for key, value in data.items():
                if key.startswith("best_") and value is not None:
                    model_id = key[5:]  # remove "best_" prefix
                    best_by_model[model_id] = value

            return RegistrySnapshot(updated_at=updated_at, best_by_model=best_by_model)
        except Exception:
            # Corrupt file, return None to signal graceful fallback
            return None

    def record_completed_run(self, *, artifact_root: str, run_summary: RunSummary) -> None:
        """
        Record a completed training run.

        Loads current registry, compares new run against current best for that model,
        updates if new is better (by MAE metric), and saves registry atomically.
        """
        model_id = run_summary.model_id
        artifact_root_path = Path(artifact_root)
        registry_path = artifact_root_path / self.REGISTRY_FILENAME

        # Load current registry or initialize if missing
        current_registry = self.load_registry(artifact_root=artifact_root)

        if current_registry is None:
            # First training or corrupt file: initialize with all models
            best_by_model = self._initialize_best_by_model()
        else:
            best_by_model = dict(current_registry.best_by_model)

        # Prepare new run entry
        new_run_entry = {
            "run_id": run_summary.run_id,
            "created_at": run_summary.created_at,
            "metrics": run_summary.metrics,
        }

        # Decide if new run is better than current best
        current_best = best_by_model.get(model_id)
        should_update = True

        if current_best is not None and isinstance(current_best, dict):
            # Compare using MAE metric (primary ranking metric)
            ranker = ModelRanker(
                rank_by=["mae"],
                metric_directions={"mae": "asc"},
            )

            new_metrics = run_summary.metrics
            current_metrics = current_best.get("metrics", {})

            # is_better returns False if any metric is missing
            if not ranker.is_better(new_metrics, current_metrics):
                should_update = False

        # Update best run for this model if new is better
        if should_update:
            best_by_model[model_id] = new_run_entry

        # Save registry atomically
        self._save_registry_atomic(
            artifact_root=artifact_root,
            best_by_model=best_by_model,
        )

    def _initialize_best_by_model(self) -> dict[str, dict[str, Any] | None]:
        """
        Initialize best_by_model dict with all known models set to None.

        Used on first run or when registry file is missing.
        """
        best_by_model: dict[str, dict[str, Any] | None] = {}

        if self.supported_model_ids:
            for model_id in self.supported_model_ids:
                best_by_model[f"best_{model_id}"] = None

        return best_by_model

    def _save_registry_atomic(
        self,
        *,
        artifact_root: str,
        best_by_model: dict[str, dict[str, Any] | None],
    ) -> None:
        """
        Save registry to disk atomically using temp file + rename.

        Prevents partial writes if save is interrupted.
        """
        artifact_root_path = Path(artifact_root)
        registry_path = artifact_root_path / self.REGISTRY_FILENAME
        temp_path = artifact_root_path / f"{self.REGISTRY_FILENAME}.tmp"

        # Ensure artifact directory exists
        artifact_root_path.mkdir(parents=True, exist_ok=True)

        # Build registry dict
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        registry_dict: dict[str, Any] = {
            "updated_at": now_iso,
        }

        # Add all best_<model_id> entries
        for key, value in best_by_model.items():
            if key.startswith("best_"):
                registry_dict[key] = value
            else:
                registry_dict[f"best_{key}"] = value

        # Write atomically: write to temp, then rename
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(registry_dict, f, indent=2)

            # Atomic rename (on most filesystems)
            temp_path.replace(registry_path)
        except Exception:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            # Re-raise to let caller handle
            raise


__all__ = ["LocalFileRunRegistry"]

