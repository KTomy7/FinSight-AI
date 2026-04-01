from __future__ import annotations

import csv
import json
import pickle
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from finsight.domain.ports import ModelRegistryPort


MODEL_FILE_NAME = "model.pkl"
MANIFEST_FILE_NAME = "manifest.json"
METRICS_FILE_NAME = "metrics.json"
PREDICTIONS_FILE_NAME = "predictions.csv"


class LocalModelRegistry(ModelRegistryPort):
    """Local filesystem-backed model registry rooted under artifacts/runs."""

    # Legacy API retained for compatibility with the current TrainModel flow.
    def create_run_dir(self, *, artifact_root: str, model_run_id: str) -> str:
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)

        base_path = self._resolve_run_dir(artifact_root=artifact_root, model_run_id=model_run_id)

        try:
            base_path.mkdir(parents=True, exist_ok=False)
            return str(base_path)
        except FileExistsError:
            pass

        suffix = 1
        while True:
            candidate = self._resolve_run_dir(
                artifact_root=artifact_root,
                model_run_id=f"{model_run_id}_{suffix}",
            )
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return str(candidate)
            except FileExistsError:
                suffix += 1

    def save_metrics(self, *, run_dir: str, metrics: Mapping[str, float | int | str]) -> None:
        self._write_json(Path(run_dir) / METRICS_FILE_NAME, metrics)

    def save_manifest(self, *, run_dir: str, manifest: Mapping[str, Any]) -> None:
        self._write_json(Path(run_dir) / MANIFEST_FILE_NAME, manifest)

    def save_predictions(self, *, run_dir: str, predictions: object) -> None:
        output_path = Path(run_dir) / PREDICTIONS_FILE_NAME

        if isinstance(predictions, pd.DataFrame):
            predictions.to_csv(output_path, index=False)
            return

        rows = self._normalize_rows(predictions)
        with output_path.open("w", newline="", encoding="utf-8") as file_obj:
            if not rows:
                file_obj.write("")
                return

            fieldnames = list(rows[0].keys())
            for row in rows[1:]:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_run(
        self,
        *,
        artifact_root: str,
        model_run_id: str,
        model_artifact: object,
        manifest: Mapping[str, Any],
        metrics: Mapping[str, float | int | str],
        predictions: object | None = None,
    ) -> str:
        run_dir = self._resolve_run_dir(artifact_root=artifact_root, model_run_id=model_run_id)
        self._validate_manifest_consistency(manifest=manifest, model_run_id=model_run_id, run_dir=run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prevent silent overwrites: fail if any expected artifacts already exist in this run.
        model_path = run_dir / MODEL_FILE_NAME
        manifest_path = run_dir / MANIFEST_FILE_NAME
        metrics_path = run_dir / METRICS_FILE_NAME
        predictions_path = run_dir / PREDICTIONS_FILE_NAME
        if any(path.exists() for path in (model_path, manifest_path, metrics_path, predictions_path)):
            raise FileExistsError(
                f"Run already exists for model_run_id='{model_run_id}' at {run_dir}. "
                "Use a unique model_run_id or delete the existing run before saving."
            )

        self._write_pickle(model_path, model_artifact)
        self._write_json(manifest_path, manifest)
        self._write_json(metrics_path, metrics)

        if predictions is not None:
            self.save_predictions(run_dir=str(run_dir), predictions=predictions)

        return str(run_dir)

    def load_run(self, *, artifact_root: str, model_run_id: str) -> Mapping[str, Any]:
        run_dir = self._resolve_run_dir(artifact_root=artifact_root, model_run_id=model_run_id)
        model_path = run_dir / MODEL_FILE_NAME
        manifest_path = run_dir / MANIFEST_FILE_NAME
        metrics_path = run_dir / METRICS_FILE_NAME
        predictions_path = run_dir / PREDICTIONS_FILE_NAME

        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist for model_run_id='{model_run_id}': {run_dir}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact for model_run_id='{model_run_id}': {model_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest for model_run_id='{model_run_id}': {manifest_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for model_run_id='{model_run_id}': {metrics_path}")

        payload: dict[str, Any] = {
            "model_run_id": model_run_id,
            "run_dir": str(run_dir),
            "model_artifact": self._read_pickle(model_path, artifact_root=artifact_root),
            "manifest": self._read_json(manifest_path),
            "metrics": self._read_json(metrics_path),
        }

        if predictions_path.exists():
            try:
                payload["predictions"] = pd.read_csv(predictions_path)
            except pd.errors.EmptyDataError:
                payload["predictions"] = pd.DataFrame()

        return payload

    @staticmethod
    def _resolve_run_dir(*, artifact_root: str, model_run_id: str) -> Path:
        # Validate run_id to prevent path traversal attacks (e.g., "../../../etc/passwd")
        if ".." in model_run_id or "/" in model_run_id or "\\" in model_run_id:
            raise ValueError(f"Invalid model_run_id: contains path separators or traversal sequences: {model_run_id}")

        run_dir = Path(artifact_root) / model_run_id
        resolved = run_dir.resolve()
        artifact_root_resolved = Path(artifact_root).resolve()

        # Ensure resolved path stays under artifact_root
        try:
            resolved.relative_to(artifact_root_resolved)
        except ValueError:
            raise ValueError(f"Resolved run_dir escapes artifact_root: {resolved} not under {artifact_root_resolved}")

        # Return a path anchored under the original artifact_root to preserve portability.
        return run_dir

    @staticmethod
    def _validate_manifest_consistency(*, manifest: Mapping[str, Any], model_run_id: str, run_dir: Path) -> None:
        manifest_run_id = manifest.get("run_id")
        if manifest_run_id is not None and manifest_run_id != model_run_id:
            raise ValueError(
                "Inconsistent run identifiers: "
                f"model_run_id='{model_run_id}' does not match manifest.run_id='{manifest_run_id}'."
            )

        artifact_paths = manifest.get("artifact_paths")
        if isinstance(artifact_paths, MappingABC):
            manifest_run_dir = artifact_paths.get("run_dir")
            if manifest_run_dir is not None and Path(manifest_run_dir).resolve() != run_dir.resolve():
                raise ValueError(
                    "Inconsistent run directories: "
                    f"resolved run_dir='{run_dir}' does not match manifest.artifact_paths.run_dir='{manifest_run_dir}'."
                )

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file_obj:
            loaded = json.load(file_obj)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected JSON object at {path}.")
        return loaded

    @staticmethod
    def _write_pickle(path: Path, payload: object) -> None:
        # Note: Pickles are not safe from untrusted sources; only load from known, trusted storage.
        with path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)

    @staticmethod
    def _read_pickle(path: Path, artifact_root: str = "") -> object:
        # Security: Only load pickles from paths under artifact_root (trusted storage).
        # Pickle can execute arbitrary code; ensure this is only called with trusted artifacts.
        if artifact_root:
            resolved = path.resolve()
            artifact_root_resolved = Path(artifact_root).resolve()
            try:
                resolved.relative_to(artifact_root_resolved)
            except ValueError:
                raise ValueError(f"Refusing to load pickle outside artifact_root: {resolved}")

        with path.open("rb") as file_obj:
            return pickle.load(file_obj)

    @staticmethod
    def _normalize_rows(predictions: object) -> list[dict[str, Any]]:
        if not isinstance(predictions, list):
            raise TypeError("predictions must be a pandas DataFrame or list of row mappings.")

        normalized: list[dict[str, Any]] = []
        for row in predictions:
            if not isinstance(row, MappingABC):
                raise TypeError("Each prediction row must be a mapping.")
            normalized.append(dict(row))
        return normalized

