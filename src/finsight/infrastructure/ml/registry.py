from __future__ import annotations

import csv
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from finsight.domain.entities import ModelRunRecord
from finsight.domain.ports import ModelRegistryPort

# run_id must start with an alphanumeric character; subsequent characters may be alphanumeric or '.', '_', or '-'.
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FileSystemModelRegistry(ModelRegistryPort):
    """Filesystem-backed registry for training run artifacts."""

    DEFAULT_MODEL_ARTIFACT_FILENAME = "model.artifact"
    METRICS_FILENAME = "metrics.json"
    MANIFEST_FILENAME = "manifest.json"
    PREDICTIONS_FILENAME = "predictions.csv"

    def create_run_dir(self, *, artifact_root: str, run_id: str) -> str:
        root = self._resolve_artifact_root(artifact_root)
        safe_run_id = self._validate_run_id(run_id)

        base_path = root / safe_run_id
        try:
            base_path.mkdir(parents=True, exist_ok=False)
            return str(base_path)
        except FileExistsError:
            pass

        suffix = 1
        while True:
            candidate = root / f"{safe_run_id}_{suffix}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return str(candidate)
            except FileExistsError:
                suffix += 1

    def save_metrics(self, *, run_dir: str, metrics: Mapping[str, float | int | str]) -> None:
        self._write_json(self._resolve_run_dir(run_dir) / self.METRICS_FILENAME, metrics)

    def save_manifest(self, *, run_dir: str, manifest: Mapping[str, Any]) -> None:
        self._write_json(self._resolve_run_dir(run_dir) / self.MANIFEST_FILENAME, manifest)

    def save_predictions(self, *, run_dir: str, predictions: object) -> None:
        output_path = self._resolve_run_dir(run_dir) / self.PREDICTIONS_FILENAME

        if isinstance(predictions, pd.DataFrame):
            predictions.to_csv(output_path, index=False)
            return

        rows = self._normalize_rows(predictions)
        with output_path.open("w", newline="", encoding="utf-8") as file_obj:
            if not rows:
                file_obj.write("")
                return

            # Collect all unique keys in insertion order across all rows.
            fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_model_artifact(
        self,
        *,
        run_dir: str,
        model_artifact: object,
        filename: str = DEFAULT_MODEL_ARTIFACT_FILENAME,
    ) -> str:
        run_path = self._resolve_run_dir(run_dir)
        artifact_name = self._validate_artifact_filename(filename)
        destination = run_path / artifact_name

        if isinstance(model_artifact, (bytes, bytearray, memoryview)):
            destination.write_bytes(bytes(model_artifact))
            return str(destination)

        if isinstance(model_artifact, (str, Path)):
            try:
                source = Path(model_artifact).expanduser().resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Source model artifact not found at '{model_artifact}'."
                ) from exc
            if not source.is_file():
                raise ValueError("model_artifact path must point to a file.")
            shutil.copy2(source, destination)
            return str(destination)

        raise TypeError("model_artifact must be bytes-like or a filesystem path.")

    def load_run(self, *, artifact_root: str, model_run_id: str) -> ModelRunRecord:
        root = self._resolve_artifact_root(artifact_root)
        safe_run_id = self._validate_run_id(model_run_id)
        run_dir = (root / safe_run_id).resolve()

        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Model run '{model_run_id}' was not found under '{root}'.")

        metrics = self._read_json_mapping(run_dir / self.METRICS_FILENAME, context="metrics")
        manifest = self._read_json_mapping(run_dir / self.MANIFEST_FILENAME, context="manifest")

        predictions_path = run_dir / self.PREDICTIONS_FILENAME
        if not predictions_path.exists() or not predictions_path.is_file():
            raise FileNotFoundError(
                f"Predictions artifact is missing for model run '{model_run_id}': '{predictions_path}'."
            )

        model_artifact_path: str | None = None
        candidate_model = run_dir / self.DEFAULT_MODEL_ARTIFACT_FILENAME
        if candidate_model.exists() and candidate_model.is_file():
            model_artifact_path = str(candidate_model)

        return ModelRunRecord(
            model_run_id=safe_run_id,
            run_dir=str(run_dir),
            metrics={str(key): value for key, value in metrics.items()},
            manifest=dict(manifest),
            predictions_path=str(predictions_path),
            model_artifact_path=model_artifact_path,
        )

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string.")

        if run_id != run_id.strip():
            raise ValueError("run_id must not contain leading or trailing whitespace.")

        candidate = run_id
        if not candidate:
            raise ValueError("run_id must be a non-empty string.")
        if len(candidate) > 255:
            raise ValueError("run_id length must be <= 255 characters.")
        if not _RUN_ID_PATTERN.fullmatch(candidate):
            raise ValueError(
                f"run_id '{candidate}' may only contain letters, digits, dot, underscore, and hyphen."
            )
        return candidate

    @staticmethod
    def _validate_artifact_filename(filename: str) -> str:
        candidate = str(filename).strip()
        if not candidate:
            raise ValueError("filename must be a non-empty string.")
        # Reject both separator styles regardless of host OS.
        if "/" in candidate or "\\" in candidate:
            raise ValueError("filename must not include directory separators.")
        path_candidate = Path(candidate)
        if path_candidate.name != candidate:
            raise ValueError("filename must not include directory separators.")
        return candidate

    @staticmethod
    def _resolve_artifact_root(artifact_root: str) -> Path:
        root = Path(artifact_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _resolve_run_dir(run_dir: str) -> Path:
        path = Path(run_dir).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"run_dir does not exist or is not a directory: '{path}'.")
        return path

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{context} artifact is missing: '{path}'.")

        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, Mapping):
            raise TypeError(f"{context} artifact must contain a JSON object, but got {type(raw_payload).__name__}.")
        return dict(raw_payload)

    @staticmethod
    def _normalize_rows(predictions: object) -> list[dict[str, Any]]:
        if not isinstance(predictions, list):
            raise TypeError("predictions must be a pandas DataFrame or list of row mappings.")

        normalized: list[dict[str, Any]] = []
        for row in predictions:
            if not isinstance(row, Mapping):
                raise TypeError("Each prediction row must be a mapping.")
            normalized.append(dict(row))
        return normalized


# Backward-compatible name used across existing imports/tests.
class LocalFileModelRegistry(FileSystemModelRegistry):
    pass

