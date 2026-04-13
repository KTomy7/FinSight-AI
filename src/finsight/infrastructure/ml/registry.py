from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

import finsight.application.dto as application_dto
from finsight.application.contracts import validate_run_manifest
from finsight.domain.ports import ModelRegistryPort


MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
MANIFEST_FILENAME = "manifest.json"
PREDICTIONS_FILENAME = "predictions.csv"


class LocalFileModelRegistry(ModelRegistryPort):
    def create_run_dir(self, *, artifact_root: str, run_id: str) -> str:
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)

        base_path = root / run_id

        try:
            base_path.mkdir(parents=True, exist_ok=False)
            return str(base_path)
        except FileExistsError:
            pass

        suffix = 1
        while True:
            candidate = Path(f"{base_path}_{suffix}")
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return str(candidate)
            except FileExistsError:
                suffix += 1

    def save_model(self, *, run_dir: str, model: object) -> None:
        joblib.dump(model, Path(run_dir) / MODEL_FILENAME)

    def load_model(self, *, artifact_root: str, run_id: str) -> object:
        model_path = self._artifact_path(
            artifact_root=artifact_root,
            run_id=run_id,
            filename=MODEL_FILENAME,
        )
        return joblib.load(model_path)

    def save_metrics(self, *, run_dir: str, metrics: Mapping[str, float | int | str]) -> None:
        self._write_json(Path(run_dir) / METRICS_FILENAME, metrics)

    def load_metrics(self, *, artifact_root: str, run_id: str) -> Mapping[str, float | int | str]:
        metrics_path = self._artifact_path(
            artifact_root=artifact_root,
            run_id=run_id,
            filename=METRICS_FILENAME,
        )
        payload = self._read_json(metrics_path)
        return payload

    def save_manifest(self, *, run_dir: str, manifest: Mapping[str, Any]) -> None:
        validated_manifest = validate_run_manifest(manifest)
        self._write_json(Path(run_dir) / MANIFEST_FILENAME, validated_manifest)

    def load_manifest(self, *, artifact_root: str, run_id: str) -> Mapping[str, Any]:
        manifest_path = self._artifact_path(
            artifact_root=artifact_root,
            run_id=run_id,
            filename=MANIFEST_FILENAME,
        )
        payload = self._read_json(manifest_path)
        return validate_run_manifest(payload)

    def save_predictions(self, *, run_dir: str, predictions: object) -> None:
        output_path = Path(run_dir) / PREDICTIONS_FILENAME

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

    def load_run_artifacts(self, *, artifact_root: str, run_id: str) -> application_dto.ModelRunArtifacts:
        run_dir = self._artifact_root_path(artifact_root=artifact_root, run_id=run_id)
        self._require_directory(run_dir)

        model_path = run_dir / MODEL_FILENAME
        metrics_path = run_dir / METRICS_FILENAME
        manifest_path = run_dir / MANIFEST_FILENAME
        predictions_path = run_dir / PREDICTIONS_FILENAME

        return application_dto.ModelRunArtifacts(
            run_id=run_dir.name,
            run_dir=str(run_dir),
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            manifest_path=str(manifest_path),
            predictions_path=str(predictions_path),
            model=self.load_model(artifact_root=artifact_root, run_id=run_id),
            metrics=self.load_metrics(artifact_root=artifact_root, run_id=run_id),
            manifest=self.load_manifest(artifact_root=artifact_root, run_id=run_id),
        )

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing model registry artifact: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Model registry artifact must contain a JSON object: {path}")
        return payload

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

    @staticmethod
    def _artifact_root_path(*, artifact_root: str, run_id: str) -> Path:
        return Path(artifact_root) / run_id

    @staticmethod
    def _artifact_path(*, artifact_root: str, run_id: str, filename: str) -> Path:
        return LocalFileModelRegistry._artifact_root_path(artifact_root=artifact_root, run_id=run_id) / filename

    @staticmethod
    def _require_directory(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Model run directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Model run path is not a directory: {path}")

