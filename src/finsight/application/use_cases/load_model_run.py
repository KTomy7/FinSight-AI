from __future__ import annotations

from finsight.application import dto as application_dto
from finsight.domain.ports import ModelRegistryPort


class LoadModelRun:
    def __init__(self, model_registry: ModelRegistryPort, artifact_root: str) -> None:
        self._model_registry = model_registry
        self._artifact_root = artifact_root

    def execute(self, request: application_dto.LoadModelRunRequest) -> application_dto.LoadModelRunResult:
        loaded = self._model_registry.load_run(
            artifact_root=self._artifact_root,
            model_run_id=request.model_run_id,
        )

        manifest = loaded.get("manifest")
        metrics = loaded.get("metrics")
        model_artifact = loaded.get("model_artifact")

        if not isinstance(manifest, dict):
            raise TypeError("Loaded run manifest must be a dictionary.")
        if not isinstance(metrics, dict):
            raise TypeError("Loaded run metrics must be a dictionary.")

        return application_dto.LoadModelRunResult(
            model_run_id=str(loaded.get("model_run_id", request.model_run_id)),
            run_dir=str(loaded.get("run_dir", "")),
            model_artifact=model_artifact,
            manifest=manifest,
            metrics={str(key): value for key, value in metrics.items()},
            predictions=loaded.get("predictions"),
        )

