from __future__ import annotations

from dataclasses import dataclass

import pytest

from finsight.application.dto import CompareModelsRequest, ModelRunArtifacts, RegistrySnapshot
from finsight.application.use_cases.compare_models import CompareModels
from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_R2, METRIC_RMSE
from finsight.domain.ports import ModelRegistryPort, RunRegistryPort


@dataclass
class _StubRegistry(ModelRegistryPort):
    artifacts_by_model_id: dict[str, ModelRunArtifacts]

    def __post_init__(self) -> None:
        self.latest_run_calls: list[tuple[str, str]] = []
        self.load_run_artifact_calls: list[tuple[str, str]] = []

    def create_run_dir(self, *, artifact_root: str, run_id: str) -> str:
        raise NotImplementedError

    def latest_run_id(self, *, artifact_root: str, model_id: str) -> str:
        self.latest_run_calls.append((artifact_root, model_id))
        if model_id not in self.artifacts_by_model_id:
            raise FileNotFoundError(f"No runs found for model_id '{model_id}' under artifact root: {artifact_root}")
        return self.artifacts_by_model_id[model_id].run_id

    def save_model(self, *, run_dir: str, model: object) -> None:
        raise NotImplementedError

    def load_model(self, *, artifact_root: str, run_id: str) -> object:
        raise NotImplementedError

    def save_metrics(self, *, run_dir: str, metrics):
        raise NotImplementedError

    def load_metrics(self, *, artifact_root: str, run_id: str):
        raise NotImplementedError

    def save_manifest(self, *, run_dir: str, manifest):
        raise NotImplementedError

    def load_manifest(self, *, artifact_root: str, run_id: str):
        raise NotImplementedError

    def save_predictions(self, *, run_dir: str, predictions):
        raise NotImplementedError

    def load_run_artifacts(self, *, artifact_root: str, run_id: str) -> ModelRunArtifacts:
        self.load_run_artifact_calls.append((artifact_root, run_id))
        for artifacts in self.artifacts_by_model_id.values():
            if artifacts.run_id == run_id:
                return artifacts
        raise FileNotFoundError(f"No run artifacts found for run_id '{run_id}' under artifact root: {artifact_root}")


@dataclass
class _StubRunRegistry(RunRegistryPort):
    snapshot: RegistrySnapshot | None

    def load_registry(self, *, artifact_root: str):
        return self.snapshot

    def record_completed_run(self, *, artifact_root: str, run_summary):
        raise NotImplementedError


def _make_model_run_artifacts(*, model_id: str, run_id: str, metrics: dict[str, float | int | str]) -> ModelRunArtifacts:
    return ModelRunArtifacts(
        run_id=run_id,
        run_dir=f"artifacts/runs/{run_id}",
        model_path=f"artifacts/runs/{run_id}/model.joblib",
        metrics_path=f"artifacts/runs/{run_id}/metrics.json",
        manifest_path=f"artifacts/runs/{run_id}/manifest.json",
        predictions_path=f"artifacts/runs/{run_id}/predictions.csv",
        model=object(),
        metrics=metrics,
        manifest={"model_id": model_id, "feature_columns": ["f1", "f2"]},
    )


def test_compare_models_ranks_by_multiple_metrics_with_expected_direction() -> None:
    registry = _StubRegistry(
        artifacts_by_model_id={
            "alpha": _make_model_run_artifacts(
                model_id="alpha",
                run_id="2026-04-11T120000Z__alpha",
                metrics={METRIC_MAE: 0.10, METRIC_RMSE: 0.22, METRIC_DIRECTION_ACCURACY: 0.80},
            ),
            "beta": _make_model_run_artifacts(
                model_id="beta",
                run_id="2026-04-12T120000Z__beta",
                metrics={METRIC_MAE: 0.10, METRIC_RMSE: 0.21, METRIC_DIRECTION_ACCURACY: 0.90},
            ),
        }
    )

    result = CompareModels(model_registry=registry).execute(
        CompareModelsRequest(
            model_ids=["alpha", "beta"],
            rank_by=[METRIC_MAE, METRIC_DIRECTION_ACCURACY],
        )
    )

    assert [row.rank for row in result.rows] == [1, 2]
    assert [row.model_id for row in result.rows] == ["beta", "alpha"]
    assert result.rank_by == [METRIC_MAE, METRIC_DIRECTION_ACCURACY]
    assert result.metric_directions == {METRIC_MAE: "asc", METRIC_DIRECTION_ACCURACY: "desc"}
    assert registry.latest_run_calls == [("artifacts/runs", "alpha"), ("artifacts/runs", "beta")]
    assert registry.load_run_artifact_calls == [
        ("artifacts/runs", "2026-04-11T120000Z__alpha"),
        ("artifacts/runs", "2026-04-12T120000Z__beta"),
    ]


def test_compare_models_ranks_by_r2_in_descending_order() -> None:
    registry = _StubRegistry(
        artifacts_by_model_id={
            "alpha": _make_model_run_artifacts(
                model_id="alpha",
                run_id="2026-04-11T120000Z__alpha",
                metrics={METRIC_MAE: 0.10, METRIC_RMSE: 0.22, METRIC_R2: 0.81},
            ),
            "beta": _make_model_run_artifacts(
                model_id="beta",
                run_id="2026-04-12T120000Z__beta",
                metrics={METRIC_MAE: 0.10, METRIC_RMSE: 0.22, METRIC_R2: 0.90},
            ),
        }
    )

    result = CompareModels(model_registry=registry).execute(
        CompareModelsRequest(model_ids=["alpha", "beta"], rank_by=[METRIC_R2])
    )

    assert [row.model_id for row in result.rows] == ["beta", "alpha"]
    assert result.rank_by == [METRIC_R2]
    assert result.metric_directions == {METRIC_R2: "desc"}


def test_compare_models_uses_model_id_as_deterministic_tie_break() -> None:
    registry = _StubRegistry(
        artifacts_by_model_id={
            "beta": _make_model_run_artifacts(
                model_id="beta",
                run_id="2026-04-12T120000Z__beta",
                metrics={METRIC_MAE: 0.10, METRIC_RMSE: 0.21, METRIC_DIRECTION_ACCURACY: 0.80},
            ),
            "alpha": _make_model_run_artifacts(
                model_id="alpha",
                run_id="2026-04-11T120000Z__alpha",
                metrics={METRIC_MAE: 0.10, METRIC_RMSE: 0.21, METRIC_DIRECTION_ACCURACY: 0.80},
            ),
        }
    )

    result = CompareModels(model_registry=registry).execute(
        CompareModelsRequest(
            model_ids=["beta", "alpha"],
            rank_by=[METRIC_MAE, METRIC_RMSE],
        )
    )

    assert [row.model_id for row in result.rows] == ["alpha", "beta"]
    assert [row.run_id for row in result.rows] == [
        "2026-04-11T120000Z__alpha",
        "2026-04-12T120000Z__beta",
    ]
    assert result.rows[0].sort_key[-2:] == ("alpha", "2026-04-11T120000Z__alpha")


def test_compare_models_rejects_duplicate_model_ids() -> None:
    registry = _StubRegistry(artifacts_by_model_id={})

    with pytest.raises(ValueError, match="model_ids must be unique"):
        CompareModels(model_registry=registry).execute(
            CompareModelsRequest(model_ids=["alpha", "alpha"], rank_by=[METRIC_MAE])
        )


def test_compare_models_rejects_missing_rank_metric() -> None:
    registry = _StubRegistry(
        artifacts_by_model_id={
            "alpha": _make_model_run_artifacts(
                model_id="alpha",
                run_id="2026-04-11T120000Z__alpha",
                metrics={METRIC_MAE: 0.10, METRIC_RMSE: 0.22},
            ),
        }
    )

    with pytest.raises(ValueError, match="missing comparison metric"):
        CompareModels(model_registry=registry).execute(
            CompareModelsRequest(model_ids=["alpha"], rank_by=[METRIC_MAE, METRIC_DIRECTION_ACCURACY])
        )


def test_compare_models_uses_registry_best_runs_when_requested() -> None:
    @dataclass
    class _BestRunRegistry(ModelRegistryPort):
        latest_by_model_id: dict[str, str]
        artifacts_by_run_id: dict[str, ModelRunArtifacts]

        def create_run_dir(self, *, artifact_root: str, run_id: str) -> str:
            raise NotImplementedError

        def latest_run_id(self, *, artifact_root: str, model_id: str) -> str:
            return self.latest_by_model_id[model_id]

        def save_model(self, *, run_dir: str, model: object) -> None:
            raise NotImplementedError

        def load_model(self, *, artifact_root: str, run_id: str) -> object:
            raise NotImplementedError

        def save_metrics(self, *, run_dir: str, metrics):
            raise NotImplementedError

        def load_metrics(self, *, artifact_root: str, run_id: str):
            raise NotImplementedError

        def save_manifest(self, *, run_dir: str, manifest):
            raise NotImplementedError

        def load_manifest(self, *, artifact_root: str, run_id: str):
            raise NotImplementedError

        def save_predictions(self, *, run_dir: str, predictions):
            raise NotImplementedError

        def load_run_artifacts(self, *, artifact_root: str, run_id: str) -> ModelRunArtifacts:
            return self.artifacts_by_run_id[run_id]

    model_registry = _BestRunRegistry(
        latest_by_model_id={
            "alpha": "2026-04-11T120000Z__alpha_latest",
            "beta": "2026-04-12T120000Z__beta_latest",
        },
        artifacts_by_run_id={
            "2026-04-11T120000Z__alpha_latest": _make_model_run_artifacts(
                model_id="alpha",
                run_id="2026-04-11T120000Z__alpha_latest",
                metrics={METRIC_MAE: 0.12, METRIC_RMSE: 0.22, METRIC_DIRECTION_ACCURACY: 0.80},
            ),
            "2026-04-11T120000Z__alpha_best": _make_model_run_artifacts(
                model_id="alpha",
                run_id="2026-04-11T120000Z__alpha_best",
                metrics={METRIC_MAE: 0.09, METRIC_RMSE: 0.20, METRIC_DIRECTION_ACCURACY: 0.84},
            ),
            "2026-04-12T120000Z__beta_latest": _make_model_run_artifacts(
                model_id="beta",
                run_id="2026-04-12T120000Z__beta_latest",
                metrics={METRIC_MAE: 0.13, METRIC_RMSE: 0.25, METRIC_DIRECTION_ACCURACY: 0.81},
            ),
            "2026-04-12T120000Z__beta_best": _make_model_run_artifacts(
                model_id="beta",
                run_id="2026-04-12T120000Z__beta_best",
                metrics={METRIC_MAE: 0.10, METRIC_RMSE: 0.20, METRIC_DIRECTION_ACCURACY: 0.90},
            ),
        },
    )
    run_registry = _StubRunRegistry(
        snapshot=RegistrySnapshot(
            updated_at="2026-05-16T12:00:00Z",
            best_by_model={
                "alpha": {
                    "run_id": "2026-04-11T120000Z__alpha_best",
                    "created_at": "2026-04-11T12:00:00Z",
                    "metrics": {METRIC_MAE: 0.09},
                },
                "beta": {
                    "run_id": "2026-04-12T120000Z__beta_best",
                    "created_at": "2026-04-12T12:00:00Z",
                    "metrics": {METRIC_MAE: 0.10},
                },
            },
        )
    )

    result = CompareModels(model_registry=model_registry, run_registry=run_registry).execute(
        CompareModelsRequest(model_ids=["alpha", "beta"], rank_by=[METRIC_MAE], use_best_runs=True)
    )

    assert [row.run_id for row in result.rows] == [
        "2026-04-11T120000Z__alpha_best",
        "2026-04-12T120000Z__beta_best",
    ]


