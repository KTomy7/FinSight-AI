from __future__ import annotations

from finsight.bootstrap import container as container_module
from finsight.config.settings import Settings
from finsight.application.use_cases.compare_models import CompareModels


def test_build_container_exposes_compare_models_use_case(monkeypatch) -> None:
    container_module.build_container.cache_clear()
    monkeypatch.setattr(container_module, "get_settings", lambda: Settings())

    app_container = container_module.build_container()

    assert isinstance(app_container.compare_models, CompareModels)
    assert app_container.compare_models is app_container.compare_models


def test_build_container_exposes_hist_gbdt_in_router(monkeypatch) -> None:
    """Verify hist_gbdt model adapter is registered in the router."""
    container_module.build_container.cache_clear()
    monkeypatch.setattr(container_module, "get_settings", lambda: Settings())

    app_container = container_module.build_container()

    # Get router from TrainModel use case
    router = app_container.train_model._model
    supported_types = router.supported_model_types()

    assert "hist_gbdt" in supported_types
    assert ("naive_zero", "naive_mean", "ridge", "hist_gbdt") == supported_types


def test_build_container_hist_gbdt_enables_training() -> None:
    """Verify hist_gbdt is training-enabled in config."""
    from finsight.config.settings import get_settings

    container_module.build_container.cache_clear()

    # Load real settings from config file
    settings = get_settings()
    training_model_ids = settings.model_defaults.training_model_ids()
    assert "hist_gbdt" in training_model_ids

