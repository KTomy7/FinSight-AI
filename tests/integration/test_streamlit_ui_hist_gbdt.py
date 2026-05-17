"""Verify Streamlit UI views automatically display hist_gbdt from config."""
from __future__ import annotations

from finsight.config.settings import get_settings


def test_streamlit_views_will_show_hist_gbdt_in_training_models() -> None:
    """Verify hist_gbdt appears in training model dropdown (train_model.py view)."""
    settings = get_settings()
    training_model_ids = list(settings.model_defaults.training_model_ids())

    # Verify hist_gbdt will appear in the multiselect dropdown
    assert "hist_gbdt" in training_model_ids
    assert len(training_model_ids) == 5  # naive_zero, naive_mean, ridge, hist_gbdt, xgboost


def test_streamlit_views_will_show_hist_gbdt_in_prediction_models() -> None:
    """Verify hist_gbdt appears in prediction model dropdown (predict.py view)."""
    settings = get_settings()
    prediction_model_ids = list(settings.model_defaults.prediction_model_ids())

    # Verify hist_gbdt will appear in the selectbox dropdown
    assert "hist_gbdt" in prediction_model_ids
    assert len(prediction_model_ids) == 5  # naive_zero, naive_mean, ridge, hist_gbdt, xgboost


def test_streamlit_views_will_show_correct_label_for_hist_gbdt() -> None:
    """Verify hist_gbdt label is correctly displayed in UI selectors."""
    settings = get_settings()
    id_to_label = settings.model_defaults.id_to_label()

    # Verify the label formatter will show correct text
    assert id_to_label["hist_gbdt"] == "Histogram Gradient Boosting"

    # Verify label_to_id mapping also works (for both views)
    label_to_id = settings.model_defaults.label_to_id()
    assert label_to_id["Histogram Gradient Boosting"] == "hist_gbdt"


def test_train_model_view_will_get_hist_gbdt_models() -> None:
    """
    Simulate what train_model.py does:
    - Get training_model_ids from settings
    - Build id_to_label lookup
    - Format into UI with labels
    """
    settings = get_settings()
    model_defaults = settings.model_defaults

    # This is what train_model.py does:
    training_model_ids = list(model_defaults.training_model_ids())
    model_id_to_label = model_defaults.id_to_label()

    # Verify hist_gbdt is discoverable
    assert "hist_gbdt" in training_model_ids
    assert model_id_to_label["hist_gbdt"] == "Histogram Gradient Boosting"

    # Verify the multiselect would show all 5 models by default
    assert len(training_model_ids) == 5


def test_predict_view_will_get_hist_gbdt_models() -> None:
    """
    Simulate what predict.py does:
    - Get prediction_model_ids from settings
    - Build id_to_label lookup
    - Format into UI with labels
    """
    settings = get_settings()
    model_defaults = settings.model_defaults

    # This is what predict.py does:
    prediction_model_ids = list(model_defaults.prediction_model_ids())
    id_to_label = model_defaults.id_to_label()

    # Verify hist_gbdt is discoverable
    assert "hist_gbdt" in prediction_model_ids
    assert id_to_label["hist_gbdt"] == "Histogram Gradient Boosting"

    # Verify the selectbox would show all 5 models
    assert len(prediction_model_ids) == 5

    # Verify default model_id logic in predict.py still works
    default_model_id = model_defaults.default_model_id
    assert default_model_id == "naive_zero"  # Should stay naive_zero
    assert default_model_id in prediction_model_ids  # default should be selectable

