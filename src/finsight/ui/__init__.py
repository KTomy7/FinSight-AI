from .landing import render as render_landing
from .predictor import render as render_predictor
from .compare_models import render as render_comparison
from .layout import render_sidebar as render_layout

# Page handlers mapping
PAGE_HANDLERS = {
    "Home": render_landing,
    "Predict": render_predictor,
    "Compare Models": render_comparison,
}
