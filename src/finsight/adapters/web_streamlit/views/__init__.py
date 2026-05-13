from .home import render as render_landing
from .predict import render as render_predictor
from .compare import render as render_comparison
from .layout import render_sidebar as render_layout
from .train_backtest import render as render_train_backtest

# Page handlers mapping
PAGE_HANDLERS = {
    "Home": render_landing,
    "Predict": render_predictor,
    "Compare Models": render_comparison,
    "Train & Backtest": render_train_backtest,
}

