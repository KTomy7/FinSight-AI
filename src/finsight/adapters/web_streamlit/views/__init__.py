from .home import render as render_landing
from .predict import render as render_predictor
from .compare import render as render_comparison
from .backtest import render as render_backtest
from .layout import render_sidebar as render_layout
from .train_model import render as render_train_model
# Backward compatibility alias
from .train_model import render as render_train_backtest

# Page handlers mapping
PAGE_HANDLERS = {
    "Home": render_landing,
    "Predict": render_predictor,
    "Backtest": render_backtest,
    "Train Model": render_train_model,
    "Compare Models": render_comparison,
}

