import yaml
from typing import Dict, Any
from pathlib import Path

def load_config() -> Dict[str, Any]:
    """
    Load configuration settings from a YAML file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the configuration settings.
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_stock_data_settings():
    """
    Get stock data settings from the configuration file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing stock data settings.
    """
    config = load_config()
    return config.get("stock_data", {})

def get_preprocessing_settings():
    """
    Get preprocessing settings from the configuration file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing preprocessing settings.
    """
    config = load_config()
    return config.get("preprocessing", {})
