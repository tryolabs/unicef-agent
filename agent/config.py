from pathlib import Path

import yaml
from logging_config import get_logger
from schemas import Config

logger = get_logger(__name__)


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default path.

    Returns:
        Config: Loaded configuration object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config contains invalid values.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        msg = "Configuration file not found: %s"
        logger.error(msg, config_path)
        raise FileNotFoundError(msg, config_path)

    with config_path.open() as f:
        config_data = yaml.safe_load(f)

    return Config(**config_data)
