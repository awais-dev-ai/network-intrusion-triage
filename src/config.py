"""Configuration loader and validator for the project."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration file.

    Args:
        config_path: Optional override path to a YAML config file.

    Returns:
        Parsed configuration dictionary with resolved project paths.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required top-level configuration sections are missing.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must parse to a dictionary.")

    required_sections = {"project", "paths", "sampling", "schema", "models", "evaluation"}
    missing = required_sections - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config sections: {sorted(missing)}")

    return config


def resolve_path(relative_or_abs: str | Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    path = Path(relative_or_abs)
    return path if path.is_absolute() else PROJECT_ROOT / path
