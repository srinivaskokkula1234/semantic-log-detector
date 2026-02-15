"""
Configuration loader with validation and defaults.
"""

import os
import yaml
import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/ids_config.yaml") -> Dict[str, Any]:
    """Load and validate pipeline configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = _apply_defaults(config)
    _validate_config(config)
    _ensure_directories(config)

    logger.info("Configuration loaded from %s", config_path)
    return config


def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values for missing keys."""
    cfg = copy.deepcopy(config)

    cfg.setdefault("data", {})
    cfg["data"].setdefault("encoding", "latin-1")
    cfg["data"].setdefault("label_column", "Label")
    cfg["data"].setdefault("test_size", 0.15)
    cfg["data"].setdefault("val_size", 0.15)
    cfg["data"].setdefault("random_seed", 42)
    cfg["data"].setdefault("remove_duplicates", True)
    cfg["data"].setdefault("max_samples", None)

    cfg.setdefault("training", {})
    cfg["training"].setdefault("seed", 42)
    cfg["training"].setdefault("task", "binary")
    cfg["training"].setdefault("epochs", 10)
    cfg["training"].setdefault("batch_size", 32)
    cfg["training"].setdefault("use_weighted_loss", True)
    cfg["training"].setdefault("early_stopping_patience", 3)

    cfg.setdefault("model", {})
    cfg["model"].setdefault("active_model", "structured")

    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("raw_dataset_dir", "Datasets/CIC-IDS- 2017")
    cfg["paths"].setdefault("processed_dir", "data/processed")
    cfg["paths"].setdefault("output_dir", "outputs")
    cfg["paths"].setdefault("checkpoints_dir", "outputs/checkpoints")
    cfg["paths"].setdefault("reports_dir", "outputs/reports")
    cfg["paths"].setdefault("metrics_dir", "outputs/metrics")
    cfg["paths"].setdefault("benchmark_dir", "outputs/benchmark")

    return cfg


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values."""
    task = config["training"]["task"]
    if task not in ("binary", "multiclass"):
        raise ValueError(f"Invalid training task: {task}. Must be 'binary' or 'multiclass'.")

    model_type = config["model"]["active_model"]
    if model_type not in ("structured", "semantic"):
        raise ValueError(f"Invalid model type: {model_type}. Must be 'structured' or 'semantic'.")

    seed = config["data"]["random_seed"]
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"random_seed must be a non-negative integer, got {seed}")

    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")


def _ensure_directories(config: Dict[str, Any]) -> None:
    """Create output directories if they don't exist."""
    for key in ["output_dir", "checkpoints_dir", "reports_dir", "metrics_dir",
                 "benchmark_dir", "processed_dir"]:
        dir_path = config["paths"].get(key)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)


def save_config_snapshot(config: Dict[str, Any], output_path: str) -> None:
    """Save a snapshot of the config used for reproducibility."""
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Config snapshot saved to %s", output_path)
