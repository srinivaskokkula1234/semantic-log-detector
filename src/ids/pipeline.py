"""
Master pipeline orchestrator for CIC-IDS2017 intrusion detection.

Runs the complete training and evaluation pipeline:
1. Data loading & validation
2. Model training (structured or semantic)
3. Evaluation with security-critical metrics
4. Threshold calibration
5. Inference benchmarking
6. Robustness testing
7. Drift baseline export
8. Final report generation
"""

import os
import sys
import json
import logging
import argparse
import random
from typing import Any, Dict

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.ids.config_loader import load_config, save_config_snapshot
from src.ids.data_loader import run_data_pipeline
from src.ids.structured_trainer import StructuredTrainer
from src.ids.evaluator import run_full_evaluation
from src.ids.threshold_calibrator import calibrate_threshold
from src.ids.benchmark import benchmark_model
from src.ids.robustness import run_robustness_test
from src.ids.drift_baseline import export_drift_baseline
from src.ids.report_generator import generate_report


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure structured logging."""
    log_level = config.get("logging", {}).get("level", "INFO")
    log_format = config.get("logging", {}).get(
        "format", "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
    )
    log_file = config.get("logging", {}).get("log_file")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    """Set deterministic seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def run_pipeline(config_path: str = "config/ids_config.yaml", task: str = None, model_type: str = None) -> None:
    """Execute the full IDS training and evaluation pipeline."""
    # Load config
    config = load_config(config_path)

    # Override from CLI args
    if task:
        config["training"]["task"] = task
    if model_type:
        config["model"]["active_model"] = model_type

    setup_logging(config)
    logger = logging.getLogger("ids.pipeline")

    seed = config["data"]["random_seed"]
    set_seed(seed)

    logger.info("=" * 70)
    logger.info("  CIC-IDS2017 INTRUSION DETECTION PIPELINE")
    logger.info("  Model: %s | Task: %s | Seed: %d",
                config["model"]["active_model"], config["training"]["task"], seed)
    logger.info("=" * 70)

    # Save config snapshot
    output_dir = config["paths"]["output_dir"]
    save_config_snapshot(config, os.path.join(output_dir, "config_used.yaml"))

    # ================================================================
    # STAGE 1: DATA LOADING & VALIDATION
    # ================================================================
    train_df, val_df, test_df, feature_cols, mc_encoding = run_data_pipeline(config)

    # Load metadata for report
    metadata_path = os.path.join(config["paths"]["processed_dir"], "metadata.json")
    with open(metadata_path, "r") as f:
        data_metadata = json.load(f)

    # ================================================================
    # STAGE 2 & 3: MODEL TRAINING
    # ================================================================
    active_model = config["model"]["active_model"]
    task = config["training"]["task"]

    if active_model == "structured":
        trainer = StructuredTrainer(config)
        train_log = trainer.train(train_df, val_df, feature_cols)
        trainer.save(config["paths"]["checkpoints_dir"])
    else:
        from src.ids.semantic_trainer import SemanticTrainer
        trainer = SemanticTrainer(config)
        train_log = trainer.train(train_df, val_df, feature_cols)
        trainer.save(config["paths"]["checkpoints_dir"])

    # ================================================================
    # STAGE 4: EVALUATION
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 4: EVALUATION")
    logger.info("=" * 60)

    label_col = "label_binary" if task == "binary" else "label_multiclass_encoded"
    y_true = test_df[label_col].values
    y_pred = trainer.predict(test_df)

    try:
        y_proba = trainer.predict_proba(test_df)
    except Exception as e:
        logger.warning("Could not get probabilities: %s", e)
        y_proba = None

    # Determine class names for multi-class
    class_names = None
    if task == "multiclass":
        # Invert the encoding
        class_names = sorted(mc_encoding.keys(), key=lambda k: mc_encoding[k])

    metrics = run_full_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        task=task,
        output_dir=config["paths"]["metrics_dir"],
        prefix=f"{active_model}_{task}",
        class_names=class_names,
    )

    # Store for report
    binary_metrics = metrics if task == "binary" else None
    multiclass_metrics = metrics if task == "multiclass" else None

    # ================================================================
    # STAGE 5: THRESHOLD CALIBRATION (binary only)
    # ================================================================
    threshold_results = None
    if task == "binary" and y_proba is not None:
        threshold_results = calibrate_threshold(
            y_true=y_true,
            y_proba=y_proba,
            config=config,
            output_dir=config["paths"]["metrics_dir"],
        )

    # ================================================================
    # STAGE 6: INFERENCE BENCHMARKING
    # ================================================================
    benchmark_results = benchmark_model(
        trainer=trainer,
        test_df=test_df,
        config=config,
        output_dir=config["paths"]["benchmark_dir"],
    )

    # ================================================================
    # STAGE 7: ROBUSTNESS TESTING
    # ================================================================
    robustness_results = run_robustness_test(
        trainer=trainer,
        test_df=test_df,
        feature_cols=feature_cols,
        config=config,
        output_dir=config["paths"]["metrics_dir"],
    )

    # ================================================================
    # STAGE 8: DRIFT BASELINE EXPORT
    # ================================================================
    drift_baseline = export_drift_baseline(
        train_df=train_df,
        feature_cols=feature_cols,
        config=config,
        output_dir=config["paths"]["metrics_dir"],
        trainer=trainer if active_model == "semantic" else None,
    )

    # ================================================================
    # STAGE 9: FINAL REPORT
    # ================================================================
    report_path = generate_report(
        config=config,
        data_metadata=data_metadata,
        binary_metrics=binary_metrics,
        multiclass_metrics=multiclass_metrics,
        threshold_results=threshold_results,
        benchmark_results=benchmark_results,
        robustness_results=robustness_results,
        drift_baseline=drift_baseline,
        training_log=[train_log],
        output_dir=config["paths"]["reports_dir"],
    )

    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Report: %s", report_path)
    logger.info("  Metrics: %s", config["paths"]["metrics_dir"])
    logger.info("  Checkpoints: %s", config["paths"]["checkpoints_dir"])
    logger.info("=" * 70)


# ------------------------------------------------------------------ #
#  Also run both tasks sequentially if requested                       #
# ------------------------------------------------------------------ #

def run_full_pipeline(config_path: str = "config/ids_config.yaml") -> None:
    """Run pipeline for both binary and multi-class tasks."""
    for task in ["binary", "multiclass"]:
        run_pipeline(config_path=config_path, task=task)


def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2017 Intrusion Detection Pipeline")
    parser.add_argument("--config", type=str, default="config/ids_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--task", type=str, choices=["binary", "multiclass", "both"],
                        default=None, help="Classification task")
    parser.add_argument("--model", type=str, choices=["structured", "semantic"],
                        default=None, help="Model type")
    args = parser.parse_args()

    if args.task == "both":
        run_full_pipeline(args.config)
    else:
        run_pipeline(
            config_path=args.config,
            task=args.task,
            model_type=args.model,
        )


if __name__ == "__main__":
    main()
