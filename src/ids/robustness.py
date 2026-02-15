"""
Robustness testing via feature noise injection.

Injects small perturbations into flow features and measures
performance degradation and detection stability.
"""

import os
import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

logger = logging.getLogger(__name__)


def inject_noise(
    df: pd.DataFrame,
    feature_cols: List[str],
    noise_level: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Inject Gaussian noise into numeric features.

    noise_level: standard deviation of noise as a fraction of each feature's std.
    """
    rng = np.random.RandomState(seed)
    noisy_df = df.copy()

    for col in feature_cols:
        if col in noisy_df.columns and pd.api.types.is_numeric_dtype(noisy_df[col]):
            col_std = noisy_df[col].std()
            if col_std > 0 and np.isfinite(col_std):
                noise = rng.normal(0, noise_level * col_std, size=len(noisy_df))
                noisy_df[col] = noisy_df[col] + noise

    return noisy_df


def run_robustness_test(
    trainer,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Test model robustness against feature perturbations.

    Injects noise at multiple levels and measures performance degradation.
    """
    logger.info("=" * 60)
    logger.info("ROBUSTNESS TESTING")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    task = config["training"]["task"]
    noise_levels = config.get("robustness", {}).get("noise_levels", [0.01, 0.05, 0.1])
    num_perturbations = config.get("robustness", {}).get("num_perturbations", 3)
    base_seed = config.get("robustness", {}).get("seed", 42)

    # Get baseline (clean) predictions
    label_col = "label_binary" if task == "binary" else "label_multiclass_encoded"
    y_true = test_df[label_col].values

    clean_preds = trainer.predict(test_df)
    avg_type = "binary" if task == "binary" else "weighted"
    clean_f1 = f1_score(y_true, clean_preds, average=avg_type, zero_division=0)
    clean_acc = accuracy_score(y_true, clean_preds)

    logger.info("Clean baseline: F1=%.4f, Accuracy=%.4f", clean_f1, clean_acc)

    results = {
        "clean_baseline": {
            "f1": round(float(clean_f1), 4),
            "accuracy": round(float(clean_acc), 4),
        },
        "noise_tests": [],
    }

    for noise_level in noise_levels:
        logger.info("Testing noise level: %.4f", noise_level)
        level_results = {
            "noise_level": noise_level,
            "perturbation_runs": [],
        }

        f1_scores = []
        acc_scores = []
        prediction_stability = []

        for run in range(num_perturbations):
            seed = base_seed + run
            noisy_df = inject_noise(test_df, feature_cols, noise_level, seed)

            noisy_preds = trainer.predict(noisy_df)
            noisy_f1 = f1_score(y_true, noisy_preds, average=avg_type, zero_division=0)
            noisy_acc = accuracy_score(y_true, noisy_preds)

            # Prediction stability: % of predictions that stayed the same
            stability = (clean_preds == noisy_preds).mean()

            f1_scores.append(noisy_f1)
            acc_scores.append(noisy_acc)
            prediction_stability.append(stability)

            level_results["perturbation_runs"].append({
                "run": run,
                "seed": seed,
                "f1": round(float(noisy_f1), 4),
                "accuracy": round(float(noisy_acc), 4),
                "prediction_stability": round(float(stability), 4),
            })

        # Summary for this noise level
        mean_f1 = float(np.mean(f1_scores))
        mean_acc = float(np.mean(acc_scores))
        mean_stability = float(np.mean(prediction_stability))

        f1_degradation = (clean_f1 - mean_f1) / clean_f1 * 100 if clean_f1 > 0 else 0
        acc_degradation = (clean_acc - mean_acc) / clean_acc * 100 if clean_acc > 0 else 0

        level_results["summary"] = {
            "mean_f1": round(mean_f1, 4),
            "mean_accuracy": round(mean_acc, 4),
            "mean_stability": round(mean_stability, 4),
            "f1_degradation_pct": round(float(f1_degradation), 2),
            "accuracy_degradation_pct": round(float(acc_degradation), 2),
            "f1_std": round(float(np.std(f1_scores)), 4),
        }

        results["noise_tests"].append(level_results)

        logger.info(
            "  Noise=%.4f: Mean F1=%.4f (deg=%.2f%%), Stability=%.4f",
            noise_level, mean_f1, f1_degradation, mean_stability
        )

    # Overall robustness score (average stability across noise levels)
    all_stabilities = [
        t["summary"]["mean_stability"] for t in results["noise_tests"]
    ]
    results["overall_robustness_score"] = round(float(np.mean(all_stabilities)), 4)

    logger.info("Overall robustness score: %.4f", results["overall_robustness_score"])

    # Save
    report_path = os.path.join(output_dir, "robustness_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Robustness report saved to %s", report_path)

    return results
