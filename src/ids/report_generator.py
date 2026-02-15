"""
Final Markdown report generator.

Compiles all results into a comprehensive evaluation report.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_report(
    config: Dict[str, Any],
    data_metadata: Dict[str, Any],
    binary_metrics: Optional[Dict[str, Any]] = None,
    multiclass_metrics: Optional[Dict[str, Any]] = None,
    threshold_results: Optional[Dict[str, Any]] = None,
    benchmark_results: Optional[Dict[str, Any]] = None,
    robustness_results: Optional[Dict[str, Any]] = None,
    drift_baseline: Optional[Dict[str, Any]] = None,
    training_log: Optional[List[Dict]] = None,
    output_dir: str = "outputs/reports",
) -> str:
    """Generate a comprehensive Markdown evaluation report."""
    logger.info("=" * 60)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    lines = []

    # ---- Header ----
    lines.extend([
        "# CIC-IDS2017 Intrusion Detection — Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model:** {config['model']['active_model']}  ",
        f"**Task:** {config['training']['task']}  ",
        f"**Seed:** {config['data']['random_seed']}  ",
        "",
        "---",
        "",
    ])

    # ---- 1. Dataset Statistics ----
    lines.extend([
        "## 1. Dataset Statistics",
        "",
    ])

    if data_metadata:
        lines.extend([
            f"- **Total samples:** {data_metadata.get('total_rows', 'N/A')}",
            f"- **Features:** {data_metadata.get('num_features', 'N/A')}",
            f"- **Train samples:** {data_metadata.get('train_rows', 'N/A')}",
            f"- **Validation samples:** {data_metadata.get('val_rows', 'N/A')}",
            f"- **Test samples:** {data_metadata.get('test_rows', 'N/A')}",
            "",
        ])

        # Binary distribution
        binary_dist = data_metadata.get("binary_distribution", {})
        if binary_dist:
            lines.extend([
                "### Binary Label Distribution",
                "",
                "| Split | BENIGN (0) | ATTACK (1) |",
                "| --- | --- | --- |",
            ])
            for split in ["train", "val", "test"]:
                dist = binary_dist.get(split, {})
                lines.append(f"| {split} | {dist.get('0', dist.get(0, 'N/A'))} | {dist.get('1', dist.get(1, 'N/A'))} |")
            lines.append("")

        # Multi-class distribution
        mc_dist = data_metadata.get("multiclass_distribution", {})
        if mc_dist and "train" in mc_dist:
            classes = sorted(mc_dist["train"].keys())
            lines.extend([
                "### Multi-class Label Distribution",
                "",
                "| Class | Train | Val | Test |",
                "| --- | --- | --- | --- |",
            ])
            for cls in classes:
                train_c = mc_dist.get("train", {}).get(cls, 0)
                val_c = mc_dist.get("val", {}).get(cls, 0)
                test_c = mc_dist.get("test", {}).get(cls, 0)
                lines.append(f"| {cls} | {train_c} | {val_c} | {test_c} |")
            lines.append("")

    lines.extend(["---", ""])

    # ---- 2. Binary Classification Results ----
    if binary_metrics:
        lines.extend([
            "## 2. Binary Classification Results",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ])
        key_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "pr_auc", "false_positive_rate",
            "false_negative_rate", "detection_rate",
        ]
        for k in key_metrics:
            v = binary_metrics.get(k)
            if v is not None:
                lines.append(f"| {k.replace('_', ' ').title()} | {v:.4f} |")
        lines.extend([
            "",
            f"- **True Positives:** {binary_metrics.get('true_positives', 'N/A')}",
            f"- **True Negatives:** {binary_metrics.get('true_negatives', 'N/A')}",
            f"- **False Positives:** {binary_metrics.get('false_positives', 'N/A')}",
            f"- **False Negatives:** {binary_metrics.get('false_negatives', 'N/A')}",
            "",
            "---",
            "",
        ])

    # ---- 3. Multi-class Results ----
    if multiclass_metrics:
        lines.extend([
            "## 3. Multi-class Classification Results",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Accuracy | {multiclass_metrics.get('accuracy', 0):.4f} |",
            f"| Macro F1 | {multiclass_metrics.get('macro_f1', 0):.4f} |",
            f"| Weighted F1 | {multiclass_metrics.get('weighted_f1', 0):.4f} |",
            f"| Macro Precision | {multiclass_metrics.get('macro_precision', 0):.4f} |",
            f"| Macro Recall | {multiclass_metrics.get('macro_recall', 0):.4f} |",
            "",
        ])

        # Per-class detection
        per_class = multiclass_metrics.get("per_class_detection", {})
        if per_class:
            lines.extend([
                "### Per-class Detection Rate",
                "",
                "| Class | Total | Detected | Detection Rate |",
                "| --- | --- | --- | --- |",
            ])
            for cls, det in sorted(per_class.items()):
                lines.append(
                    f"| {cls} | {det['total']} | {det['detected']} | {det['detection_rate']:.4f} |"
                )
            lines.extend(["", "---", ""])

    # ---- 4. Threshold Calibration ----
    if threshold_results:
        lines.extend([
            "## 4. Threshold Calibration Results",
            "",
        ])
        default = threshold_results.get("default_threshold", {})
        lines.extend([
            f"**Default threshold (0.5):** F1={default.get('f1_score', 0):.4f}, "
            f"FPR={default.get('fpr', 0):.4f}, FNR={default.get('fnr', 0):.4f}",
            "",
            "### Calibrated Thresholds",
            "",
            "| Method | Threshold | F1 | FPR | FNR | Detection Rate |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for method, data in threshold_results.get("calibration_methods", {}).items():
            m = data["metrics"]
            lines.append(
                f"| {method} | {data['threshold']:.4f} | {m['f1_score']:.4f} | "
                f"{m['fpr']:.4f} | {m['fnr']:.4f} | {m['detection_rate']:.4f} |"
            )
        rec = threshold_results.get("recommended_threshold")
        if rec:
            lines.append(f"\n**Recommended threshold:** {rec:.4f} "
                         f"(method: {threshold_results.get('recommended_method', 'N/A')})")
        lines.extend(["", "---", ""])

    # ---- 5. Inference Benchmark ----
    if benchmark_results:
        lines.extend([
            "## 5. Inference Benchmark",
            "",
            "| Metric | " + " | ".join(benchmark_results.keys()) + " |",
            "| --- | " + " | ".join(["---"] * len(benchmark_results)) + " |",
        ])
        for label, fn in [
            ("Batch Size", lambda r: str(r["batch_size"])),
            ("Latency/Sample (ms)", lambda r: f"{r['latency_per_sample_ms']:.4f}"),
            ("Throughput (flows/sec)", lambda r: f"{r['throughput_flows_per_sec']:.2f}"),
            ("P95 Latency (ms)", lambda r: f"{r['latency_per_batch_ms']['p95']:.4f}"),
            ("Memory RSS (MB)", lambda r: f"{r['resource_usage']['after']['memory_rss_mb']:.2f}"),
        ]:
            vals = [fn(benchmark_results[k]) for k in benchmark_results]
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        lines.extend(["", "---", ""])

    # ---- 6. Robustness ----
    if robustness_results:
        lines.extend([
            "## 6. Robustness Testing",
            "",
            f"**Overall Robustness Score:** {robustness_results.get('overall_robustness_score', 0):.4f}",
            "",
            "| Noise Level | Mean F1 | F1 Degradation (%) | Prediction Stability |",
            "| --- | --- | --- | --- |",
        ])
        for test in robustness_results.get("noise_tests", []):
            s = test["summary"]
            lines.append(
                f"| {test['noise_level']:.4f} | {s['mean_f1']:.4f} | "
                f"{s['f1_degradation_pct']:.2f}% | {s['mean_stability']:.4f} |"
            )
        lines.extend(["", "---", ""])

    # ---- 7. Observations & Recommendations ----
    lines.extend([
        "## 7. Observations",
        "",
    ])

    observations = []
    if binary_metrics:
        f1 = binary_metrics.get("f1_score", 0)
        fpr = binary_metrics.get("false_positive_rate", 0)
        if f1 > 0.95:
            observations.append(f"- **Excellent binary detection:** F1={f1:.4f} indicates strong overall performance.")
        elif f1 > 0.9:
            observations.append(f"- **Good binary detection:** F1={f1:.4f} shows solid performance.")
        else:
            observations.append(f"- **Binary detection needs improvement:** F1={f1:.4f}.")

        if fpr < 0.01:
            observations.append(f"- **Very low false positive rate:** FPR={fpr:.4f} is excellent for production.")
        elif fpr > 0.05:
            observations.append(f"- **High false positive rate:** FPR={fpr:.4f} may cause alert fatigue.")

    if robustness_results:
        score = robustness_results.get("overall_robustness_score", 0)
        if score > 0.95:
            observations.append(f"- **Robust model:** stability score={score:.4f}.")
        else:
            observations.append(f"- **Moderate robustness:** stability score={score:.4f}; consider data augmentation.")

    if not observations:
        observations.append("- N/A")
    lines.extend(observations)

    lines.extend([
        "",
        "## 8. Limitations",
        "",
        "- CIC-IDS2017 is a labeled benchmark; real-world traffic patterns will differ.",
        "- The dataset was captured in a specific network topology and may not generalize to all environments.",
        "- Class imbalance (e.g., Heartbleed, Infiltration) affects per-class performance.",
        "- BERT-based semantic model requires significant compute resources for inference.",
        "- Threshold calibration is optimized on validation data and should be verified on live traffic.",
        "",
        "## 9. Deployment Recommendations",
        "",
        "1. **Start with the structured model** for production deployment due to lower latency.",
        "2. **Use the calibrated threshold** (Youden's J or Max F1) instead of the default 0.5.",
        "3. **Monitor for drift** using the exported baseline statistics.",
        "4. **Set up alerting** with tiered confidence: high-confidence alerts for p>0.9, review queue for 0.5-0.9.",
        "5. **Retrain periodically** as new attack patterns emerge.",
        "6. **Consider ensemble deployment** combining structured and semantic models for critical environments.",
        "7. **A/B test** threshold changes before full rollout.",
        "",
        "---",
        "",
        "*Report generated by the CIC-IDS2017 Intrusion Detection Pipeline.*",
    ])

    report_text = "\n".join(lines)

    report_path = os.path.join(output_dir, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Final report saved to %s", report_path)
    return report_path
