"""
Inference benchmarking for latency, throughput, CPU, and memory.

Supports both batch size 1 and batch size 32 benchmarks.
"""

import os
import gc
import json
import time
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


def _get_resource_usage() -> Dict[str, float]:
    """Get current CPU and memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_rss_mb": mem_info.rss / (1024 * 1024),
        "memory_vms_mb": mem_info.vms / (1024 * 1024),
    }


def benchmark_model(
    trainer,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Benchmark model inference performance.

    Measures latency per sample, throughput, CPU usage, and memory usage
    at both batch_size=1 and batch_size=32.
    """
    logger.info("=" * 60)
    logger.info("INFERENCE BENCHMARKING")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    batch_sizes = config.get("benchmark", {}).get("batch_sizes", [1, 32])
    num_warmup = config.get("benchmark", {}).get("num_warmup", 10)
    num_iterations = config.get("benchmark", {}).get("num_iterations", 100)

    results = {}

    for bs in batch_sizes:
        logger.info("Benchmarking with batch_size=%d ...", bs)
        gc.collect()

        # Prepare test samples
        n_samples = min(bs * num_iterations, len(test_df))
        sample_df = test_df.head(n_samples).copy()

        # Warmup
        logger.info("  Warmup (%d iterations) ...", num_warmup)
        warmup_size = min(bs, len(sample_df))
        for _ in range(min(num_warmup, 3)):
            try:
                _ = trainer.predict(sample_df.head(warmup_size))
            except Exception:
                pass

        # Benchmark
        latencies = []
        resource_usage_before = _get_resource_usage()

        iterations = min(num_iterations, len(sample_df) // bs)
        if iterations < 1:
            iterations = 1

        logger.info("  Running %d iterations ...", iterations)
        total_start = time.perf_counter()

        for i in range(iterations):
            start = i * bs
            end = min(start + bs, len(sample_df))
            batch = sample_df.iloc[start:end]

            t0 = time.perf_counter()
            _ = trainer.predict(batch)
            t1 = time.perf_counter()

            latencies.append(t1 - t0)

        total_time = time.perf_counter() - total_start
        resource_usage_after = _get_resource_usage()

        total_samples = iterations * bs
        latencies_ms = [l * 1000 for l in latencies]

        result = {
            "batch_size": bs,
            "num_iterations": iterations,
            "total_samples": total_samples,
            "total_time_seconds": round(total_time, 4),
            "latency_per_batch_ms": {
                "mean": round(np.mean(latencies_ms), 4),
                "std": round(np.std(latencies_ms), 4),
                "min": round(np.min(latencies_ms), 4),
                "max": round(np.max(latencies_ms), 4),
                "p50": round(np.percentile(latencies_ms, 50), 4),
                "p95": round(np.percentile(latencies_ms, 95), 4),
                "p99": round(np.percentile(latencies_ms, 99), 4),
            },
            "latency_per_sample_ms": round(np.mean(latencies_ms) / bs, 4),
            "throughput_flows_per_sec": round(total_samples / total_time, 2),
            "resource_usage": {
                "before": resource_usage_before,
                "after": resource_usage_after,
                "memory_delta_mb": round(
                    resource_usage_after["memory_rss_mb"] - resource_usage_before["memory_rss_mb"], 2
                ),
            },
        }

        results[f"batch_size_{bs}"] = result

        logger.info("  Batch size %d results:", bs)
        logger.info("    Latency per sample: %.4f ms", result["latency_per_sample_ms"])
        logger.info("    Throughput: %.2f flows/sec", result["throughput_flows_per_sec"])
        logger.info("    Memory (RSS): %.2f MB", resource_usage_after["memory_rss_mb"])

    # Save results
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Benchmark results saved to %s", results_path)

    # Generate summary markdown
    _generate_benchmark_summary(results, output_dir)

    return results


def _generate_benchmark_summary(results: Dict[str, Any], output_dir: str) -> None:
    """Generate a markdown summary of benchmark results."""
    lines = [
        "# Inference Benchmark Summary",
        "",
        "## Results",
        "",
        "| Metric | " + " | ".join(results.keys()) + " |",
        "| --- | " + " | ".join(["---"] * len(results)) + " |",
    ]

    metrics_to_show = [
        ("Batch Size", lambda r: str(r["batch_size"])),
        ("Latency/Sample (ms)", lambda r: f"{r['latency_per_sample_ms']:.4f}"),
        ("Throughput (flows/sec)", lambda r: f"{r['throughput_flows_per_sec']:.2f}"),
        ("Mean Batch Latency (ms)", lambda r: f"{r['latency_per_batch_ms']['mean']:.4f}"),
        ("P95 Batch Latency (ms)", lambda r: f"{r['latency_per_batch_ms']['p95']:.4f}"),
        ("P99 Batch Latency (ms)", lambda r: f"{r['latency_per_batch_ms']['p99']:.4f}"),
        ("Memory RSS (MB)", lambda r: f"{r['resource_usage']['after']['memory_rss_mb']:.2f}"),
        ("Memory Delta (MB)", lambda r: f"{r['resource_usage']['memory_delta_mb']:.2f}"),
    ]

    for name, fn in metrics_to_show:
        values = [fn(results[k]) for k in results]
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Latency measured using `time.perf_counter()` for wall-clock accuracy.",
        "- Memory measured using `psutil.Process.memory_info().rss`.",
        "- Warmup iterations excluded from measurements.",
        "",
    ])

    summary_path = os.path.join(output_dir, "benchmark_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Benchmark summary saved to %s", summary_path)
