# Inference Benchmark Summary

## Results

| Metric | batch_size_1 | batch_size_32 |
| --- | --- | --- |
| Batch Size | 1 | 32 |
| Latency/Sample (ms) | 8.2628 | 1.0528 |
| Throughput (flows/sec) | 118.08 | 944.02 |
| Mean Batch Latency (ms) | 8.2628 | 33.6892 |
| P95 Batch Latency (ms) | 10.0761 | 28.1274 |
| P99 Batch Latency (ms) | 12.3502 | 508.7083 |
| Memory RSS (MB) | 461.90 | 463.43 |
| Memory Delta (MB) | 0.03 | 0.00 |

## Notes

- Latency measured using `time.perf_counter()` for wall-clock accuracy.
- Memory measured using `psutil.Process.memory_info().rss`.
- Warmup iterations excluded from measurements.
