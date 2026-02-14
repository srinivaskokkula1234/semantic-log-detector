import time
import psutil
import statistics
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class PerformanceBenchmarker:
    """
    Evaluates detection performance and system metrics.
    """
    
    def __init__(self):
        self.latencies = []
        
    def evaluate_model(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        """
        Compute classification metrics.
        Args:
            y_true: Ground truth labels (0 for benign, 1 for anomaly)
            y_pred: Predicted labels
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
    def measure_latency(self, func, *args, **kwargs) -> float:
        """Measure execution time of a function in ms."""
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        duration = (end - start) * 1000
        self.latencies.append(duration)
        return duration
        
    def get_runtime_stats(self) -> Dict[str, float]:
        if not self.latencies:
            return {}
            
        return {
            'avg_latency_ms': statistics.mean(self.latencies),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'p99_latency_ms': np.percentile(self.latencies, 99),
            'throughput_eps': 1000 / statistics.mean(self.latencies) if self.latencies else 0
        }
        
    def get_system_metrics(self) -> Dict[str, float]:
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024
        }

if __name__ == "__main__":
    # Example usage / Load test script
    print("Running Benchmark...")
    benchmarker = PerformanceBenchmarker()
    
    # Simulate predictions
    y_true = [0] * 90 + [1] * 10
    y_pred = [0] * 85 + [1] * 5 + [1] * 5 + [0] * 5 # Some errors
    
    metrics = benchmarker.evaluate_model(y_true, y_pred)
    print("Model Metrics:", metrics)
    
    # Simulate latency
    for _ in range(100):
        benchmarker.measure_latency(time.sleep, 0.01)
        
    print("Runtime Stats:", benchmarker.get_runtime_stats())
    print("System Metrics:", benchmarker.get_system_metrics())
