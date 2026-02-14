import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class DriftMonitor:
    """Detects concept drift in embedding space."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.embedding_means = []
        self.metric_window: List[float] = []
        
        # Drift Detection State
        self.current_distribution = []
        self.baseline_stats = None
        self.drift_alert = False
        
    def add_sample(self, embedding: np.ndarray):
        """Add new embedding and check for drift."""
        # Calculate embedding magnitude/norm changes
        metric = np.linalg.norm(embedding)
        self.metric_window.append(metric)
        
        if len(self.metric_window) >= self.window_size:
            # Simple statistical drift (distribution shift)
            new_logs = np.array(self.metric_window[-self.window_size:])
            
            # If no baseline, set it
            if self.baseline_stats is None:
                self.baseline_stats = {
                    'mean': np.mean(new_logs),
                    'std': np.std(new_logs)
                }
                return False
                
            # Compare current mean to baseline
            current_mean = np.mean(new_logs)
            z_score = abs(current_mean - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-6)
            
            # Trigger if significant shift
            if z_score > 3.0:
                self.drift_alert = True
                return True
            else:
                self.drift_alert = False
                
        return False
        
    def get_status(self) -> Dict:
        return {
            'drift_alert': self.drift_alert,
            'metric_mean': list(self.metric_window[-10:]) if self.metric_window else [],
            'samples': len(self.metric_window)
        }
