#!/usr/bin/env python3
"""
Enhanced Train Model - Combines semantic embeddings with metadata features.
Refactored to use modular architecture.
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.siem_dataset import SIEMDataLoader
from src.models.hybrid_model import HybridAnomalyPipeline

def main():
    print("=" * 60)
    print("🔍 Hybrid Semantic Log Anomaly Detection")
    print("   Using: Sentence-BERT + Isolation Forest + k-NN")
    print("=" * 60)
    
    # Load dataset
    print("\n📦 Loading Advanced SIEM Dataset...")
    loader = SIEMDataLoader(max_samples=3000)
    loader.load()
    
    stats = loader.get_stats()
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total records: {stats['total_records']}")
    print(f"   Normal logs: {stats['normal_count']}")
    print(f"   Anomaly logs: {stats['anomaly_count']}")
    
    # Get full metadata
    all_metadata = loader.get_metadata()
    
    # Split data
    df = loader.df
    normal_mask = ~df['is_anomaly']
    anomaly_mask = df['is_anomaly']
    
    normal_logs = df.loc[normal_mask, 'description'].tolist()
    normal_meta = [m for m, is_anom in zip(all_metadata, df['is_anomaly']) if not is_anom]
    normal_ids = df.loc[normal_mask, 'event_id'].tolist()
    
    anomaly_logs = df.loc[anomaly_mask, 'description'].tolist()
    anomaly_meta = [m for m, is_anom in zip(all_metadata, df['is_anomaly']) if is_anom]
    
    print(f"\n   Normal logs for training: {len(normal_logs)}")
    print(f"   Anomaly logs for testing: {len(anomaly_logs)}")
    
    # Initialize pipeline
    pipeline = HybridAnomalyPipeline(k_neighbors=5, semantic_weight=0.6)
    
    # Train on normal logs
    pipeline.fit(normal_logs, normal_ids, normal_meta)
    
    # Test on normal logs
    print("\n✅ Testing on normal logs...")
    tp_normal = 0
    test_normal = list(zip(normal_logs[:100], normal_meta[:100]))
    for log, meta in test_normal:
        result = pipeline.detect(log, meta)
        if not result['is_anomaly']:
            tp_normal += 1
    print(f"   True Negatives: {tp_normal}/100 ({tp_normal}%)")
    
    # Test on anomaly logs
    print("\n🚨 Testing on anomaly logs...")
    tp_anomaly = 0
    detected = []
    test_anomaly = list(zip(anomaly_logs[:100], anomaly_meta[:100]))
    for log, meta in test_anomaly:
        result = pipeline.detect(log, meta)
        if result['is_anomaly']:
            tp_anomaly += 1
            detected.append((result, meta))
    print(f"   True Positives: {tp_anomaly}/100 ({tp_anomaly}%)")
    
    # Show samples
    if detected:
        print("\n📋 Sample Detections:")
        for i, (result, meta) in enumerate(detected[:3]):
            print(f"\n   --- Anomaly {i+1} ---")
            print(f"   Severity: {meta.get('severity', 'N/A')}")
            print(f"   Risk Score: {meta.get('risk_score', 'N/A')}")
            print(f"   Event Type: {meta.get('event_type', 'N/A')}")
            print(f"   Detection Score: {result['score']:.2%}")
            print(f"   Log: {result['cleaned_log'][:80]}...")
            if result.get('explanation'):
                print(f"   Explanation: {result['explanation'].summary}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Final Results")
    print("=" * 60)
    accuracy = (tp_normal + tp_anomaly) / 200
    print(f"   True Negative Rate: {tp_normal}%")
    print(f"   True Positive Rate: {tp_anomaly}%")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print("\n✨ Training and evaluation completed!")
    
    # Save model
    pipeline.save("models/siem_model")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
