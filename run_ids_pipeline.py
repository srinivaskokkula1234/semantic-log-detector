"""
CIC-IDS2017 Intrusion Detection Training Pipeline — Entry Point.

Usage:
    # Run binary classification with structured model (default)
    python run_ids_pipeline.py

    # Run specific task
    python run_ids_pipeline.py --task binary
    python run_ids_pipeline.py --task multiclass
    python run_ids_pipeline.py --task both

    # Run with BERT semantic model
    python run_ids_pipeline.py --model semantic --task binary

    # Use custom config
    python run_ids_pipeline.py --config config/ids_config.yaml --task binary --model structured
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ids.pipeline import main

if __name__ == "__main__":
    main()
