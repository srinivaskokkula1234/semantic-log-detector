"""
Drift baseline export.

Computes and saves feature distribution statistics and embedding
distributions for future drift detection.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_feature_statistics(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Compute distribution statistics for each numeric feature."""
    stats = {}

    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()

        if len(series) == 0:
            continue

        values = series.values.astype(float)
        finite_mask = np.isfinite(values)
        values = values[finite_mask]

        if len(values) == 0:
            continue

        stats[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "skewness": float(_safe_skewness(values)),
            "kurtosis": float(_safe_kurtosis(values)),
            "num_zeros": int((values == 0).sum()),
            "num_unique": int(len(np.unique(values))),
            "count": int(len(values)),
        }

    return stats


def _safe_skewness(values: np.ndarray) -> float:
    """Compute skewness safely."""
    try:
        from scipy.stats import skew
        return float(skew(values, nan_policy="omit"))
    except (ImportError, ValueError):
        n = len(values)
        if n < 3:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return float(np.mean(((values - mean) / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    """Compute kurtosis safely."""
    try:
        from scipy.stats import kurtosis
        return float(kurtosis(values, nan_policy="omit"))
    except (ImportError, ValueError):
        n = len(values)
        if n < 4:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return float(np.mean(((values - mean) / std) ** 4) - 3.0)


def compute_embedding_statistics(
    trainer,
    df: pd.DataFrame,
    feature_cols: List[str],
    max_samples: int = 5000,
) -> Optional[Dict[str, Any]]:
    """
    Compute embedding distribution statistics for BERT model.

    Returns None if the trainer is not a semantic model.
    """
    try:
        import torch
        from src.ids.semantic_trainer import batch_flow_to_text, FlowTextDataset
        from torch.utils.data import DataLoader

        if not hasattr(trainer, "tokenizer") or trainer.tokenizer is None:
            return None

        # Sample a subset
        if len(df) > max_samples:
            df_sample = df.sample(n=max_samples, random_state=42)
        else:
            df_sample = df

        texts = batch_flow_to_text(df_sample, feature_cols)
        labels = np.zeros(len(texts), dtype=int)
        dataset = FlowTextDataset(texts, labels, trainer.tokenizer, 256)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        embeddings = []
        trainer.model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(trainer.device)
                attention_mask = batch["attention_mask"].to(trainer.device)
                outputs = trainer.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(outputs.pooler_output.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        stats = {
            "embedding_dim": int(embeddings.shape[1]),
            "num_samples": int(embeddings.shape[0]),
            "per_dim_mean": embeddings.mean(axis=0).tolist(),
            "per_dim_std": embeddings.std(axis=0).tolist(),
            "global_mean": float(embeddings.mean()),
            "global_std": float(embeddings.std()),
            "global_min": float(embeddings.min()),
            "global_max": float(embeddings.max()),
            "norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
            "norm_std": float(np.linalg.norm(embeddings, axis=1).std()),
        }

        logger.info("Computed embedding statistics: dim=%d, samples=%d",
                     stats["embedding_dim"], stats["num_samples"])
        return stats

    except Exception as e:
        logger.warning("Could not compute embedding statistics: %s", e)
        return None


def export_drift_baseline(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    config: Dict[str, Any],
    output_dir: str,
    trainer=None,
) -> Dict[str, Any]:
    """
    Export drift detection baseline.

    Computes feature distribution stats and optionally embedding stats.
    """
    logger.info("=" * 60)
    logger.info("DRIFT BASELINE EXPORT")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Feature distribution statistics
    feature_stats = compute_feature_statistics(train_df, feature_cols)
    logger.info("Computed statistics for %d features", len(feature_stats))

    baseline = {
        "num_features": len(feature_stats),
        "num_training_samples": len(train_df),
        "feature_statistics": feature_stats,
    }

    # Embedding statistics (if semantic model)
    if trainer is not None and config["model"]["active_model"] == "semantic":
        emb_stats = compute_embedding_statistics(trainer, train_df, feature_cols)
        if emb_stats is not None:
            baseline["embedding_statistics"] = emb_stats

    # Save
    baseline_path = os.path.join(output_dir, "drift_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Drift baseline saved to %s", baseline_path)

    return baseline
