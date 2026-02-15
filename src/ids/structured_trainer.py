"""
Structured model trainer using LightGBM / XGBoost.

Serves as the performance baseline for CIC-IDS2017 intrusion detection.
"""

import os
import json
import time
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.astype(int), weights))


def _build_lightgbm(params: Dict[str, Any], num_classes: int, task: str, class_weights: Optional[Dict] = None):
    """Create a LightGBM classifier."""
    import lightgbm as lgb

    lgb_params = {k: v for k, v in params.items()}

    if task == "binary":
        lgb_params["objective"] = "binary"
        lgb_params["metric"] = "binary_logloss"
        if class_weights:
            lgb_params["scale_pos_weight"] = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
        return lgb.LGBMClassifier(**lgb_params)
    else:
        lgb_params["objective"] = "multiclass"
        lgb_params["num_class"] = num_classes
        lgb_params["metric"] = "multi_logloss"
        if class_weights:
            lgb_params["class_weight"] = class_weights
        return lgb.LGBMClassifier(**lgb_params)


def _build_xgboost(params: Dict[str, Any], num_classes: int, task: str, class_weights: Optional[Dict] = None):
    """Create an XGBoost classifier."""
    import xgboost as xgb

    xgb_params = {k: v for k, v in params.items()}
    xgb_params["use_label_encoder"] = False

    if task == "binary":
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "logloss"
        if class_weights:
            xgb_params["scale_pos_weight"] = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
        return xgb.XGBClassifier(**xgb_params)
    else:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = num_classes
        xgb_params["eval_metric"] = "mlogloss"
        return xgb.XGBClassifier(**xgb_params)


class StructuredTrainer:
    """
    Trainer for structured (tabular) models on CIC-IDS2017 flow features.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task = config["training"]["task"]
        self.model_type = config["model"]["structured"]["type"]
        self.model_params = config["model"]["structured"]["params"]
        self.scaler = StandardScaler()
        self.model = None
        self.feature_cols = None
        self.training_log = []

    def _prepare_features(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scale numeric features."""
        self.feature_cols = feature_cols

        X_train = self.scaler.fit_transform(train[feature_cols].values.astype(np.float32))
        X_val = self.scaler.transform(val[feature_cols].values.astype(np.float32))
        X_test = self.scaler.transform(test[feature_cols].values.astype(np.float32))

        # Replace any remaining NaN/inf
        for arr in [X_train, X_val, X_test]:
            arr[~np.isfinite(arr)] = 0.0

        return X_train, X_val, X_test

    def _get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Extract labels based on task."""
        if self.task == "binary":
            return df["label_binary"].values
        else:
            return df["label_multiclass_encoded"].values

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """Train the structured model with early stopping."""
        logger.info("=" * 60)
        logger.info("TRAINING STRUCTURED MODEL (%s, task=%s)", self.model_type, self.task)
        logger.info("=" * 60)

        X_train, X_val, _ = self._prepare_features(train_df, val_df, val_df, feature_cols)
        y_train = self._get_labels(train_df)
        y_val = self._get_labels(val_df)

        num_classes = len(np.unique(y_train))
        class_weights = None
        if self.config["training"].get("use_weighted_loss", True):
            class_weights = _get_class_weights(y_train)
            logger.info("Class weights: %s", class_weights)

        # Build model
        if self.model_type == "lightgbm":
            self.model = _build_lightgbm(self.model_params, num_classes, self.task, class_weights)
        else:
            self.model = _build_xgboost(self.model_params, num_classes, self.task, class_weights)

        logger.info("Training with %d samples, validating with %d samples", len(X_train), len(X_val))
        start_time = time.time()

        # Fit with early stopping callbacks
        eval_set = [(X_val, y_val)]

        fit_params = {
            "eval_set": eval_set,
        }

        if self.model_type == "lightgbm":
            import lightgbm as lgb
            fit_params["callbacks"] = [
                lgb.early_stopping(
                    stopping_rounds=self.config["training"]["early_stopping_patience"],
                    verbose=True
                ),
                lgb.log_evaluation(period=50),
            ]
        else:
            fit_params["early_stopping_rounds"] = self.config["training"]["early_stopping_patience"]
            fit_params["verbose"] = 50

        self.model.fit(X_train, y_train, **fit_params)

        train_time = time.time() - start_time
        logger.info("Training completed in %.2f seconds", train_time)

        # Get best iteration
        best_iter = getattr(self.model, "best_iteration_", -1)
        if best_iter == -1:
            best_iter = getattr(self.model, "best_ntree_limit", -1)

        train_log = {
            "model_type": self.model_type,
            "task": self.task,
            "train_time_seconds": round(train_time, 2),
            "best_iteration": best_iter,
            "n_features": len(feature_cols),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()} if class_weights else None,
        }
        self.training_log.append(train_log)

        return train_log

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        X = self.scaler.transform(df[self.feature_cols].values.astype(np.float32))
        X[~np.isfinite(X)] = 0.0
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X = self.scaler.transform(df[self.feature_cols].values.astype(np.float32))
        X[~np.isfinite(X)] = 0.0
        proba = self.model.predict_proba(X)
        if self.task == "binary" and proba.ndim == 2:
            return proba[:, 1]  # Return probability of positive class
        return proba

    def save(self, output_dir: str) -> None:
        """Save model, scaler, and training log."""
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "structured_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        scaler_path = os.path.join(output_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        meta = {
            "model_type": self.model_type,
            "task": self.task,
            "feature_cols": self.feature_cols,
            "training_log": self.training_log,
        }
        with open(os.path.join(output_dir, "structured_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        log_path = os.path.join(output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2, default=str)

        logger.info("Structured model saved to %s", output_dir)

    @classmethod
    def load(cls, output_dir: str, config: Dict[str, Any]) -> "StructuredTrainer":
        """Load a saved structured model."""
        trainer = cls(config)

        with open(os.path.join(output_dir, "structured_model.pkl"), "rb") as f:
            trainer.model = pickle.load(f)

        with open(os.path.join(output_dir, "scaler.pkl"), "rb") as f:
            trainer.scaler = pickle.load(f)

        with open(os.path.join(output_dir, "structured_meta.json"), "r") as f:
            meta = json.load(f)

        trainer.feature_cols = meta["feature_cols"]
        trainer.model_type = meta["model_type"]
        trainer.task = meta["task"]

        logger.info("Structured model loaded from %s", output_dir)
        return trainer
