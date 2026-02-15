"""
Semantic (BERT-based) model for intrusion detection.

Converts structured CIC-IDS2017 flow records into textual descriptions,
tokenizes them with a BERT tokenizer, and fine-tunes a classifier head.
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Flow-to-Text Conversion                                            #
# ------------------------------------------------------------------ #

# Key features to include in the textual description
TEXT_FEATURES = [
    ("destination_port", "destination port {}"),
    ("flow_duration", "flow duration {} microseconds"),
    ("total_fwd_packets", "{} forward packets"),
    ("total_backward_packets", "{} backward packets"),
    ("total_length_of_fwd_packets", "{} bytes forward"),
    ("total_length_of_bwd_packets", "{} bytes backward"),
    ("fwd_packet_length_max", "max forward packet {} bytes"),
    ("bwd_packet_length_max", "max backward packet {} bytes"),
    ("flow_bytes_s", "{} bytes per second"),
    ("flow_packets_s", "{} packets per second"),
    ("flow_iat_mean", "mean inter-arrival time {} microseconds"),
    ("fwd_iat_total", "total forward IAT {} microseconds"),
    ("bwd_iat_total", "total backward IAT {} microseconds"),
    ("fwd_psh_flags", "{} forward PSH flags"),
    ("syn_flag_count", "{} SYN flags"),
    ("rst_flag_count", "{} RST flags"),
    ("ack_flag_count", "{} ACK flags"),
    ("fin_flag_count", "{} FIN flags"),
    ("subflow_fwd_packets", "{} subflow forward packets"),
    ("subflow_bwd_packets", "{} subflow backward packets"),
    ("init_win_bytes_forward", "initial forward window {} bytes"),
    ("init_win_bytes_backward", "initial backward window {} bytes"),
    ("active_mean", "active mean {} microseconds"),
    ("idle_mean", "idle mean {} microseconds"),
]


def flow_to_text(row: pd.Series, feature_cols: List[str]) -> str:
    """Convert a flow record into a structured textual description."""
    parts = ["Network flow:"]
    row_lower = {k.lower(): v for k, v in row.items()}

    for feat_name, template in TEXT_FEATURES:
        # Try to match feature name
        matched_key = None
        for col in feature_cols:
            if feat_name in col.lower().replace(" ", "_"):
                matched_key = col
                break
        if matched_key is not None and matched_key in row.index:
            val = row[matched_key]
            if pd.notna(val) and np.isfinite(float(val)):
                formatted = f"{float(val):.0f}" if abs(float(val)) > 1 else f"{float(val):.4f}"
                parts.append(template.format(formatted))

    return ", ".join(parts) + "."


def batch_flow_to_text(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """Convert a batch of flow records to text."""
    texts = []
    for _, row in df.iterrows():
        texts.append(flow_to_text(row, feature_cols))
    return texts


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #

class FlowTextDataset(Dataset):
    """PyTorch dataset for tokenized flow text descriptions."""

    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ------------------------------------------------------------------ #
#  BERT Classifier Model                                               #
# ------------------------------------------------------------------ #

class BERTFlowClassifier(nn.Module):
    """BERT-based classifier for network flow classification."""

    def __init__(self, pretrained_model: str, num_classes: int,
                 hidden_dropout: float = 0.2, classifier_hidden_dim: int = 256,
                 freeze_layers: int = 8):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(pretrained_model)

        # Freeze early layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer[:freeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info("Froze first %d BERT layers", freeze_layers)

        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS] token
        logits = self.classifier(pooled)
        return logits


# ------------------------------------------------------------------ #
#  Semantic Trainer                                                    #
# ------------------------------------------------------------------ #

class SemanticTrainer:
    """
    Trainer for BERT-based semantic intrusion detection model.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task = config["training"]["task"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.feature_cols = None
        self.training_log = []

    def _build_model(self, num_classes: int) -> BERTFlowClassifier:
        """Build the BERT classifier."""
        sem_config = self.config["model"]["semantic"]
        model = BERTFlowClassifier(
            pretrained_model=sem_config["pretrained_model"],
            num_classes=num_classes,
            hidden_dropout=sem_config.get("hidden_dropout", 0.2),
            classifier_hidden_dim=sem_config.get("classifier_hidden_dim", 256),
            freeze_layers=sem_config.get("freeze_bert_layers", 8),
        )
        return model.to(self.device)

    def _get_tokenizer(self):
        """Get BERT tokenizer."""
        from transformers import BertTokenizer
        model_name = self.config["model"]["semantic"]["pretrained_model"]
        return BertTokenizer.from_pretrained(model_name)

    def _get_labels(self, df: pd.DataFrame) -> np.ndarray:
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
        """Fine-tune BERT for flow classification."""
        logger.info("=" * 60)
        logger.info("TRAINING SEMANTIC MODEL (BERT, task=%s)", self.task)
        logger.info("=" * 60)
        logger.info("Device: %s", self.device)

        self.feature_cols = feature_cols
        self.tokenizer = self._get_tokenizer()

        # Convert flows to text
        logger.info("Converting flow records to text descriptions...")
        train_texts = batch_flow_to_text(train_df, feature_cols)
        val_texts = batch_flow_to_text(val_df, feature_cols)
        logger.info("Sample text: %s", train_texts[0][:200])

        y_train = self._get_labels(train_df)
        y_val = self._get_labels(val_df)
        num_classes = len(np.unique(y_train))

        # Build datasets
        max_len = self.config["model"]["semantic"].get("max_seq_length", 256)
        train_dataset = FlowTextDataset(train_texts, y_train, self.tokenizer, max_len)
        val_dataset = FlowTextDataset(val_texts, y_val, self.tokenizer, max_len)

        batch_size = self.config["training"]["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Build model
        self.model = self._build_model(num_classes)

        # Loss with class weights
        if self.config["training"].get("use_weighted_loss", True):
            weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info("Using weighted loss: %s", weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer & scheduler
        lr = self.config["training"]["learning_rate"]
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config["training"].get("weight_decay", 0.01),
        )

        total_steps = len(train_loader) * self.config["training"]["epochs"]
        warmup_steps = int(total_steps * self.config["training"].get("warmup_ratio", 0.1))
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Training loop
        epochs = self.config["training"]["epochs"]
        patience = self.config["training"]["early_stopping_patience"]
        grad_clip = self.config["training"].get("gradient_clip_max_norm", 1.0)

        best_val_f1 = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        best_state = None

        start_time = time.time()

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

                if (batch_idx + 1) % 100 == 0:
                    logger.info(
                        "Epoch %d/%d - Batch %d/%d - Loss: %.4f",
                        epoch + 1, epochs, batch_idx + 1, len(train_loader),
                        loss.item()
                    )

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validate
            val_metrics = self._validate(val_loader, criterion)

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_acc": round(val_metrics["accuracy"], 4),
                "val_f1": round(val_metrics["f1"], 4),
                "lr": scheduler.get_last_lr()[0],
            }
            self.training_log.append(epoch_log)

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f | "
                "Val Loss: %.4f, Val Acc: %.4f, Val F1: %.4f",
                epoch + 1, epochs, train_loss, train_acc,
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1"]
            )

            # Early stopping
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch + 1
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("Early stopping at epoch %d (best epoch: %d)", epoch + 1, best_epoch)
                    break

        total_time = time.time() - start_time

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best model from epoch %d (F1=%.4f)", best_epoch, best_val_f1)

        result = {
            "task": self.task,
            "train_time_seconds": round(total_time, 2),
            "best_epoch": best_epoch,
            "best_val_f1": round(best_val_f1, 4),
            "total_epochs": len(self.training_log),
        }
        return result

    def _validate(self, val_loader, criterion) -> Dict[str, float]:
        """Run validation and compute metrics."""
        from sklearn.metrics import f1_score

        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        avg = "binary" if self.task == "binary" else "weighted"
        f1 = f1_score(all_labels, all_preds, average=avg, zero_division=0)

        return {
            "loss": val_loss / len(all_labels),
            "accuracy": (all_preds == all_labels).mean(),
            "f1": f1,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        texts = batch_flow_to_text(df, self.feature_cols)
        max_len = self.config["model"]["semantic"].get("max_seq_length", 256)
        dummy_labels = np.zeros(len(texts), dtype=int)
        dataset = FlowTextDataset(texts, dummy_labels, self.tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=self.config["training"]["batch_size"],
                            shuffle=False, num_workers=0)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        texts = batch_flow_to_text(df, self.feature_cols)
        max_len = self.config["model"]["semantic"].get("max_seq_length", 256)
        dummy_labels = np.zeros(len(texts), dtype=int)
        dataset = FlowTextDataset(texts, dummy_labels, self.tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=self.config["training"]["batch_size"],
                            shuffle=False, num_workers=0)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        if self.task == "binary":
            return probs[:, 1]  # probability of attack class
        return probs

    def save(self, output_dir: str) -> None:
        """Save model, tokenizer, and training log."""
        os.makedirs(output_dir, exist_ok=True)

        # Save model weights
        model_path = os.path.join(output_dir, "semantic_model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save tokenizer
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_dir)

        # Save metadata
        meta = {
            "task": self.task,
            "feature_cols": self.feature_cols,
            "pretrained_model": self.config["model"]["semantic"]["pretrained_model"],
            "training_log": self.training_log,
        }
        with open(os.path.join(output_dir, "semantic_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        log_path = os.path.join(output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2, default=str)

        logger.info("Semantic model saved to %s", output_dir)

    @classmethod
    def load(cls, output_dir: str, config: Dict[str, Any]) -> "SemanticTrainer":
        """Load a saved semantic model."""
        from transformers import BertTokenizer

        trainer = cls(config)

        with open(os.path.join(output_dir, "semantic_meta.json"), "r") as f:
            meta = json.load(f)

        trainer.feature_cols = meta["feature_cols"]
        trainer.task = meta["task"]

        # Load tokenizer
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        trainer.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

        # Build and load model
        num_classes = 2 if trainer.task == "binary" else len(
            config.get("labels", {}).get("multiclass", {})
        )
        # Fallback
        if num_classes < 2:
            num_classes = 2

        trainer.model = trainer._build_model(num_classes)
        model_path = os.path.join(output_dir, "semantic_model.pt")
        trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))

        logger.info("Semantic model loaded from %s", output_dir)
        return trainer
