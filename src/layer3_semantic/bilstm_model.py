"""
Layer 3 — Bidirectional LSTM (BiLSTM) Model

PyTorch BiLSTM classifier for phishing detection operating on DistilBERT
CLS embeddings (768-dim). The CLS vector is treated as a single-timestep
sequence so the BiLSTM's concatenated forward/backward hidden states (256-dim)
feed into a 3-class linear output head.

Architecture:
    Input:  (batch, 1, 768)  — CLS embedding unsqueezed as 1-step sequence
    BiLSTM: hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.3
    Output: (batch, 3)       — logits for [legitimate, phishing_human, phishing_ai]

Training:
    - Adam optimizer, lr=1e-3
    - CrossEntropyLoss
    - 20 epochs, batch_size=64
    - Best model saved by validation accuracy

Dependencies: torch, sklearn (for SMOTE-compatible label encoding)
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import (
    BILSTM_BATCH_SIZE,
    BILSTM_DROPOUT,
    BILSTM_EPOCHS,
    BILSTM_HIDDEN_DIM,
    BILSTM_LEARNING_RATE,
    BILSTM_NUM_LAYERS,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)

_DL_DIR = MODELS_DIR / "dl"

CLASS_NAMES = ["legitimate", "phishing_human", "phishing_ai"]


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier on top of DistilBERT CLS embeddings.

    The 768-dim CLS embedding is treated as a 1-step sequence. The BiLSTM
    extracts a 256-dim representation (128 forward + 128 backward hidden
    states) which is passed through dropout and a linear output head.

    Args:
        input_dim: Dimensionality of input embeddings (default 768).
        hidden_dim: LSTM hidden state size per direction.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability (applied between layers and before FC).
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = BILSTM_HIDDEN_DIM,
        num_layers: int = BILSTM_NUM_LAYERS,
        dropout: float = BILSTM_DROPOUT,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Embedding tensor of shape (batch, seq_len, input_dim).
               Pass CLS embeddings unsqueezed to (batch, 1, 768).

        Returns:
            Logit tensor of shape (batch, num_classes).
        """
        _, (hidden, _) = self.lstm(x)
        # Concatenate final forward (layer -2) and backward (layer -1) hidden states
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out))


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_bilstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_to_idx: dict[str, int],
    device: torch.device | None = None,
    epochs: int = BILSTM_EPOCHS,
    batch_size: int = BILSTM_BATCH_SIZE,
    lr: float = BILSTM_LEARNING_RATE,
    save_path: Path | None = None,
) -> tuple["BiLSTMClassifier", list[dict]]:
    """Train the BiLSTM model with early saving on best validation accuracy.

    Args:
        X_train: Training embeddings of shape (n_train, 768).
        y_train: Training labels (string array).
        X_val: Validation embeddings of shape (n_val, 768).
        y_val: Validation labels (string array).
        label_to_idx: Mapping from class name to integer index.
        device: Torch device. Auto-detected if None.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        save_path: Path to save the best model .pt file.

    Returns:
        Tuple of (best_model, history) where history is a list of per-epoch
        dicts with keys: epoch, train_loss, val_acc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encode labels
    y_tr_enc = np.array([label_to_idx[l] for l in y_train], dtype=np.int64)
    y_vl_enc = np.array([label_to_idx[l] for l in y_val],   dtype=np.int64)

    # Tensors: unsqueeze embeddings to (batch, 1, 768) for LSTM
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_vl_t = torch.tensor(X_val,   dtype=torch.float32).unsqueeze(1)
    y_tr_t = torch.tensor(y_tr_enc, dtype=torch.long)
    y_vl_t = torch.tensor(y_vl_enc, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )

    model = BiLSTMClassifier(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    history: list[dict] = []

    dest = Path(save_path) if save_path else (_DL_DIR / "bilstm_best.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            # Process validation in batches to avoid OOM
            preds_list = []
            for i in range(0, len(X_vl_t), 256):
                Xb = X_vl_t[i : i + 256].to(device)
                logits = model(Xb)
                preds_list.append(logits.argmax(dim=1).cpu())
            val_preds = torch.cat(preds_list)
            val_acc = (val_preds == y_vl_t).float().mean().item()

        history.append({"epoch": epoch, "train_loss": avg_loss, "val_acc": val_acc})
        logger.info(
            "Epoch %2d/%d  loss=%.4f  val_acc=%.4f%s",
            epoch, epochs, avg_loss, val_acc,
            "  *** best ***" if val_acc > best_val_acc else "",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, dest)

    # Reload best weights
    model.load_state_dict(torch.load(dest, map_location="cpu", weights_only=True))
    logger.info("Best validation accuracy: %.4f — saved to %s", best_val_acc, dest)
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict_bilstm(
    model: BiLSTMClassifier,
    X: np.ndarray,
    idx_to_label: dict[int, str],
    device: torch.device | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on embedding array.

    Args:
        model: Fitted BiLSTMClassifier.
        X: Embedding array of shape (n_samples, 768).
        idx_to_label: Mapping from integer index to class name string.
        device: Torch device. Auto-detected if None.
        batch_size: Inference batch size.

    Returns:
        Tuple of (y_pred_labels, y_proba) where y_proba is (n_samples, n_classes).
    """
    import torch.nn.functional as F

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    all_proba, all_pred = [], []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            Xb = X_t[i : i + batch_size].to(device)
            logits = model(Xb)
            proba = F.softmax(logits, dim=1).cpu().numpy()
            pred  = proba.argmax(axis=1)
            all_proba.append(proba)
            all_pred.append(pred)

    y_proba = np.vstack(all_proba)
    y_pred_idx = np.concatenate(all_pred)
    y_pred_labels = np.array([idx_to_label[i] for i in y_pred_idx])
    return y_pred_labels, y_proba


def load_bilstm(path: Path | None = None, input_dim: int = 768) -> BiLSTMClassifier:
    """Load a saved BiLSTM model from disk.

    Args:
        path: Path to the .pt state dict file.
        input_dim: Input embedding dimensionality.

    Returns:
        BiLSTMClassifier with loaded weights.
    """
    src = Path(path) if path else (_DL_DIR / "bilstm_best.pt")
    model = BiLSTMClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(src, map_location="cpu", weights_only=True))
    logger.info("Loaded BiLSTM from %s", src)
    return model
