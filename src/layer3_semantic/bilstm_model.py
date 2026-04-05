"""
Layer 3 — Bidirectional LSTM (BiLSTM) Model

PyTorch implementation of a BiLSTM sequence classifier for phishing detection.
Operates on token-index sequences derived from a vocabulary built over the
training corpus. Runs in parallel with the Super Learner ensemble.

Dependencies: torch
"""

import logging
from pathlib import Path

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


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for binary phishing classification.

    Args:
        vocab_size: Size of the token vocabulary (including padding index 0).
        embed_dim: Dimensionality of the token embeddings.
        hidden_dim: Hidden state size for each LSTM direction.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between layers.
        num_classes: Number of output classes (default 2).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = BILSTM_HIDDEN_DIM,
        num_layers: int = BILSTM_NUM_LAYERS,
        dropout: float = BILSTM_DROPOUT,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
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
            x: Token index tensor of shape (batch, seq_len).

        Returns:
            Logit tensor of shape (batch, num_classes).
        """
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out))


def train_bilstm(
    model: BiLSTMClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device | None = None,
) -> BiLSTMClassifier:
    """Train the BiLSTM model.

    Args:
        model: Initialized BiLSTMClassifier.
        X_train: Token index tensor of shape (n_samples, seq_len).
        y_train: Label tensor of shape (n_samples,).
        device: Torch device to run training on. Auto-detected if None.

    Returns:
        Trained BiLSTMClassifier.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BILSTM_BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BILSTM_LEARNING_RATE)

    for epoch in range(BILSTM_EPOCHS):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, BILSTM_EPOCHS, total_loss / len(loader))

    return model


def save_bilstm(model: BiLSTMClassifier, name: str = "bilstm") -> Path:
    """Save BiLSTM state dict to disk.

    Args:
        model: Trained BiLSTMClassifier.
        name: Filename stem.

    Returns:
        Path to the saved .pt file.
    """
    _DL_DIR.mkdir(parents=True, exist_ok=True)
    path = _DL_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    logger.info("Saved BiLSTM to %s", path)
    return path


def load_bilstm(model: BiLSTMClassifier, name: str = "bilstm") -> BiLSTMClassifier:
    """Load BiLSTM state dict from disk.

    Args:
        model: Initialized BiLSTMClassifier with matching architecture.
        name: Filename stem.

    Returns:
        Model with loaded weights.
    """
    path = _DL_DIR / f"{name}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    logger.info("Loaded BiLSTM from %s", path)
    return model
