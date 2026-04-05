"""
Layer 3 — Character-Level GRU Model

PyTorch implementation of a character-level GRU classifier. Operates directly
on character sequences rather than word tokens, making it robust to obfuscation
techniques such as misspellings, character substitution, and zero-width spaces
commonly used in AI-generated phishing.

Dependencies: torch
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import BILSTM_BATCH_SIZE, BILSTM_DROPOUT, BILSTM_EPOCHS, BILSTM_LEARNING_RATE, MODELS_DIR

logger = logging.getLogger(__name__)

_DL_DIR = MODELS_DIR / "dl"

# Printable ASCII characters (32–126) plus padding index 0
CHAR_VOCAB_SIZE = 128


class CharGRUClassifier(nn.Module):
    """Character-level GRU for binary phishing classification.

    Args:
        vocab_size: Character vocabulary size (default 128 for ASCII).
        embed_dim: Character embedding dimensionality.
        hidden_dim: GRU hidden state size.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability.
        num_classes: Output classes (default 2).
    """

    def __init__(
        self,
        vocab_size: int = CHAR_VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = BILSTM_DROPOUT,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
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
            x: Character index tensor of shape (batch, seq_len).

        Returns:
            Logit tensor of shape (batch, num_classes).
        """
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.gru(embedded)
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out))


def text_to_char_tensor(text: str, max_len: int = 1024) -> torch.Tensor:
    """Convert a text string to a character index tensor.

    Args:
        text: Input string.
        max_len: Maximum sequence length (truncated or zero-padded).

    Returns:
        1-D LongTensor of character indices.
    """
    indices = [min(ord(c), CHAR_VOCAB_SIZE - 1) for c in text[:max_len]]
    # Zero-pad to max_len
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


def train_chargru(
    model: CharGRUClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device | None = None,
) -> CharGRUClassifier:
    """Train the CharGRU model.

    Args:
        model: Initialized CharGRUClassifier.
        X_train: Character index tensor of shape (n_samples, seq_len).
        y_train: Label tensor of shape (n_samples,).
        device: Training device. Auto-detected if None.

    Returns:
        Trained CharGRUClassifier.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BILSTM_BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BILSTM_LEARNING_RATE)

    for epoch in range(BILSTM_EPOCHS):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("CharGRU Epoch %d/%d — loss: %.4f", epoch + 1, BILSTM_EPOCHS, total_loss / len(loader))

    return model


def save_chargru(model: CharGRUClassifier, name: str = "chargru") -> Path:
    """Save CharGRU state dict to disk.

    Args:
        model: Trained CharGRUClassifier.
        name: Filename stem.

    Returns:
        Path to the saved .pt file.
    """
    _DL_DIR.mkdir(parents=True, exist_ok=True)
    path = _DL_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    logger.info("Saved CharGRU to %s", path)
    return path
