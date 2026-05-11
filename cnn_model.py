"""Sklearn-compatible 1D CNN over raw 1000-frame OpenFace sequences.

The pipeline takes channel sequences of shape (n_samples, n_channels, n_frames)
and predicts the binary self-esteem label. Hyperparameters are fixed to keep
the comparison fair against the other models in pipeline.py.

Architecture
------------
Conv1d(in_channels, 32, k=5, padding=2) -> BatchNorm -> ReLU -> MaxPool(2)
Conv1d(32, 64, k=5, padding=2)          -> BatchNorm -> ReLU -> MaxPool(2)
Conv1d(64, 64, k=3, padding=1)          -> BatchNorm -> ReLU -> AdaptiveAvgPool1d(1)
Dropout(0.4) -> Linear(64, 2)

Loss
----
CrossEntropyLoss with class-balanced weights, Adam(lr=1e-3, weight_decay=1e-4).
Early stopping on a 15% internal validation split inside .fit().
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _CNNArchitecture(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2, dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around the 1D CNN above.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 47 for the full channel set).
    epochs : int
        Maximum training epochs (early stopping may cut it short).
    batch_size : int
    lr : float
    dropout : float
    patience : int
        Stop if validation loss doesn't improve for this many epochs.
    device : str | None
        'cpu' | 'cuda' | None (auto-detect).
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        in_channels: int = 47,
        epochs: int = 60,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.4,
        patience: int = 8,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.in_channels = in_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNN1DClassifier":
        """X shape: (n_samples, n_channels, n_frames). y shape: (n_samples,)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (n,c,t); got {X.shape}")
        if X.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {X.shape[1]}")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        device = self._resolve_device()
        self.device_ = device

        # Per-channel z-score normalization fitted on training data
        self.feature_mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.feature_std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        X_norm = (X - self.feature_mean_) / self.feature_std_

        # Internal val split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_norm, y, test_size=0.15, stratify=y, random_state=self.random_state
        )

        class_counts = np.bincount(y_tr, minlength=n_classes)
        class_weight = torch.tensor(
            (y_tr.shape[0] / (n_classes * class_counts)).astype(np.float32), device=device
        )

        model = _CNNArchitecture(self.in_channels, n_classes=n_classes, dropout=self.dropout).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)

        tr_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).long().to(device)

        best_val_loss = float("inf")
        best_state = None
        patience_left = self.patience
        for epoch in range(self.epochs):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = loss_fn(val_logits, y_val_t).item()
                val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()

            if self.verbose:
                print(f"  epoch {epoch+1:3d} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if self.verbose:
                        print(f"  early stop at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    def _forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.feature_mean_) / self.feature_std_
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_norm).float().to(self.device_))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
