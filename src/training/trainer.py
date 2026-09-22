"""Training loops for baseline, numerical, and fusion models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.metrics import MetricReport, compute_metrics


@dataclass
class TrainResult:
    model: nn.Module
    train_metrics: MetricReport
    val_metrics: MetricReport
    history: list[dict[str, float]]


def _train_epoch(model, loader, optimizer, criterion, device, modality: str) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        labels = batch["label"].to(device)
        if modality == "image":
            logits = model(batch["image"].to(device))
        elif modality == "numerical":
            logits = model(batch["features"].to(device))
        else:
            logits = model(batch["image"].to(device), batch["features"].to(device))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def _predict(model, loader, device, modality: str) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        y = batch["label"].to(device)
        if modality == "image":
            logits = model(batch["image"].to(device))
        elif modality == "numerical":
            logits = model(batch["features"].to(device))
        else:
            logits = model(batch["image"].to(device), batch["features"].to(device))
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(p)
        labels.append(y.cpu().numpy())
    y_true = np.concatenate(labels) if labels else np.array([])
    y_prob = np.concatenate(probs) if probs else np.array([])
    avg_loss = total_loss / max(len(loader.dataset), 1)
    return y_true, y_prob, avg_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    modality: str,
    class_weights: torch.Tensor | None = None,
) -> TrainResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device, modality)
        y_true, y_prob, val_loss = _predict(model, val_loader, device, modality)
        val_metrics = compute_metrics(y_true, y_prob, loss=val_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    y_train, p_train, train_loss = _predict(model, train_loader, device, modality)
    y_val, p_val, val_loss = _predict(model, val_loader, device, modality)
    return TrainResult(
        model=model,
        train_metrics=compute_metrics(y_train, p_train, loss=train_loss),
        val_metrics=compute_metrics(y_val, p_val, loss=val_loss),
        history=history,
    )


def train_numerical_from_arrays(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> TrainResult:
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
    )

    def collate(batch):
        feats = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return {"features": feats, "label": labels}

    train_loader.collate_fn = collate
    val_loader.collate_fn = collate
    return train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        modality="numerical",
        class_weights=class_weights,
    )
