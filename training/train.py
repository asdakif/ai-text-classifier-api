from __future__ import annotations

from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import get_settings
from app.models.classifier import TextClassifier
from training.preprocess import (
    build_label_index,
    build_vocab,
    labels_to_tensor,
    load_dataset,
    texts_to_tensor,
)


def evaluate(model: TextClassifier, data_loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.inference_mode():
        for features, labels in data_loader:
            logits = model(features)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples
    return average_loss, accuracy


def train() -> Path:
    settings = get_settings()
    torch.manual_seed(42)
    dataset = load_dataset(settings.dataset_path)

    train_df, validation_df = train_test_split(
        dataset,
        test_size=0.25,
        random_state=42,
        stratify=dataset["label"],
    )

    max_length = settings.max_sequence_length
    vocab = build_vocab(train_df["text"].tolist(), min_freq=1)
    label_to_index = build_label_index(dataset["label"].tolist())
    labels = list(label_to_index.keys())

    x_train = texts_to_tensor(train_df["text"].tolist(), vocab, max_length=max_length)
    y_train = labels_to_tensor(train_df["label"].tolist(), label_to_index)
    x_validation = texts_to_tensor(validation_df["text"].tolist(), vocab, max_length=max_length)
    y_validation = labels_to_tensor(validation_df["label"].tolist(), label_to_index)
    x_full = texts_to_tensor(dataset["text"].tolist(), vocab, max_length=max_length)
    y_full = labels_to_tensor(dataset["label"].tolist(), label_to_index)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8, shuffle=True)
    validation_loader = DataLoader(TensorDataset(x_validation, y_validation), batch_size=8)
    full_loader = DataLoader(TensorDataset(x_full, y_full), batch_size=8, shuffle=True)

    model = TextClassifier(
        input_dim=len(vocab),
        hidden_dim=64,
        num_classes=len(labels),
        dropout=0.2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 30
    best_validation_accuracy = 0.0
    best_state_dict = model.state_dict()
    validation_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for features, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_labels.size(0)
            total_examples += batch_labels.size(0)

        train_loss = running_loss / total_examples
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_fn)

        if validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"- train_loss={train_loss:.4f} "
            f"- val_loss={validation_loss:.4f} "
            f"- val_accuracy={validation_accuracy:.4f}"
        )

    model.load_state_dict(best_state_dict)

    for _ in range(10):
        model.train()
        for features, batch_labels in full_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

    settings.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "labels": labels,
        "max_length": max_length,
        "model_config": {
            "input_dim": len(vocab),
            "hidden_dim": 64,
            "num_classes": len(labels),
            "dropout": 0.2,
        },
        "metadata": {
            "model_version": settings.app_version,
            "model_type": "bag-of-words-feedforward",
            "max_sequence_length": max_length,
        },
        "metrics": {
            "validation_loss": round(validation_loss, 4),
            "validation_accuracy": round(best_validation_accuracy, 4),
        },
    }

    torch.save(checkpoint, settings.model_artifact_path)
    print(f"Saved trained artifact to {settings.model_artifact_path}")
    return settings.model_artifact_path


if __name__ == "__main__":
    train()
