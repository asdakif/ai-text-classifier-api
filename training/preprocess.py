from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from app.services.preprocessing import build_vocab, vectorize_text


REQUIRED_COLUMNS = {"text", "label"}


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset must contain columns {sorted(REQUIRED_COLUMNS)}. Missing: {sorted(missing_columns)}"
        )

    dataset = dataset.dropna(subset=["text", "label"]).copy()
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset["label"] = dataset["label"].astype(str).str.strip()
    dataset = dataset[(dataset["text"] != "") & (dataset["label"] != "")]

    if dataset.empty:
        raise ValueError("Dataset is empty after preprocessing.")

    return dataset


def build_label_index(labels: list[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted(set(labels)))}


def texts_to_tensor(texts: list[str], vocab: dict[str, int], max_length: int) -> torch.Tensor:
    encoded_texts = [vectorize_text(text=text, vocab=vocab, max_length=max_length) for text in texts]
    return torch.tensor(encoded_texts, dtype=torch.float32)


def labels_to_tensor(labels: list[str], label_to_index: dict[str, int]) -> torch.Tensor:
    indices = [label_to_index[label] for label in labels]
    return torch.tensor(indices, dtype=torch.long)


__all__ = [
    "build_label_index",
    "build_vocab",
    "labels_to_tensor",
    "load_dataset",
    "texts_to_tensor",
]
