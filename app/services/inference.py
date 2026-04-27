from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from app.core.config import Settings, get_settings
from app.models.classifier import TextClassifier
from app.services.preprocessing import vectorize_text


class ModelNotReadyError(RuntimeError):
    """Raised when the trained model artifact is unavailable."""


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_info: dict[str, str | int]


class InferenceService:
    def __init__(self, artifact_path: Path | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.artifact_path = artifact_path or settings.model_artifact_path
        self.device = torch.device("cpu")
        self.model: TextClassifier | None = None
        self.vocab: dict[str, int] = {}
        self.labels: list[str] = []
        self.max_length = settings.max_sequence_length
        self.metadata: dict[str, str | int] = {
            "model_version": settings.app_version,
            "model_type": "bag-of-words-feedforward",
            "max_sequence_length": self.max_length,
        }

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> dict[str, str | int] | None:
        if not self.is_loaded:
            return None
        return self.metadata

    def load(self) -> None:
        if not self.artifact_path.exists():
            raise ModelNotReadyError(
                "Model artifact not found. Run `python -m training.train` before calling /predict."
            )

        checkpoint = torch.load(self.artifact_path, map_location=self.device)
        model_config = checkpoint["model_config"]

        self.vocab = checkpoint["vocab"]
        self.labels = checkpoint["labels"]
        self.max_length = checkpoint.get("max_length", self.max_length)
        self.metadata = checkpoint.get(
            "metadata",
            {
                "model_version": "1.0.0",
                "model_type": "bag-of-words-feedforward",
                "max_sequence_length": self.max_length,
            },
        )

        self.model = TextClassifier(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        if not self.is_loaded:
            self.load()

        assert self.model is not None

        features = vectorize_text(text=text, vocab=self.vocab, max_length=self.max_length)
        input_tensor = torch.tensor([features], dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        probabilities_map = self._build_probability_map(probabilities.tolist())
        top_index = int(torch.argmax(probabilities).item())

        return PredictionResult(
            label=self.labels[top_index],
            confidence=round(float(probabilities[top_index].item()), 4),
            probabilities=probabilities_map,
            model_info=self.metadata,
        )

    def _build_probability_map(self, raw_probabilities: list[float]) -> dict[str, float]:
        return {
            label: round(float(probability), 4)
            for label, probability in zip(self.labels, raw_probabilities)
        }
