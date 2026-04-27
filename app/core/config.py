from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str, default: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    model_artifact_path: Path
    dataset_path: Path
    max_sequence_length: int
    api_key: str | None
    rate_limit_requests: int
    rate_limit_window_seconds: int
    log_level: str


def get_settings() -> Settings:
    default_model_path = ROOT_DIR / "data" / "artifacts" / "text_classifier.pt"
    default_dataset_path = ROOT_DIR / "data" / "sample_sentiment.csv"

    model_path = _resolve_path(
        os.getenv("MODEL_ARTIFACT_PATH", str(default_model_path)),
        default_model_path,
    )
    dataset_path = _resolve_path(
        os.getenv("DATASET_PATH", str(default_dataset_path)),
        default_dataset_path,
    )

    return Settings(
        app_name="AI Text Classifier API",
        app_version="0.1.0",
        model_artifact_path=model_path,
        dataset_path=dataset_path,
        max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "32")),
        api_key=os.getenv("API_KEY"),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "20")),
        rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
