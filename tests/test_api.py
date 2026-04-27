from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.inference import ModelNotReadyError, PredictionResult


class StubInferenceService:
    def __init__(self, result: PredictionResult | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self._model_info = (
            result.model_info
            if result is not None
            else {
                "model_version": "0.1.0",
                "model_type": "bag-of-words-feedforward",
                "max_sequence_length": 32,
            }
        )

    @property
    def is_loaded(self) -> bool:
        return self._error is None

    def get_model_info(self) -> dict[str, str | int] | None:
        if self._error:
            return None
        return self._model_info

    def load(self) -> None:
        if self._error:
            raise self._error

    def predict(self, text: str) -> PredictionResult:
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result


def build_settings(**overrides: object) -> Settings:
    base_settings = Settings(
        app_name="AI Text Classifier API",
        app_version="0.1.0",
        model_artifact_path=Path.cwd() / "data" / "artifacts" / "text_classifier.pt",
        dataset_path=Path.cwd() / "data" / "sample_sentiment.csv",
        max_sequence_length=32,
        api_key=None,
        rate_limit_requests=20,
        rate_limit_window_seconds=60,
        log_level="INFO",
    )
    return replace(base_settings, **overrides)


def test_predict_returns_classification_result() -> None:
    app = create_app(
        inference_service=StubInferenceService(
            result=PredictionResult(
                label="positive",
                confidence=0.94,
                probabilities={"positive": 0.94, "negative": 0.06},
                model_info={
                    "model_version": "0.1.0",
                    "model_type": "bag-of-words-feedforward",
                    "max_sequence_length": 32,
                },
            )
        ),
        settings=build_settings(),
    )
    client = TestClient(app)

    response = client.post("/predict", json={"text": "I loved this product, it was amazing!"})

    assert response.status_code == 200
    assert response.json() == {
        "label": "positive",
        "confidence": 0.94,
        "probabilities": {"positive": 0.94, "negative": 0.06},
        "model": {
            "model_version": "0.1.0",
            "model_type": "bag-of-words-feedforward",
            "max_sequence_length": 32,
        },
    }
    assert response.headers["X-Request-ID"]
    assert "X-Process-Time-Ms" in response.headers


def test_predict_rejects_blank_input() -> None:
    app = create_app(
        inference_service=StubInferenceService(
            result=PredictionResult(
                label="positive",
                confidence=0.9,
                probabilities={"positive": 0.9, "negative": 0.1},
                model_info={
                    "model_version": "0.1.0",
                    "model_type": "bag-of-words-feedforward",
                    "max_sequence_length": 32,
                },
            )
        ),
        settings=build_settings(),
    )
    client = TestClient(app)

    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
    assert "Text input must not be empty." in response.text


def test_predict_returns_503_when_model_artifact_is_missing() -> None:
    app = create_app(
        inference_service=StubInferenceService(error=ModelNotReadyError("Model missing")),
        settings=build_settings(),
    )
    client = TestClient(app)

    response = client.post("/predict", json={"text": "This should fail gracefully."})

    assert response.status_code == 503
    assert response.json() == {"detail": "Model missing"}


def test_health_returns_model_metadata() -> None:
    app = create_app(
        inference_service=StubInferenceService(
            result=PredictionResult(
                label="positive",
                confidence=0.94,
                probabilities={"positive": 0.94, "negative": 0.06},
                model_info={
                    "model_version": "0.1.0",
                    "model_type": "bag-of-words-feedforward",
                    "max_sequence_length": 32,
                },
            )
        ),
        settings=build_settings(),
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_status": "loaded",
        "model": {
            "model_version": "0.1.0",
            "model_type": "bag-of-words-feedforward",
            "max_sequence_length": 32,
        },
    }


def test_predict_requires_api_key_when_configured() -> None:
    app = create_app(
        inference_service=StubInferenceService(
            result=PredictionResult(
                label="positive",
                confidence=0.94,
                probabilities={"positive": 0.94, "negative": 0.06},
                model_info={
                    "model_version": "0.1.0",
                    "model_type": "bag-of-words-feedforward",
                    "max_sequence_length": 32,
                },
            )
        ),
        settings=build_settings(api_key="secret-key"),
    )
    client = TestClient(app)

    response = client.post("/predict", json={"text": "Protected endpoint test"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key."}


def test_predict_rate_limit_returns_429() -> None:
    app = create_app(
        inference_service=StubInferenceService(
            result=PredictionResult(
                label="positive",
                confidence=0.94,
                probabilities={"positive": 0.94, "negative": 0.06},
                model_info={
                    "model_version": "0.1.0",
                    "model_type": "bag-of-words-feedforward",
                    "max_sequence_length": 32,
                },
            )
        ),
        settings=build_settings(rate_limit_requests=1, rate_limit_window_seconds=60),
    )
    client = TestClient(app)

    first_response = client.post("/predict", json={"text": "First request"})
    second_response = client.post("/predict", json={"text": "Second request"})

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.json() == {"detail": "Rate limit exceeded. Please try again later."}
