from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.inference import ModelNotReadyError, PredictionResult


class StubInferenceService:
    def __init__(self, result: PredictionResult | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    @property
    def is_loaded(self) -> bool:
        return self._error is None

    def load(self) -> None:
        if self._error:
            raise self._error

    def predict(self, text: str) -> PredictionResult:
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result


def test_predict_returns_classification_result() -> None:
    app = create_app(
        inference_service=StubInferenceService(
            result=PredictionResult(
                label="positive",
                confidence=0.94,
                probabilities={"positive": 0.94, "negative": 0.06},
            )
        )
    )
    client = TestClient(app)

    response = client.post("/predict", json={"text": "I loved this product, it was amazing!"})

    assert response.status_code == 200
    assert response.json() == {
        "label": "positive",
        "confidence": 0.94,
        "probabilities": {"positive": 0.94, "negative": 0.06},
    }


def test_predict_rejects_blank_input() -> None:
    app = create_app(
        inference_service=StubInferenceService(
            result=PredictionResult(
                label="positive",
                confidence=0.9,
                probabilities={"positive": 0.9, "negative": 0.1},
            )
        )
    )
    client = TestClient(app)

    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
    assert "Text input must not be empty." in response.text


def test_predict_returns_503_when_model_artifact_is_missing() -> None:
    app = create_app(inference_service=StubInferenceService(error=ModelNotReadyError("Model missing")))
    client = TestClient(app)

    response = client.post("/predict", json={"text": "This should fail gracefully."})

    assert response.status_code == 503
    assert response.json() == {"detail": "Model missing"}
