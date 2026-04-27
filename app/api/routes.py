from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.inference import InferenceService, ModelNotReadyError


router = APIRouter()


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


@router.get("/health", summary="Health check")
def health_check(request: Request) -> dict[str, str]:
    service = request.app.state.inference_service
    model_status = "loaded" if service.is_loaded else "not_loaded"
    return {"status": "ok", "model_status": model_status}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict a class label for input text",
)
def predict(
    payload: PredictionRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    try:
        result = inference_service.predict(payload.text)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal server error.",
        ) from exc

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
    )
