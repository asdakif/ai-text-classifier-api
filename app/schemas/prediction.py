from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"text": "I loved this product."}})

    text: str = Field(..., min_length=1, description="Raw text to classify.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text input must not be empty.")
        return cleaned


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the top class.")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across supported labels.",
    )
