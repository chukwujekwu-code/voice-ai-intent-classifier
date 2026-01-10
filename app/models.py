from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request model for single intent prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The utterance text to classify",
        examples=["Ni izihe documents nkeneye kuri business registration?"]
    )
    include_all_probabilities: bool = Field(
        default=False,
        description="Whether to include probabilities for all intents"
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom confidence threshold (0-1). Below this marks low confidence."
    )


class PredictBatchRequest(BaseModel):
    """Request model for batch intent prediction."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of utterance texts to classify"
    )
    include_all_probabilities: bool = Field(
        default=False,
        description="Whether to include probabilities for all intents"
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom confidence threshold"
    )


class IntentPrediction(BaseModel):
    """Single intent prediction result."""

    intent: str = Field(description="Predicted intent class")
    confidence: float = Field(description="Confidence score (0-1)")
    is_low_confidence: bool = Field(
        description="True if confidence is below threshold (may need human review)"
    )
    all_probabilities: Optional[dict[str, float]] = Field(
        default=None,
        description="Probabilities for all intent classes"
    )


class PredictResponse(BaseModel):
    """Response model for single prediction."""

    success: bool = True
    prediction: IntentPrediction
    input_text: str


class PredictBatchResponse(BaseModel):
    """Response model for batch predictions."""

    success: bool = True
    predictions: list[IntentPrediction]
    count: int


class TopKRequest(BaseModel):
    """Request model for top-k intents."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The utterance text to classify"
    )
    k: int = Field(
        default=3,
        ge=1,
        le=13,
        description="Number of top intents to return"
    )


class TopKIntent(BaseModel):
    """Single intent with probability."""

    intent: str
    probability: float


class TopKResponse(BaseModel):
    """Response model for top-k intents."""

    success: bool = True
    input_text: str
    top_intents: list[TopKIntent]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool
    embedding_model: str
    num_classes: int
    version: str


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = False
    error: str
    detail: Optional[str] = None
