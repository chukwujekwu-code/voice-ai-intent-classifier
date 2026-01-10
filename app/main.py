from contextlib import asynccontextmanager
from typing import Optional
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import (
    PredictRequest,
    PredictResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    TopKRequest,
    TopKResponse,
    TopKIntent,
    IntentPrediction,
    HealthResponse,
    ErrorResponse,
)
from src.classifier import IntentClassifier
from src.config import settings


# Global classifier instance
classifier: Optional[IntentClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global classifier
    print("Loading intent classifier...")
    classifier = IntentClassifier()
    print(f"Classifier loaded with {len(classifier.classes)} intent classes")
    yield
    print("Shutting down...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=classifier is not None,
        embedding_model=settings.embedding_model,
        num_classes=len(classifier.classes) if classifier else 0,
        version=settings.api_version
    )


@app.get("/intents", tags=["Info"])
async def list_intents():
    """List all available intent classes."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "intents": classifier.classes,
        "count": len(classifier.classes)
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_intent(request: PredictRequest):
    """
    Predict the intent for a single utterance.

    Returns the predicted intent, confidence score, and optionally
    probabilities for all intent classes.

    **Example utterances:**
    - "Ndashaka kureba status ya application yanjye" (Kinyarwanda + English)
    - "What are the requirements for passport?" (English)
    - "Nashaka gufata rendez-vous" (Kinyarwanda + French)
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = classifier.predict(
        text=request.text,
        threshold=request.confidence_threshold
    )

    prediction = IntentPrediction(
        intent=result.intent,
        confidence=result.confidence,
        is_low_confidence=result.is_low_confidence,
        all_probabilities=result.all_probabilities if request.include_all_probabilities else None
    )

    return PredictResponse(
        prediction=prediction,
        input_text=request.text
    )


@app.post("/predict/batch", response_model=PredictBatchResponse, tags=["Prediction"])
async def predict_intent_batch(request: PredictBatchRequest):
    """
    Predict intents for multiple utterances in a single request.

    More efficient than calling /predict multiple times.
    Maximum 100 utterances per request.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 texts per batch request"
        )

    results = classifier.predict_batch(
        texts=request.texts,
        threshold=request.confidence_threshold
    )

    predictions = [
        IntentPrediction(
            intent=r.intent,
            confidence=r.confidence,
            is_low_confidence=r.is_low_confidence,
            all_probabilities=r.all_probabilities if request.include_all_probabilities else None
        )
        for r in results
    ]

    return PredictBatchResponse(
        predictions=predictions,
        count=len(predictions)
    )


@app.post("/predict/top-k", response_model=TopKResponse, tags=["Prediction"])
async def predict_top_k_intents(request: TopKRequest):
    """
    Get the top-k most likely intents for an utterance.

    Useful for:
    - Displaying alternative suggestions to users
    - Confidence-based routing/escalation
    - Debugging model behavior
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    top_intents = classifier.get_top_k_intents(
        text=request.text,
        k=request.k
    )

    return TopKResponse(
        input_text=request.text,
        top_intents=[
            TopKIntent(intent=intent, probability=prob)
            for intent, prob in top_intents
        ]
    )


@app.post("/predict/with-fallback", tags=["Prediction"])
async def predict_with_fallback(request: PredictRequest):
    """
    Predict intent with automatic fallback logic.

    If confidence is below threshold:
    - Flags for human review
    - Suggests top alternatives
    - Recommends escalation action

    This is the recommended endpoint for production use.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    threshold = request.confidence_threshold or settings.confidence_threshold
    result = classifier.predict(text=request.text, threshold=threshold)

    response = {
        "input_text": request.text,
        "prediction": {
            "intent": result.intent,
            "confidence": result.confidence,
        },
        "requires_review": result.is_low_confidence,
        "action": "proceed" if not result.is_low_confidence else "escalate_to_human"
    }

    # Add alternatives if low confidence
    if result.is_low_confidence:
        top_intents = classifier.get_top_k_intents(request.text, k=3)
        response["alternatives"] = [
            {"intent": intent, "probability": prob}
            for intent, prob in top_intents
        ]
        response["message"] = (
            f"Low confidence ({result.confidence:.2%}). "
            "Consider routing to human agent or asking for clarification."
        )

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
