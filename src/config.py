from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    models_dir: Path = base_dir / "models"

    # Model settings
    embedding_model: str = "sentence-transformers/LaBSE"
    classifier_path: str = "labse_logreg_classifier.joblib"
    model_info_path: str = "labse_logreg_info.joblib"

    # API settings
    api_title: str = "Voice AI Intent Classification API"
    api_version: str = "1.0.0"
    api_description: str = "Intent classification for Rwandan government services Voice AI"

    # Inference settings
    confidence_threshold: float = 0.7  # Below this, flag for human review
    max_text_length: int = 500

    class Config:
        env_prefix = "VOICE_AI_"


settings = Settings()
