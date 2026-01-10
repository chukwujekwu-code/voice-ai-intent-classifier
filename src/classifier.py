from dataclasses import dataclass
from typing import Any, List, Optional
import joblib

from .config import settings
from .encoder import LaBSEEncoder


@dataclass
class PredictionResult:
    """Result of an intent classification prediction."""

    intent: str
    confidence: float
    is_low_confidence: bool
    all_probabilities: dict[str, float]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "is_low_confidence": self.is_low_confidence,
            "all_probabilities": self.all_probabilities
        }


class IntentClassifier:
    """Intent classifier using LaBSE embeddings + trained classifier."""

    _instance: "IntentClassifier | None" = None

    def __new__(cls) -> "IntentClassifier":
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the classifier."""
        if self._initialized:
            return

        self.encoder = LaBSEEncoder()
        self._classifier: Any = None
        self._model_info: Optional[dict] = None
        self._classes: List[str] = []
        self._load_classifier()
        self._initialized = True

    def _load_classifier(self) -> None:
        """Load the trained classifier and model info."""
        classifier_path = settings.models_dir / settings.classifier_path
        info_path = settings.models_dir / settings.model_info_path

        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")

        print(f"Loading classifier from: {classifier_path}")
        self._classifier = joblib.load(classifier_path)

        if info_path.exists():
            self._model_info = joblib.load(info_path)
            self._classes = self._model_info.get("classes", [])
            print(f"Model info loaded. Classes: {len(self._classes)}")
        else:
            self._classes = list(self._classifier.classes_)
            print(f"Classes from classifier: {len(self._classes)}")

    @property
    def classes(self) -> List[str]:
        """Return the list of intent classes."""
        return self._classes

    @property
    def model_info(self) -> Optional[dict]:
        """Return model metadata."""
        return self._model_info

    def predict(
        self,
        text: str,
        threshold: Optional[float] = None
    ) -> PredictionResult:
        """
        Predict the intent for a single utterance.

        Args:
            text: The utterance text to classify
            threshold: Confidence threshold (defaults to settings.confidence_threshold)

        Returns:
            PredictionResult with intent, confidence, and probabilities
        """
        if threshold is None:
            threshold = settings.confidence_threshold

        # Encode text
        embedding = self.encoder.encode_single(text)

        # Get prediction and probabilities
        intent = self._classifier.predict([embedding])[0]
        probabilities = self._classifier.predict_proba([embedding])[0]

        # Get confidence (probability of predicted class)
        intent_idx = list(self._classifier.classes_).index(intent)
        confidence = float(probabilities[intent_idx])

        # Build probability dict
        all_probs = {
            cls: float(prob)
            for cls, prob in zip(self._classifier.classes_, probabilities)
        }

        return PredictionResult(
            intent=intent,
            confidence=confidence,
            is_low_confidence=confidence < threshold,
            all_probabilities=all_probs
        )

    def predict_batch(
        self,
        texts: List[str],
        threshold: Optional[float] = None
    ) -> List[PredictionResult]:
        """
        Predict intents for multiple utterances.

        Args:
            texts: List of utterance texts to classify
            threshold: Confidence threshold

        Returns:
            List of PredictionResult objects
        """
        if threshold is None:
            threshold = settings.confidence_threshold

        # Encode all texts
        embeddings = self.encoder.encode(texts)

        # Get predictions and probabilities
        intents = self._classifier.predict(embeddings)
        all_probabilities = self._classifier.predict_proba(embeddings)

        results = []
        for intent, probs in zip(intents, all_probabilities):
            intent_idx = list(self._classifier.classes_).index(intent)
            confidence = float(probs[intent_idx])

            all_probs = {
                cls: float(prob)
                for cls, prob in zip(self._classifier.classes_, probs)
            }

            results.append(PredictionResult(
                intent=intent,
                confidence=confidence,
                is_low_confidence=confidence < threshold,
                all_probabilities=all_probs
            ))

        return results

    def get_top_k_intents(
        self,
        text: str,
        k: int = 3
    ) -> List[tuple[str, float]]:
        """
        Get top-k most likely intents for an utterance.

        Args:
            text: The utterance text
            k: Number of top intents to return

        Returns:
            List of (intent, probability) tuples sorted by probability
        """
        embedding = self.encoder.encode_single(text)
        probabilities = self._classifier.predict_proba([embedding])[0]

        # Sort by probability
        intent_probs = list(zip(self._classifier.classes_, probabilities))
        intent_probs.sort(key=lambda x: x[1], reverse=True)

        return [(intent, float(prob)) for intent, prob in intent_probs[:k]]
