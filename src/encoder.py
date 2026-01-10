from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


class LaBSEEncoder:
    """Wrapper for LaBSE sentence embeddings."""

    _instance: "LaBSEEncoder | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "LaBSEEncoder":
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the encoder (loads model on first call)."""
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """Load the LaBSE model."""
        print(f"Loading embedding model: {settings.embedding_model}")
        self._model = SentenceTransformer(settings.embedding_model)
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if self._model is None:
            self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single string or list of strings to encode
            show_progress_bar: Whether to show progress bar for batch encoding

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if self._model is None:
            self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.

        Args:
            text: Text to encode

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.encode(text)[0]
