from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingModel:
    """
    Wrapper for a sentence-transformers model to generate embeddings for a list of texts.
    """

    def __init__(self, model_name: str):
        """
        Initialize the embedding model.

        Args:
            model_name: Name or path of the pretrained sentence-transformers model.
        """

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into dense vector embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            A numpy array of shape (len(texts), embedding_dim) containing the embeddings.
        """
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
