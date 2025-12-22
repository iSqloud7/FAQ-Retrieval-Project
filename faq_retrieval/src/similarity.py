import numpy as np


def cosine_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """
        Compute cosine similarity between a query embedding and multiple document embeddings.

        Args:
            query_embedding: A 1D numpy array representing the query embedding.
            doc_embeddings: A 2D numpy array where each row is a document embedding.

        Returns:
            A 1D numpy array of similarity scores between the query and each document.
        """

    # Query normalization.
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Document normalization.
    docs_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    # Cosine similarity.
    return np.dot(docs_norm, query_norm)
