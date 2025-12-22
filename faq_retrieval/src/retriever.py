import numpy as np
from typing import List, Dict
from src.similarity import cosine_similarity


class FAQRetriever:
    """
    Retrieves the top-K most semantically similar FAQs for a given query embedding.
    """

    def __init__(self, faq_path: str, embeddings: np.ndarray, faq_data: List[Dict]):
        """
        Initialize the FAQ retriever.

        Args:
            faq_path: Path to the FAQ JSON file (for reference, not used in logic).
            faq_data: List of FAQ dictionaries with 'question' and 'answer'.
            embeddings: Precomputed embeddings of all FAQ questions.
        """

        self.faq_data = faq_data
        self.embeddings = embeddings

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-K FAQs most similar to the query.

        Args:
            query_embedding: Vector representation of the user query.
            top_k: Number of top results to return.

        Returns:
            List of dictionaries containing 'question', 'answer', and 'score'.
        """

        similarities = cosine_similarity(query_embedding, self.embeddings)
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "question": self.faq_data[idx]["question"],
                "answer": self.faq_data[idx]["answer"],
                "score": float(similarities[idx])
            })

        return results
