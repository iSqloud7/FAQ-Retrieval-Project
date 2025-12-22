import json
from src.embeddings import EmbeddingModel
from src.retriever import FAQRetriever

FAQ_PATH = "data/faq.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_faq(path: str):
    """
    Load FAQ data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        List of FAQ dictionaries containing 'question' and 'answer'.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """
    Main entry point for the FAQ Retrieval application.
    Loads FAQ data, initializes the embedding model and retriever,
    and interacts with the user for query input.
    """

    faq_data = load_faq(FAQ_PATH)
    questions = [item["question"] for item in faq_data]

    embedding_model = EmbeddingModel(MODEL_NAME)
    faq_embeddings = embedding_model.encode(questions)

    retriever = FAQRetriever(FAQ_PATH, faq_embeddings, faq_data)

    print("Welcome to the FAQ Retrieval System! Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your question: ")
        if user_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        query_embedding = embedding_model.encode([user_query])[0]
        results = retriever.retrieve(query_embedding)

        print("\nTop Results:")
        for r in results:
            print(f"\nQ: {r['question']}")
            print(f"A: {r['answer']}")
            print(f"Confidence: {r['score']:.2f}")


if __name__ == "__main__":
    main()
