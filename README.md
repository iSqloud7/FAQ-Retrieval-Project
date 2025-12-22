# FAQ Retrieval Project

## Overview
This project implements a **semantic FAQ retrieval system** for customer support use cases.  
Instead of relying on keyword matching, it uses **sentence embeddings** and **cosine similarity** to retrieve the most relevant answers based on the meaning of a user’s query.

The system supports **multilingual queries** (e.g. English and Macedonian), follows **clean code principles**, and is designed with **scalability and production-readiness** in mind.

## Approach
The solution follows a standard semantic search / information retrieval pipeline:

### =>FAQ Dataset
- Small mock dataset (20 Q&A pairs) simulates real customer-support FAQs.
- Includes overlapping intents (e.g. login vs password reset) and multilingual entries.

### =>Embedding Generation
- FAQ questions are converted into dense vector representations using a pretrained sentence embedding model.
- User queries are embedded using the same model to ensure semantic comparability.

### =>Similarity Computation
- Cosine similarity is used to measure semantic similarity between the query embedding and FAQ embeddings.
- This metric is widely used in embedding-based retrieval due to its **efficiency and robustness**.

### =>Retrieval
- Retrieves the **top 3 most relevant FAQs** based on similarity scores.
- Highest-scoring result is returned as the **best matching answer**.

### =>Confidence Score
- Similarity scores are exposed as **confidence indicators**, which can be used for thresholding or escalation to human agents.

## Tools & Technologies Used
- Python 3  
- sentence-transformers  
- NumPy  

## Project Structure
<img width="457" height="281" alt="image" src="https://github.com/user-attachments/assets/99b3d3cc-4ab7-4d58-843d-e5ba217fdda7" />

- Separation of concerns for **scalability and maintainability**.

## How to Run the Project
1. **Clone the repository**
```bash
git clone https://github.com/iSqloud7/FAQ-Retrieval-Project.git
cd FAQ-Retrieval-Project
```

2. **Create and activate virtual environment**
- Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

- macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies.**
```bash
pip install -r requirements.txt
```
- The first run may take a minute as the embedding model downloads.

4. **Run the application.**
```bash
python -m src.main
```

5. **Test with example queries.**
```bash
English: I forgot my password
Macedonian: Не можам да се најавам
```
- The system returns the top relevant FAQs with confidence scores.
