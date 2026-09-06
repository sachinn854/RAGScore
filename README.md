# RAGScore

A from-scratch **RAG evaluation harness**. Give it a document corpus and a set of
questions with ground-truth answers; it scores a RAG pipeline's answers on:

- **Retrieval:** precision@k, recall@k, hit@k, MRR
- **Faithfulness:** are the answer's claims supported by the retrieved context?
- **Answer relevance:** does the answer address the question?
- **Answer correctness:** how close is the answer to the ideal answer?

Built without LangChain / LlamaIndex — every piece is hand-written to keep the
evaluation logic transparent.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env      # then add your OpenRouter key
```

## Usage
_Coming with milestone M7._
```bash
python -m ragscore.runner
```

## Results
_Experiment writeups will go here (milestone M9)._
