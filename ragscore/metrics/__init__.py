"""Evaluation metrics for RAGScore, grouped by what they need to run.

`retrieval` needs nothing but ranked lists (pure functions, no LLM).
`faithfulness`, `answer_relevance`, `answer_correctness` call the judge LLM.
"""
