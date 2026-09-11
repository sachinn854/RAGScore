"""A minimal RAG pipeline: retrieve top-k chunks, then generate an answer.

Flow: question -> embed -> Chroma top-k -> stuff chunks into a prompt ->
`gen_model` answers using only that context. The returned `RagResult` carries
everything the M4-M6 metrics need: the retrieved chunk ids, the exact context
string the model saw, and the generated answer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import chromadb

from .config import Config, build_arg_parser, load_config
from .ingest import COLLECTION_NAME
from .llm import LLM

_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the context "
    "below. If the answer is not in the context, reply exactly: I don't know."
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float          # cosine similarity in [0, 1]; 1.0 == identical


@dataclass
class RagResult:
    question: str
    retrieved: list[RetrievedChunk]
    context: str           # the exact stuffed context the model saw
    answer: str
    usage: dict            # tokens + cost from the generation call


def get_collection(cfg: Config) -> "chromadb.Collection":
    """Open the persistent 'corpus' collection, or explain how to build it."""
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:  # collection missing / store not built yet
        raise RuntimeError(
            "Chroma collection not found -- run: python -m ragscore.ingest"
        ) from exc


def retrieve(
    question: str,
    *,
    cfg: Config,
    llm: LLM,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Embed the question and return the top-k most similar chunks."""
    top_k = top_k or cfg.top_k
    collection = get_collection(cfg)

    query_vec = llm.embed([question])[0]
    res = collection.query(query_embeddings=[query_vec], n_results=top_k)

    # Chroma returns each field as a list-of-lists (one inner list per query).
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    return [
        RetrievedChunk(
            chunk_id=cid,
            doc_id=str(meta.get("doc_id", "")),
            text=doc,
            score=1.0 - dist,
        )
        for cid, doc, meta, dist in zip(ids, docs, metas, dists)
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the chunks and tag each with its source doc for traceability."""
    blocks = [
        f"[{n}] ({c.doc_id})\n{c.text}"
        for n, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> list[dict]:
    """Assemble the system + user messages for the generation call."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def generate(
    question: str,
    *,
    cfg: Config | None = None,
    llm: LLM | None = None,
    top_k: int | None = None,
) -> RagResult:
    """Run the full retrieve-then-generate pipeline for one question."""
    cfg = cfg or load_config()
    llm = llm or LLM(cfg)

    chunks = retrieve(question, cfg=cfg, llm=llm, top_k=top_k)
    context = format_context(chunks)
    messages = build_messages(question, context)

    chat = llm.chat(messages, model=cfg.gen_model, temperature=0.0)
    return RagResult(
        question=question,
        retrieved=chunks,
        context=context,
        answer=chat.text.strip(),
        usage={
            "model": chat.model,
            "prompt_tokens": chat.prompt_tokens,
            "completion_tokens": chat.completion_tokens,
            "cost": chat.cost,
            "cached": chat.cached,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()],
        description="Ask one question against the ingested corpus.",
    )
    parser.add_argument("--question", "-q", required=True, help="the question to answer")
    args = parser.parse_args()

    cfg = load_config()
    result = generate(args.question, cfg=cfg, top_k=args.top_k)

    print("retrieved:")
    for c in result.retrieved:
        print(f"  {c.score:5.3f}  {c.chunk_id}")
    print(f"\nanswer:\n{result.answer}")
    print(f"\nusage: {result.usage}")
