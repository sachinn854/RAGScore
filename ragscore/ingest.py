"""Ingest: turn the source corpus into a searchable Chroma index.

Part A: load the corpus files and split them into overlapping fixed-size
character windows. Pure functions, no API calls, cheap to test.

Part B: embed each chunk with Gemini and store it in a persistent Chroma
collection that `rag.py` (M2) will query.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .config import Config, build_arg_parser, load_config
from .llm import LLM

_CORPUS_SUFFIXES = (".md", ".txt")

# Chroma collection name and how many chunks to embed per API call.
COLLECTION_NAME = "corpus"
_EMBED_BATCH = 64


@dataclass
class Chunk:
    """One piece of a source document."""

    chunk_id: str   # f"{doc_id}::{index}", e.g. "02_self_attention::3"
    doc_id: str     # source file stem, e.g. "02_self_attention"
    text: str       # the chunk content
    index: int      # 0-based position of this chunk within its document


def list_corpus_files(corpus_dir: Path) -> list[Path]:
    """Return all .md and .txt files under corpus_dir, sorted by filename."""
    files = sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _CORPUS_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No {' or '.join(_CORPUS_SUFFIXES)} files found in {corpus_dir}"
        )
    return files


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size character windows.

    Consecutive windows share `overlap` characters so a fact that straddles a
    window boundary still appears whole in at least one chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    i = 0
    while i < len(text):
        piece = text[i : i + chunk_size].strip()
        if piece:
            chunks.append(piece)
        i += step
    return chunks


def build_chunks(
    corpus_dir: Path, *, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Read every corpus file and turn it into Chunk objects with stable ids."""
    all_chunks: list[Chunk] = []
    for path in list_corpus_files(corpus_dir):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        for index, piece in enumerate(
            chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{index}",
                    doc_id=doc_id,
                    text=piece,
                    index=index,
                )
            )
    return all_chunks


# --------------------------------------------------------------------------
# Part B: embed the chunks and store them in Chroma
# --------------------------------------------------------------------------

def _batches(items: list, size: int):
    """Yield successive slices of `items` of length at most `size`."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_and_store(
    chunks: list[Chunk], *, cfg: Config | None = None, llm: LLM | None = None
) -> "chromadb.Collection":
    """Embed every chunk and rebuild the Chroma collection from scratch.

    The collection is dropped and recreated on each run so a re-ingest never
    leaves stale chunks behind. We pass our own Gemini vectors to
    ``collection.add``; Chroma's built-in embedding model is never used.
    """
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    llm = llm or LLM(cfg)

    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME, embedding_function=None
    )

    for batch in _batches(chunks, _EMBED_BATCH):
        vectors = llm.embed([c.text for c in batch])
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors,
            metadatas=[{"doc_id": c.doc_id, "index": c.index} for c in batch],
        )
    return collection


def ingest(cfg: Config | None = None) -> "chromadb.Collection":
    """Full pipeline: corpus files -> chunks -> embeddings -> Chroma."""
    cfg = cfg or load_config()
    chunks = build_chunks(
        cfg.corpus_dir, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap
    )
    return embed_and_store(chunks, cfg=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()], description="Build the Chroma index from the corpus."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="chunk only, skip embedding/storage"
    )
    args = parser.parse_args()
    cfg = load_config()

    chunks = build_chunks(
        cfg.corpus_dir, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap
    )
    docs = sorted({c.doc_id for c in chunks})
    print(f"{len(chunks)} chunks from {len(docs)} docs in {cfg.corpus_dir}")
    print(f"chunk_size={cfg.chunk_size}  overlap={cfg.chunk_overlap}")
    print("\n--- first chunk ---")
    print(f"[{chunks[0].chunk_id}]  ({len(chunks[0].text)} chars)")
    print(chunks[0].text)

    if args.dry_run:
        raise SystemExit(0)

    llm = LLM(cfg)
    collection = embed_and_store(chunks, cfg=cfg, llm=llm)
    print(f"\nstored {collection.count()} chunks in {cfg.chroma_dir}")
    print("llm usage:", llm.usage.as_dict())
