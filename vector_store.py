"""ChromaDB vector store for chunk embeddings with modality tags."""

import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv() 


_COLLECTION_NAME = "rag_chunks"
_EMBED_MODEL = "BAAI/bge-large-en"
_CHROMA_PATH = "./chroma_db"

_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embedder: Optional[SentenceTransformer] = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _chunk_id(document_id: str, modality: str, idx: int) -> str:
    raw = f"{document_id}:{modality}:{idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def index_document(document_id: str, text: str, tables: list, charts: list, equations: list) -> int:
    """Chunk and embed all modalities. Returns number of chunks indexed."""
    collection = _get_collection()
    embedder = _get_embedder()

    all_chunks: list[tuple[str, str, str]] = []  # (chunk_id, text, modality)

    # Text chunks
    for idx, chunk in enumerate(_chunk_text(text)):
        all_chunks.append((_chunk_id(document_id, "text", idx), chunk, "text"))

    # Table chunks: serialize each table to text
    for idx, tbl in enumerate(tables):
        headers = tbl.get("headers", [])
        rows = tbl.get("rows", [])
        serialized = "Table:\n" + " | ".join(str(h) for h in headers) + "\n"
        for row in rows:
            serialized += " | ".join(str(c) for c in row) + "\n"
        all_chunks.append((_chunk_id(document_id, "table", idx), serialized.strip(), "table"))

    # Chart description chunks
    for idx, chart in enumerate(charts):
        desc = chart.get("description", f"Chart {idx}")
        all_chunks.append((_chunk_id(document_id, "chart", idx), desc, "chart"))

    # Equation chunks
    for idx, eq in enumerate(equations):
        all_chunks.append((_chunk_id(document_id, "equation", idx), eq, "equation"))

    if not all_chunks:
        return 0

    ids, texts, modalities = zip(*all_chunks)
    embeddings = embedder.encode(list(texts), show_progress_bar=False).tolist()

    collection.upsert(
        ids=list(ids),
        embeddings=embeddings,
        documents=list(texts),
        metadatas=[{"document_id": document_id, "modality": m} for m in modalities],
    )
    return len(all_chunks)


def retrieve(query: str, document_id: Optional[str] = None, top_k: int = 5) -> list[dict]:
    """Return top-k relevant chunks for a query."""
    collection = _get_collection()
    embedder = _get_embedder()

    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()[0]

    where = {"document_id": document_id} if document_id else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(collection.count(), 1)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for text, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": text,
            "modality": meta.get("modality", "unknown"),
            "document_id": meta.get("document_id", ""),
            "score": round(1 - dist, 4),
        })
    return chunks