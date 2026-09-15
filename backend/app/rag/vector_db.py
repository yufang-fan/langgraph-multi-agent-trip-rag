"""Markdown ingestion, keyword fallback, and Chroma vector retrieval."""

from __future__ import annotations

import logging
import os
import re
from hashlib import md5
from pathlib import Path
from typing import Any

from ..config import get_settings
from .guide_catalog import canonical_destination, destination_for_guide

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"


def _flush_chunk(
    chunks: list[dict[str, str]],
    title: str,
    lines: list[str],
    source_name: str,
) -> None:
    if not lines:
        return
    text = "\n".join(lines).strip()
    if not text:
        return
    chunks.append({"title": title, "text": text, "source": source_name})


def _split_markdown_into_chunks(markdown_text: str, source_name: str) -> list[dict[str, str]]:
    """Split guide markdown by headings, which matches travel guide structure well."""
    chunks: list[dict[str, str]] = []
    current_title = "文档开头"
    current_lines: list[str] = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            _flush_chunk(chunks, current_title, current_lines, source_name)
            current_title = heading.group(2).strip()
            current_lines = []
        elif line:
            current_lines.append(line)

    _flush_chunk(chunks, current_title, current_lines, source_name)
    return chunks


def _build_chunk_id(source: str, title: str, text: str) -> str:
    digest = md5(f"{source}|{title}|{text}".encode("utf-8")).hexdigest()
    return f"{source}_{digest}"


def _build_document_text(chunk: dict[str, str]) -> str:
    return f"{chunk['title']}\n{chunk['text']}"


def load_guide_chunks() -> list[dict[str, str]]:
    """Load all local travel guides and convert them into searchable chunks."""
    chunks: list[dict[str, str]] = []
    if not DATA_DIR.exists():
        return chunks

    for guide_file in sorted(DATA_DIR.glob("*.md")):
        destination = destination_for_guide(guide_file.name)
        if destination is None:
            logger.warning("Guide file is not registered: %s", guide_file.name)
            continue

        text = guide_file.read_text(encoding="utf-8")
        for chunk in _split_markdown_into_chunks(text, guide_file.name):
            chunks.append(
                {
                    "id": _build_chunk_id(chunk["source"], chunk["title"], chunk["text"]),
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "destination": destination,
                }
            )
    return chunks


def _extract_keywords(query: str) -> list[str]:
    """Extract usable keywords from a mostly-Chinese query."""
    parts = [part.strip() for part in re.split(r"[\s,，。；;、|/]+", query.strip()) if part.strip()]
    keywords = list(dict.fromkeys(parts))

    if len(keywords) == 1 and len(keywords[0]) > 2:
        keyword = keywords[0]
        keywords.extend(keyword[index:index + 2] for index in range(0, len(keyword) - 1))

    return list(dict.fromkeys(keywords))


def _score_chunk(query: str, chunk_text: str) -> int:
    return sum(1 for keyword in _extract_keywords(query) if keyword in chunk_text)


def _search_guide_chunks_by_keywords(
    query: str,
    top_k: int = 6,
    destination: str | None = None,
) -> list[dict[str, str]]:
    scored_chunks: list[tuple[int, dict[str, str]]] = []
    for chunk in load_guide_chunks():
        if destination and chunk.get("destination") != destination:
            continue
        score = _score_chunk(query, _build_document_text(chunk))
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]


def _resolved_chroma_dir() -> Path:
    settings = get_settings()
    raw_path = Path(settings.chroma_db_dir)
    if raw_path.is_absolute():
        return raw_path
    return BACKEND_DIR / raw_path


def _build_embedding_function() -> Any | None:
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            logger.warning("langchain_openai is unavailable; OpenAI embeddings disabled.")
            return None

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or None
        if not api_key:
            logger.warning("No LLM API key found; OpenAI embeddings disabled.")
            return None
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=api_key,
            base_url=base_url,
            chunk_size=settings.embedding_batch_size,
        )

    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    except ImportError:
        logger.warning("chromadb is unavailable; using keyword-only RAG fallback.")
        return None

    try:
        return DefaultEmbeddingFunction()
    except Exception as exc:
        logger.warning("Local embedding function init failed: %s", exc)
        return None


def _embed_texts(embedding_function: Any, texts: list[str]) -> list[list[float]]:
    if hasattr(embedding_function, "embed_documents"):
        return embedding_function.embed_documents(texts)
    return embedding_function(texts)


def _embed_query(embedding_function: Any, query: str) -> list[float] | None:
    if hasattr(embedding_function, "embed_query"):
        return embedding_function.embed_query(query)
    values = embedding_function([query])
    if values and values[0]:
        return values[0]
    return None


def _get_chroma_collection() -> Any | None:
    settings = get_settings()
    try:
        import chromadb
    except ImportError:
        return None

    try:
        chroma_dir = _resolved_chroma_dir()
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        return client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        logger.warning("Chroma collection init failed: %s", exc)
        return None


def ingest_guide_chunks_to_chroma() -> int:
    """Build or update the Chroma collection from all registered guides."""
    embedding_function = _build_embedding_function()
    collection = _get_chroma_collection()
    chunks = load_guide_chunks()

    if embedding_function is None:
        raise RuntimeError("No embedding function is available for Chroma ingestion.")
    if collection is None:
        raise RuntimeError("ChromaDB is not installed or could not be initialized.")
    if not chunks:
        raise RuntimeError("No registered guide chunks were found.")

    documents = [_build_document_text(chunk) for chunk in chunks]
    vectors = _embed_texts(embedding_function, documents)
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [
        {
            "title": chunk["title"],
            "source": chunk["source"],
            "destination": chunk["destination"],
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors,
    )
    return len(chunks)


def _search_guide_chunks_by_chroma(
    query: str,
    top_k: int = 6,
    destination: str | None = None,
) -> list[dict[str, str]]:
    collection = _get_chroma_collection()
    if collection is None or collection.count() == 0:
        return []

    embedding_function = _build_embedding_function()
    if embedding_function is None:
        return []

    query_embedding = _embed_query(embedding_function, query)
    if query_embedding is None:
        return []

    query_args: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas"],
    }
    if destination:
        query_args["where"] = {"destination": destination}

    result = collection.query(**query_args)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    matched_chunks: list[dict[str, str]] = []
    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        title = metadata.get("title", "未命名片段")
        source = metadata.get("source", "未知来源")
        chunk_destination = metadata.get("destination", "")
        text = document.split("\n", 1)[1] if "\n" in document else document
        matched_chunks.append(
            {
                "title": title,
                "text": text,
                "source": source,
                "destination": chunk_destination,
            }
        )
    return matched_chunks


def _build_ngram_vector(text: str, dimension: int = 1024) -> list[float]:
    """Build a dependency-free character n-gram vector for the local fallback."""
    import numpy as np

    vector = np.zeros(dimension, dtype=np.float32)
    compact = re.sub(r"\s+", "", text)
    tokens = [compact]
    if len(compact) > 1:
        tokens.extend(compact[index:index + 2] for index in range(0, len(compact) - 1))
        tokens.extend(compact[index:index + 3] for index in range(0, max(0, len(compact) - 2)))

    for token in tokens:
        index = int(md5(token.encode("utf-8")).hexdigest()[:8], 16) % dimension
        vector[index] += 1.0

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.tolist()


def _search_guide_chunks_by_local_vector(
    query: str,
    top_k: int = 6,
    destination: str | None = None,
) -> list[dict[str, str]]:
    """Vector retrieval fallback that works without Chroma or native extensions."""
    chunks = [
        chunk
        for chunk in load_guide_chunks()
        if not destination or chunk.get("destination") == destination
    ]
    if not chunks:
        return []

    query_vector = _build_ngram_vector(query)
    scored: list[tuple[float, dict[str, str]]] = []
    for chunk in chunks:
        chunk_vector = _build_ngram_vector(_build_document_text(chunk))
        similarity = sum(a * b for a, b in zip(query_vector, chunk_vector))
        scored.append((similarity, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def search_guide_chunks(
    query: str,
    top_k: int = 6,
    destination: str | None = None,
) -> list[dict[str, str]]:
    """Prefer Chroma vector retrieval and fall back to keyword matching."""
    canonical = canonical_destination(destination)
    chroma_results = _search_guide_chunks_by_chroma(query, top_k, canonical)
    if chroma_results:
        return chroma_results
    local_vector_results = _search_guide_chunks_by_local_vector(query, top_k, canonical)
    if local_vector_results:
        return local_vector_results
    return _search_guide_chunks_by_keywords(query, top_k, canonical)


def get_rag_status() -> dict[str, Any]:
    """Return enough status to surface in logs and health checks."""
    collection = _get_chroma_collection()
    settings = get_settings()
    chroma_count = collection.count() if collection is not None else 0
    return {
        "enabled": True,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "chroma_collection": settings.chroma_collection_name,
        "chroma_count": chroma_count,
        "vector_backend": "chroma" if chroma_count > 0 else "local_ngram",
        "rerank_provider": settings.rerank_provider,
        "rerank_model": settings.rerank_model,
        "noise_prefilter": True,
        "guide_files": len(list(DATA_DIR.glob("*.md"))) if DATA_DIR.exists() else 0,
    }
