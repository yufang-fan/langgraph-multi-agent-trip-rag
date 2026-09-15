"""Ingest local markdown guides into ChromaDB."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings
from app.rag.vector_db import ingest_guide_chunks_to_chroma, load_guide_chunks


def main() -> int:
    settings = get_settings()
    chunks = load_guide_chunks()
    print("=== RAG ingestion ===")
    print(f"guide_chunks: {len(chunks)}")
    print(f"embedding_provider: {settings.embedding_provider}")
    print(f"embedding_model: {settings.embedding_model}")
    print(f"chroma_db_dir: {settings.chroma_db_dir}")
    print(f"collection: {settings.chroma_collection_name}")
    print()

    written_count = ingest_guide_chunks_to_chroma()
    print(f"written_count: {written_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
