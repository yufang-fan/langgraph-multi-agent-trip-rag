"""Print RAG retrieval output for a sample destination and preference."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag import get_rag_status, retrieve_guide_context
from app.rag.retriever import build_travel_query


def main() -> int:
    destination = sys.argv[1] if len(sys.argv) > 1 else "北京"
    preferences = ["历史文化", "美食"]
    travel_style = "舒适均衡"
    free_text = "想拍照，也想去博物馆"

    query = build_travel_query(destination, preferences, travel_style, free_text)
    contexts = retrieve_guide_context(
        destination=destination,
        preferences=preferences,
        travel_style=travel_style,
        free_text_input=free_text,
    )
    print(json.dumps(get_rag_status(), ensure_ascii=False, indent=2))
    print()
    print(f"query: {query}")
    print(f"context_count: {len(contexts)}")
    for index, context in enumerate(contexts, start=1):
        print(f"\n--- context {index} ---")
        print(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
