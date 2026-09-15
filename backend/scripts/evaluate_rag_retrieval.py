"""Evaluate the current local RAG path on the built-in destination cases."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.guide_catalog import canonical_destination
from app.rag.retriever import _rerank_chunks, _rule_based_query
from app.rag.vector_db import search_guide_chunks


EVAL_CASES_PATH = BACKEND_DIR / "eval" / "rag_eval_cases.json"
NOISE_TITLES = {"文档开头", "目的地简介"}


def main() -> int:
    cases = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))
    latencies: list[float] = []
    top1_hits = 0
    reciprocal_rank_total = 0.0
    noise_hits = 0
    pollution_hits = 0
    inspected_count = 0

    for case in cases:
        destination = case["destination"]
        canonical = canonical_destination(destination)
        query = _rule_based_query(
            destination=destination,
            preferences=case.get("preferences", []),
            travel_style=case.get("travel_style"),
            free_text_input=case.get("free_text_input"),
        )

        started = time.perf_counter()
        candidates = search_guide_chunks(
            query=query,
            top_k=8,
            destination=canonical,
        )
        reranked = _rerank_chunks(query, candidates, top_k=3, destination=canonical)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)

        titles = [chunk.get("title", "") for chunk in reranked]
        expected_keywords = case.get("expected_title_keywords", [])

        hit_rank = None
        for rank, title in enumerate(titles, start=1):
            if any(keyword in title for keyword in expected_keywords):
                hit_rank = rank
                break

        if hit_rank == 1:
            top1_hits += 1
        if hit_rank is not None:
            reciprocal_rank_total += 1.0 / hit_rank

        for chunk in reranked:
            inspected_count += 1
            title = chunk.get("title", "")
            chunk_destination = chunk.get("destination", "")
            if title in NOISE_TITLES:
                noise_hits += 1
            if canonical and chunk_destination and chunk_destination != canonical:
                pollution_hits += 1

    top1_rate = top1_hits / len(cases)
    mrr = reciprocal_rank_total / len(cases)
    noise_rate = noise_hits / max(inspected_count, 1)
    pollution_rate = pollution_hits / max(inspected_count, 1)

    print(json.dumps({
        "cases": len(cases),
        "top1_hits": top1_hits,
        "top1_rate": round(top1_rate, 4),
        "mrr": round(mrr, 4),
        "noise_rate": round(noise_rate, 4),
        "cross_destination_pollution_rate": round(pollution_rate, 4),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p95_latency_ms": round(
            sorted(latencies)[int(len(latencies) * 0.95) - 1],
            1,
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
