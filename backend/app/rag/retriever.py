"""Query construction, retrieval, reranking, and cache orchestration."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from ..config import get_settings
from .cache_service import get_cached_json, set_cached_json
from .guide_catalog import canonical_destination
from .vector_db import search_guide_chunks

logger = logging.getLogger(__name__)

_NOISE_TITLES = {"文档开头", "目的地简介"}


def _append_unique(parts: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized not in parts:
        parts.append(normalized)


def _rule_based_query(
    destination: str,
    preferences: list[str] | None = None,
    travel_style: str | None = None,
    free_text_input: str | None = None,
) -> str:
    parts = [destination]
    for preference in preferences or []:
        _append_unique(parts, preference)
    if travel_style:
        _append_unique(parts, travel_style)
    if free_text_input:
        for piece in re.split(r"[\s,，。；;、]+", free_text_input.strip()):
            _append_unique(parts, piece)
    for stable_term in ["景点", "行程", "攻略", "推荐"]:
        _append_unique(parts, stable_term)
    return " ".join(parts)


def _llm_rewrite_query(
    destination: str,
    preferences: list[str] | None,
    travel_style: str | None,
    free_text_input: str | None,
) -> str | None:
    try:
        from ..services.llm_service import get_llm

        llm = get_llm()
        system_prompt = (
            "你是旅行攻略 RAG 查询改写专家。"
            "请把用户需求改写为适合向量检索的中文关键词组合。"
            "只输出空格分隔的关键词，不要解释，不要标点。"
            "关键词要包含目的地、旅行偏好、景点或餐饮等具体词。"
        )
        human_parts = [f"目的地：{destination}"]
        if preferences:
            human_parts.append(f"偏好：{'、'.join(preferences)}")
        if travel_style:
            human_parts.append(f"风格：{travel_style}")
        if free_text_input:
            human_parts.append(f"额外要求：{free_text_input}")

        response = llm.invoke(
            [
                ("system", system_prompt),
                ("human", "\n".join(human_parts)),
            ]
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        query = str(content).strip()
        if query:
            return query
    except Exception as exc:
        logger.warning("LLM query rewrite failed, using rule-based query: %s", exc)
    return None


def build_travel_query(
    destination: str,
    preferences: list[str] | None = None,
    travel_style: str | None = None,
    free_text_input: str | None = None,
) -> str:
    """Build a retrieval query with LLM rewrite and rule-based fallback."""
    rewritten = _llm_rewrite_query(
        destination=destination,
        preferences=preferences,
        travel_style=travel_style,
        free_text_input=free_text_input,
    )
    if rewritten:
        canonical = canonical_destination(destination)
        if canonical and canonical not in rewritten:
            rewritten = f"{canonical} {rewritten}"
        return rewritten
    return _rule_based_query(
        destination=destination,
        preferences=preferences,
        travel_style=travel_style,
        free_text_input=free_text_input,
    )


def _extract_keywords(query: str) -> list[str]:
    return [part for part in re.split(r"[\s,，。；;、|/]+", query.strip()) if part]


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _prefilter_noise_chunks(
    query: str,
    chunks: list[dict[str, str]],
    destination: str | None,
) -> list[dict[str, str]]:
    """Drop known low-information chunks before expensive cross-encoder rerank."""
    clean_chunks: list[dict[str, str]] = []
    for chunk in chunks:
        title = chunk.get("title", "")
        text = chunk.get("text", "")
        chunk_destination = chunk.get("destination", "")
        reasons: list[str] = []

        if title in _NOISE_TITLES:
            reasons.append(f"noise_title:{title}")
        if len(text) < 20:
            reasons.append("noise_short_text")
        if destination and chunk_destination and chunk_destination != destination:
            reasons.append("noise_destination_mismatch")

        if not reasons:
            clean_chunks.append(chunk)
            continue

        enriched = dict(chunk)
        enriched["noise_reasons"] = reasons
        clean_chunks.append(enriched)

    # qwen3-rerank should not spend slots on explicit noise. Rule rerank can still
    # use the annotated reason when we need to explain why a chunk was removed.
    return [
        chunk for chunk in clean_chunks
        if not chunk.get("noise_reasons")
    ] or chunks


def _score_chunk_for_rerank(
    query: str,
    chunk: dict[str, str],
    destination: str | None,
) -> int:
    title = chunk.get("title", "")
    text = chunk.get("text", "")
    source = chunk.get("source", "")
    combined_text = f"{title}\n{text}"
    keywords = _extract_keywords(query)
    score = 0
    query_lower = query.lower()
    has_attraction_intent = any(
        keyword in query
        for keyword in (
            "历史文化",
            "自然风光",
            "艺术",
            "博物馆",
            "长城",
            "古城",
            "古镇",
            "公园",
            "骑行",
            "日落",
            "休闲",
            "茶馆",
            "咖啡",
            "拍照",
            "摄影",
            "夜景",
            "胡同",
            "外滩",
            "陆家嘴",
        )
    )
    has_food_intent = any(
        keyword in query
        for keyword in ("美食", "小吃", "餐饮", "吃", "火锅", "烤鸭")
    )
    has_hotel_intent = any(
        keyword in query
        for keyword in ("住宿", "酒店", "民宿", "客栈", "宾馆")
    )
    generic_terms = {"景点", "行程", "攻略", "推荐"}
    is_hotel_section = _contains_any(
        title,
        ["区域", "酒店", "民宿", "客栈", "宾馆", "饭店", "旅馆", "旅店", "海景"],
    )
    is_meal_section = _contains_any(
        title,
        ["小吃", "美食", "火锅", "烤鸭", "菜", "餐厅", "饭", "面", "汤", "粥", "粉"],
    )

    for keyword in keywords:
        if keyword in title:
            score += 3
        if keyword in text:
            score += 1 if keyword in generic_terms else 3

    if title == "文档开头":
        score -= 8
    if "行程" in title and "行程参考" not in title:
        score += 4
    if "行程参考" in title and has_attraction_intent:
        score += 5
    if "行程参考" in title and not has_attraction_intent and (has_food_intent or has_hotel_intent):
        score -= 5
    if "目的地简介" in title:
        score -= 2
    if is_hotel_section and not has_hotel_intent:
        score -= 8
    if is_hotel_section and has_hotel_intent:
        score += 8
    if is_meal_section and not has_food_intent and has_attraction_intent:
        score -= 4
    if is_meal_section and has_food_intent:
        score += 6
    if not is_hotel_section and not is_meal_section and has_attraction_intent:
        score += 3
    if _contains_any(title, ["餐饮", "预算"]) and not _contains_any(
        combined_text,
        ["日落", "傍晚", "拍照", "摄影", "出片", "洱海", "双廊", "慢节奏"],
    ):
        score -= 3

    if destination:
        chunk_destination = chunk.get("destination", "")
        if chunk_destination and chunk_destination != destination:
            score -= 5
        elif not chunk_destination and destination.lower() not in f"{source} {title} {text}".lower():
            score -= 5

    return score


def _rerank_with_dashscope(
    query: str,
    chunks: list[dict[str, str]],
    top_k: int,
) -> list[tuple[float, int]] | None:
    """Optional qwen3-rerank cross-encoder path."""
    settings = get_settings()
    api_key = settings.rerank_api_key or os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key or not chunks:
        return None

    documents = [
        f"{chunk.get('title', '')}\n{chunk.get('text', '')}"
        for chunk in chunks
    ]
    payload = {
        "model": settings.rerank_model,
        "documents": documents,
        "query": query,
        "top_n": min(top_k, len(documents)),
        "instruct": (
            "你是一个旅行攻略检索专家。"
            "从候选文档中检索最具体、最能直接回答用户旅行规划问题的片段。"
        ),
    }

    try:
        endpoint = f"{settings.rerank_base_url.rstrip('/')}/reranks"
        response = httpx.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if response.status_code != 200:
            logger.warning("DashScope rerank HTTP %s", response.status_code)
            return None

        data = response.json()
        results = data.get("output", {}).get("results", []) or data.get("results", [])
        if not results:
            return None

        scored = []
        for item in results:
            index = int(item.get("index", 0))
            score = float(item.get("relevance_score", 0))
            if 0 <= index < len(chunks):
                scored.append((score, index))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored or None
    except Exception as exc:
        logger.warning("DashScope rerank failed, using rule-based fallback: %s", exc)
        return None


def _rerank_chunks(
    query: str,
    chunks: list[dict[str, str]],
    top_k: int,
    destination: str | None,
) -> list[dict[str, str]]:
    settings = get_settings()
    candidate_chunks = _prefilter_noise_chunks(query, chunks, destination)
    if settings.rerank_provider.lower() == "dashscope":
        cross_encoder_results = _rerank_with_dashscope(query, candidate_chunks, top_k)
        if cross_encoder_results:
            reranked = []
            for score, original_index in cross_encoder_results:
                enriched_chunk = dict(candidate_chunks[original_index])
                enriched_chunk["rerank_score"] = round(score, 4)
                enriched_chunk["rerank_reasons"] = ["cross-encoder:qwen3-rerank"]
                reranked.append(enriched_chunk)
            return reranked[:top_k]

    scored = []
    for index, chunk in enumerate(candidate_chunks):
        enriched = dict(chunk)
        enriched["_score"] = _score_chunk_for_rerank(query, enriched, destination)
        scored.append((enriched["_score"], -index, enriched))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [chunk for _, _, chunk in scored[:top_k]]


def _format_chunk(chunk: dict[str, str]) -> str:
    return f"[来源: {chunk['source']} | 标题: {chunk['title']}]\n{chunk['text']}"


def retrieve_guide_context(
    destination: str,
    preferences: list[str] | None = None,
    travel_style: str | None = None,
    free_text_input: str | None = None,
    top_k: int = 5,
) -> list[str]:
    """Return reranked guide snippets for the trip planner prompt."""
    settings = get_settings()
    canonical = canonical_destination(destination)
    query = build_travel_query(
        destination=destination,
        preferences=preferences,
        travel_style=travel_style,
        free_text_input=free_text_input,
    )
    cache_key = (
        f"rag:guide:{canonical or 'all'}:{query}:{top_k}"
        .replace(" ", "_")
    )
    cached = get_cached_json(cache_key)
    if cached is not None:
        return [str(item) for item in cached]

    candidates = search_guide_chunks(
        query=query,
        top_k=max(top_k * 2, 8),
        destination=canonical,
    )
    reranked = _rerank_chunks(query, candidates, top_k, canonical)

    if len(reranked) < top_k:
        supplements = [
            f"{destination} 住宿 酒店 民宿",
            f"{destination} 餐饮 美食 餐厅",
        ]
        existing = {(chunk["source"], chunk["title"], chunk["text"]) for chunk in reranked}
        for supplement in supplements:
            for chunk in search_guide_chunks(supplement, top_k=2, destination=canonical):
                key = (chunk["source"], chunk["title"], chunk["text"])
                if key not in existing:
                    reranked.append(chunk)
                    existing.add(key)
                if len(reranked) >= top_k:
                    break
            if len(reranked) >= top_k:
                break

    contexts = [_format_chunk(chunk) for chunk in reranked[:top_k]]
    set_cached_json(cache_key, contexts, settings.rag_cache_ttl_seconds)
    return contexts


def rag_context_summary(contexts: list[str]) -> str:
    """Build a compact prompt block from retrieved snippets."""
    if not contexts:
        return "暂无本地攻略上下文，请优先使用高德 MCP 工具返回的实时信息。"
    return "\n\n".join(contexts)
