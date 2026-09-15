"""Extract grounded spot, meal, and hotel details from retrieved guide snippets."""

from __future__ import annotations

import re


_HEADING_PATTERN = re.compile(r"^\[来源:\s*.+?\s*\|\s*标题:\s*(?P<title>.+?)\]$")
_MARKDOWN_NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d+)*\s+")
_POSITION_PATTERN = re.compile(r"\*\*位置\*\*[:：]\s*(?P<address>.+)")
_MEAL_NAME_PATTERN = re.compile(r"【(?P<name>[^】]+)】[^\n]*(?:招牌菜|餐厅|小吃|火锅|美食)")
_HOTEL_NAME_PATTERN = re.compile(r"【(?P<name>[^】]+)】[^\n]*(?:酒店|民宿|客栈|宾馆|住宿)")


def _append_unique(candidates: list[str], value: str | None) -> None:
    normalized = (value or "").strip()
    if normalized and normalized not in candidates:
        candidates.append(normalized)


def _extract_spot_details(rag_contexts: list[str]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for context in rag_contexts:
        header, separator, body = context.partition("\n")
        if not separator:
            continue
        match = _HEADING_PATTERN.match(header.strip())
        if match is None:
            continue
        position = _POSITION_PATTERN.search(body)
        if position is None:
            continue

        name = _MARKDOWN_NUMBER_PATTERN.sub("", match.group("title")).strip()
        if not name or name in seen_names:
            continue

        description = ""
        for line in body.splitlines():
            cleaned = line.strip()
            if cleaned and cleaned != position.group(0):
                description = cleaned
                break

        candidates.append({
            "name": name,
            "address": position.group("address").strip(),
            "description": description,
        })
        seen_names.add(name)
    return candidates


def extract_fallback_candidates(rag_contexts: list[str]) -> dict[str, list]:
    """Return grounded candidates extracted from RAG contexts."""
    meals: list[str] = []
    hotels: list[str] = []
    for context in rag_contexts:
        for match in _MEAL_NAME_PATTERN.finditer(context):
            _append_unique(meals, match.group("name"))
        for match in _HOTEL_NAME_PATTERN.finditer(context):
            _append_unique(hotels, match.group("name"))

    return {
        "spots": _extract_spot_details(rag_contexts),
        "meals": meals,
        "hotels": hotels,
    }
