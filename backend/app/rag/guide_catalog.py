"""Canonical destination registry for local guide files."""

from __future__ import annotations

from pathlib import Path


GUIDE_DESTINATIONS = {
    "beijing_guide.md": "北京",
    "chengdu_guide.md": "成都",
    "dali_guide.md": "大理",
    "shanghai_guide.md": "上海",
    "xian_guide.md": "西安",
}

CITY_ALIASES = {
    "北京市": "北京",
    "北京": "北京",
    "成都市": "成都",
    "成都": "成都",
    "大理市": "大理",
    "大理": "大理",
    "上海市": "上海",
    "上海": "上海",
    "西安市": "西安",
    "西安": "西安",
}

CITY_CENTERS = {
    "北京": (116.4074, 39.9042),
    "上海": (121.4737, 31.2304),
    "成都": (104.0665, 30.5728),
    "大理": (100.2676, 25.6065),
    "西安": (108.9398, 34.3416),
}


def destination_for_guide(source_name: str) -> str | None:
    """Return the canonical destination for a guide filename."""
    return GUIDE_DESTINATIONS.get(Path(source_name).name)


def canonical_destination(destination: str | None) -> str | None:
    """Normalize user input such as ``北京市`` to ``北京`` when possible."""
    if not destination:
        return None

    value = destination.strip()
    if value in CITY_ALIASES:
        return CITY_ALIASES[value]

    for alias, canonical in CITY_ALIASES.items():
        if value.startswith(alias) and value[len(alias):] in {"", "市", "地区"}:
            return canonical

    return value


def known_destinations() -> set[str]:
    """Return all maintained destinations."""
    return set(GUIDE_DESTINATIONS.values())


def city_center(destination: str | None) -> tuple[float, float] | None:
    """Return the longitude and latitude for a supported destination."""
    canonical = canonical_destination(destination)
    return CITY_CENTERS.get(canonical or "")
