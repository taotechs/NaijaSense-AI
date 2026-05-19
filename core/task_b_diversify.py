"""Stage-1 pool diversification for Task B (cross-domain spread)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from core.candidate_catalog import CatalogItem


def diversify_stage1_pool(
    pool: Sequence[Tuple[CatalogItem, float]],
    *,
    limit: int,
    persona_domains: Sequence[str],
    min_unique_domains: int = 3,
) -> List[Tuple[CatalogItem, float]]:
    """
    Re-order stage-1 hits so the LLM sees variety across domains (food, movies, etc.).
    Preserves high scores within each domain via round-robin selection.
    """
    if not pool:
        return []
    if len(pool) <= limit and len({(i.domain or "").lower() for i, _ in pool}) >= min_unique_domains:
        return list(pool[:limit])

    buckets: Dict[str, List[Tuple[CatalogItem, float]]] = {}
    for item, score in sorted(pool, key=lambda x: x[1], reverse=True):
        dom = (item.domain or "general").strip().lower()
        buckets.setdefault(dom, []).append((item, score))

    priority = [d.lower() for d in persona_domains if d]
    for dom in buckets:
        if dom not in priority:
            priority.append(dom)

    picked: List[Tuple[CatalogItem, float]] = []
    seen: Set[str] = set()

    while len(picked) < limit:
        added = False
        for dom in priority:
            items = buckets.get(dom) or []
            while items and items[0][0].item_id in seen:
                items.pop(0)
            if not items:
                continue
            item, score = items.pop(0)
            seen.add(item.item_id)
            picked.append((item, score))
            added = True
            if len(picked) >= limit:
                break
        if not added:
            break

    if len(picked) < limit:
        for item, score in sorted(pool, key=lambda x: x[1], reverse=True):
            if item.item_id in seen:
                continue
            picked.append((item, score))
            seen.add(item.item_id)
            if len(picked) >= limit:
                break

    return picked[:limit]
