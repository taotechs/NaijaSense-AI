"""Structured candidate catalog for Task B stage-1 retrieval (top-30 pool)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Set


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    title: str
    domain: str
    tags: tuple[str, ...]


# 35+ localized items across food, tech, entertainment, wellness, fashion.
CATALOG: tuple[CatalogItem, ...] = (
    CatalogItem("food_iya_eba", "Iya Eba Amala Spot — Yaba", "food", ("amala", "egusi", "buka", "lagos")),
    CatalogItem("food_suya_ikeja", "Suya & Chill Stand — Ikeja", "food", ("suya", "street", "spicy", "meat")),
    CatalogItem("food_shawarma_vi", "Shawarma Alley — VI", "food", ("shawarma", "late-night", "quick")),
    CatalogItem("food_jollof_surulere", "Local Jollof Kitchen — Surulere", "food", ("jollof", "rice", "budget")),
    CatalogItem("food_akara_yaba", "Late-night Akara & Pap — Yaba", "food", ("akara", "breakfast", "student")),
    CatalogItem("food_buka_garri", "Mama Put Garri & Soup — Mushin", "food", ("garri", "soup", "affordable")),
    CatalogItem("food_pepper_lunch", "Pepper Lunch Express — Lekki", "food", ("pepper", "fast", "mall")),
    CatalogItem("food_smoothie_abuja", "Abuja Smoothie Bar — Wuse", "food", ("drinks", "healthy", "abuja")),
    CatalogItem("tech_powerbank", "Budget USB-C Power Bank 20Ah", "tech", ("power", "phone", "budget", "student")),
    CatalogItem("tech_earbuds", "Wireless Earbuds (wallet-friendly)", "tech", ("audio", "commute", "gadget")),
    CatalogItem("tech_hub", "USB-C Hub for Laptop", "tech", ("laptop", "work", "productivity")),
    CatalogItem("tech_router", "4G MiFi Router — data saver", "tech", ("internet", "remote", "data")),
    CatalogItem("tech_keyboard", "Compact Bluetooth Keyboard", "tech", ("typing", "student", "office")),
    CatalogItem("ent_nollywood", "Nollywood weekend drama pick", "entertainment", ("movie", "relax", "weekend")),
    CatalogItem("ent_afrobeats", "Afrobeats live lounge — Lagos Island", "entertainment", ("music", "social", "night")),
    CatalogItem("ent_comedy", "Stand-up comedy night — mainland", "entertainment", ("comedy", "friends", "social")),
    CatalogItem("ent_docu", "African history documentary series", "entertainment", ("learn", "culture", "home")),
    CatalogItem("book_cafe", "African lit paperback + café — Ikeja", "books", ("read", "cafe", "chill")),
    CatalogItem("well_tea", "Cozy tea corner for de-stress", "wellness", ("tea", "calm", "stress")),
    CatalogItem("well_spa", "Affordable spa hour — VI", "wellness", ("relax", "self-care", "treat")),
    CatalogItem("well_yoga", "Community yoga in the park", "wellness", ("fitness", "morning", "health")),
    CatalogItem("fashion_ankara", "Street-style Ankara pop-up", "fashion", ("ankara", "style", "event")),
    CatalogItem("fashion_thrift", "Yaba thrift market haul guide", "fashion", ("budget", "student", "shopping")),
    CatalogItem("travel_buka", "Weekend buka hopping guide — Lagos", "experiences", ("food", "weekend", "local")),
    CatalogItem("travel_10k_abuja", "Abuja weekend on 10k — curated list", "experiences", ("abuja", "budget", "weekend")),
    CatalogItem("svc_data", "Mobile data bundle saver plan", "services", ("data", "mtn", "airtel", "budget")),
    CatalogItem("svc_laundry", "Express laundry pickup — campus area", "services", ("student", "convenience")),
    CatalogItem("food_catfish", "Grilled catfish & plantain — VI", "food", ("seafood", "dinner", "social")),
    CatalogItem("food_vegan", "Plant-based bowl kitchen — Lekki", "food", ("vegan", "health", "trend")),
    CatalogItem("ent_board", "Board-game café — Surulere", "entertainment", ("games", "friends", "indoor")),
    CatalogItem("tech_watch", "Fitness band (step & sleep tracking)", "tech", ("fitness", "health", "wearable")),
    CatalogItem("food_campus", "Campus canteen jollof special", "food", ("student", "campus", "cheap")),
    CatalogItem("ent_stream", "Local streaming bundle + movie night kit", "entertainment", ("netflix", "home", "movie")),
    CatalogItem("exp_night_market", "Evening market stroll + street food", "experiences", ("market", "street", "social")),
    CatalogItem("food_delivery", "Jollof delivery under 3k — Yaba", "food", ("delivery", "budget", "quick")),
)


def _terms(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_top_k(
    *,
    interests: Sequence[str],
    context: str | None,
    limit: int = 30,
    cold_start: bool = False,
    cross_domain: bool = False,
) -> List[tuple[CatalogItem, float]]:
    """
    Stage-1 semantic retrieval: score catalog rows, return top ``limit`` (default 30).
    """
    interest_terms = _terms(" ".join(interests))
    query_terms = _terms(context or "")
    query_terms |= interest_terms

    scored: List[tuple[CatalogItem, float]] = []
    for item in CATALOG:
        tag_terms = set(item.tags) | _terms(item.title)
        overlap = len(query_terms & tag_terms)
        domain_hit = 1.0 if any(t in item.domain for t in interest_terms) else 0.0
        score = overlap * 0.35 + domain_hit * 0.15 + 0.25

        if cold_start:
            # Demographic prior: popular localized staples for new users.
            if item.domain in ("food", "experiences", "services"):
                score += 0.35
            if "budget" in item.tags or "student" in item.tags:
                score += 0.15

        if cross_domain:
            # Map social/high-energy food prefs → entertainment experiences.
            if "social" in interest_terms or "street" in interest_terms:
                if item.domain in ("entertainment", "experiences"):
                    score += 0.25
            if "tech" in interest_terms and item.domain == "tech":
                score += 0.2
            if "food" in interest_terms and item.domain == "entertainment":
                score += 0.1  # cross-domain bridge

        scored.append((item, round(score, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]
