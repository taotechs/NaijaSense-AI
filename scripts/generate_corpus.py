"""
Generate a local structured JSON corpus at data/large_corpus.json.

3,000 rows total (1,000 per source domain):
  - Yelp:   food/drink venues with location, price_tier, sentiment tags
  - Amazon: tech/utilities products with durability, price
  - Goodreads: books/media with narrative_style, thematic tags

Each record includes normalized fields (item_name, text, rating, tags) for Stage-1
semantic retrieval plus a ``domain_record`` block with source-specific metadata.

Usage:
  python scripts/generate_corpus.py
  python scripts/generate_corpus.py --output data/large_corpus.json --per-domain 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "large_corpus.json"

# --- Yelp seeds (Nigerian lifestyle) ------------------------------------------

_YELP_FOOD_NAMES = [
    "Iya Eba Amala Spot - Yaba",
    "Mama Put Garri & Soup - Mushin",
    "Late-night Suya & Chill - Ikeja",
    "Local Jollof Kitchen - Surulere",
    "Shawarma Alley - Victoria Island",
    "Pepper Soup & Catfish Grill - Lekki",
    "Campus Canteen Jollof - UNILAG",
    "Akara & Pap Corner - Yaba",
    "Buka Hop - Lagos mainland",
    "Grilled Plantain & Fish - PH GRA",
]

_YELP_DRINK_NAMES = [
    "Smoothie Bar - Wuse, Abuja",
    "Chapman & Mocktail Lounge - VI",
    "Palm Wine Terrace - Lekki",
    "Kunu & Zobo Stand - Kaduna Road",
    "Specialty Coffee Hut - Ikeja",
    "Fresh Juice Cart - Yaba market",
    "Afrobeats Rooftop Bar - Lagos Island",
    "Tea & Pastry Corner - Surulere",
]

_YELP_LOCATIONS = [
    "Yaba, Lagos",
    "Ikeja, Lagos",
    "Victoria Island, Lagos",
    "Surulere, Lagos",
    "Lekki, Lagos",
    "Mushin, Lagos",
    "Wuse, Abuja",
    "Port Harcourt GRA",
    "UNILAG campus, Lagos",
    "Mainland Lagos",
]

_YELP_SENTIMENT_TAGS = [
    "value_for_money",
    "spicy_lover",
    "quick_service",
    "student_budget",
    "social_hangout",
    "late_night",
    "portion_generous",
    "slow_service",
    "weekend_regular",
    "founder_team_lunch",
]

_YELP_REVIEW_TEMPLATES = [
    "Paid about ₦{price} - {sentiment} The {category_lower} vibe at {location} fits a {tag} crowd.",
    "{business} in {location}: jollof/smoke level was {quality}. Tags I'd use: {tags}.",
    "Honest buka run - {sentiment} Portion and wait time felt right for {price_tier} tier around {location}.",
    "Weekend outing spot; {sentiment} Would bring friends who care about {tag} and local flavour.",
]

# --- Amazon seeds -------------------------------------------------------------

_AMAZON_TECH = [
    ("Wireless Earbuds Pro", 89.99, 4.2),
    ("USB-C Power Bank 20Ah", 45.00, 4.5),
    ("Mechanical Keyboard 60%", 120.00, 4.3),
    ("4G MiFi Router - data saver", 65.00, 4.0),
    ("Compact Bluetooth Keyboard", 55.00, 4.1),
    ("Noise-cancelling Headphones", 149.00, 4.4),
    ("Laptop USB-C Hub 7-in-1", 38.00, 4.2),
    ("Fitness Band - sleep tracking", 42.00, 3.9),
    ("Portable SSD 1TB", 95.00, 4.6),
    ("Ring light for video calls", 28.00, 4.0),
]

_AMAZON_UTILITIES = [
    ("LED Desk Lamp - adjustable", 32.00, 4.3),
    ("Surge Protector 6-outlet", 24.00, 4.5),
    ("Reusable Water Bottle 1L", 18.00, 4.4),
    ("Electric Kettle - fast boil", 40.00, 4.2),
    ("Extension Cord 5m", 15.00, 4.1),
    ("Storage Organiser Set", 22.00, 4.0),
    ("Travel Adapter Universal", 19.00, 4.3),
    ("Desk Fan - quiet mode", 35.00, 3.8),
]

_AMAZON_REVIEW_TEMPLATES = [
    "Durability rated {durability}/5 in daily Lagos use. {sentiment} Price ₦{price_naira} felt {price_verdict}.",
    "{product}: build quality {quality}. Good for remote work / commute; tags: {tags}.",
    "After {weeks} weeks - still holds charge/structure. {sentiment} Category: {category}.",
]

# --- Goodreads seeds ----------------------------------------------------------

_GOODREADS_BOOKS = [
    ("Things Fall Apart", "sparse_literary", ["colonialism", "igbo_culture", "classic"]),
    ("Half of a Yellow Sun", "epic_historical", ["biafra", "war", "family"]),
    ("Americanah", "contemporary_realist", ["migration", "identity", "lagos_diaspora"]),
    ("Welcome to Lagos", "urban_ensemble", ["lagos", "friendship", "city_life"]),
    ("Born on a Tuesday", "coming_of_age", ["north_nigeria", "religion", "youth"]),
    ("Purple Hibiscus", "intimate_family", ["abuse", "catholicism", "growth"]),
    ("Stay With Me", "marriage_drama", ["infertility", "yoruba", "secrets"]),
    ("Freshwater", "lyrical_speculative", ["identity", "spirituality", "lgbtq"]),
    ("Lagoon", "speculative_lagos", ["sci_fi", "lagos", "afrofuturism"]),
    ("Every Day Is for the Thief", "travelogue", ["lagos", "observation", "nonfiction_tone"]),
]

_MEDIA_TITLES = [
    ("Nollywood Weekend Drama - streaming pick", "cinematic", ["nollywood", "weekend", "family"]),
    ("Afrobeats Live Session Recording", "musical", ["afrobeats", "concert", "energy"]),
    ("African History Documentary Series", "documentary", ["history", "education", "culture"]),
    ("Stand-up Comedy Special - Lagos", "humorous", ["comedy", "social", "night_out"]),
]

_GOODREADS_TEMPLATES = [
    "Narrative style: {style}. Themes: {themes}. {sentiment} Pacing suited a {mood} weekend read.",
    "{title} - {sentiment} The {style} voice and tags {tags} match readers into {theme_primary}.",
    "Lagos/Nigeria context: {context}. {sentiment} Would recommend for {tag} lovers.",
]


def _price_tier_from_naira(naira: int) -> str:
    if naira <= 2500:
        return "budget"
    if naira >= 8000:
        return "premium"
    return "mid"


def _stars_from_sentiment(sentiment: str) -> float:
    if sentiment in ("glowing", "enthusiastic", "highly_positive"):
        return round(random.uniform(4.3, 5.0), 1)
    if sentiment in ("mixed", "balanced"):
        return round(random.uniform(3.2, 4.1), 1)
    return round(random.uniform(1.8, 3.0), 1)


def _pick_sentiment(rng: random.Random) -> str:
    return rng.choice(
        ["glowing", "enthusiastic", "balanced", "mixed", "critical", "highly_positive"]
    )


def generate_yelp_rows(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        is_drink = i % 5 == 0
        category = "Drinks" if is_drink else "Food"
        pool = _YELP_DRINK_NAMES if is_drink else _YELP_FOOD_NAMES
        business_name = pool[i % len(pool)] + (f" #{i // len(pool) + 1}" if i >= len(pool) else "")
        location = rng.choice(_YELP_LOCATIONS)
        price_tier = rng.choice(["budget", "mid", "premium"])
        naira = rng.choice([1200, 1800, 2500, 3500, 5000, 8000, 12000])
        if price_tier == "budget":
            naira = min(naira, 3000)
        elif price_tier == "premium":
            naira = max(naira, 6000)

        sentiment_tags = rng.sample(
            _YELP_SENTIMENT_TAGS, k=rng.randint(2, 4)
        )
        sentiment = _pick_sentiment(rng)
        quality = rng.choice(["solid", "average", "excellent", "inconsistent"])
        tpl = rng.choice(_YELP_REVIEW_TEMPLATES)
        text = tpl.format(
            price=naira,
            sentiment=sentiment,
            category_lower=category.lower(),
            location=location,
            tag=rng.choice(sentiment_tags),
            business=business_name,
            quality=quality,
            price_tier=price_tier,
            tags=", ".join(sentiment_tags),
        )

        item_domain = "drinks" if is_drink else "food"
        row: Dict[str, Any] = {
            "source": "yelp",
            "user_id": f"yelp_user_{i:04d}",
            "item_id": f"yelp_{item_domain}_{i:05d}",
            "item_name": business_name,
            "item_domain": item_domain,
            "text": text,
            "rating": _stars_from_sentiment(sentiment),
            "price_tier": price_tier,
            "tags": sentiment_tags + [category.lower(), location.split(",")[0].lower()],
            "domain_record": {
                "item_id": f"yelp_{item_domain}_{i:05d}",
                "business_name": business_name,
                "category": category,
                "price_tier": price_tier,
                "location": location,
                "historical_user_sentiment_tags": sentiment_tags,
            },
        }
        rows.append(row)
    return rows


def generate_amazon_rows(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        is_tech = i % 3 != 2
        category = "Tech" if is_tech else "Utilities"
        pool = _AMAZON_TECH if is_tech else _AMAZON_UTILITIES
        product_name, usd_price, durability_rating = pool[i % len(pool)]
        if i >= len(pool):
            product_name = f"{product_name} (variant {i // len(pool) + 1})"

        price_usd = round(usd_price * rng.uniform(0.9, 1.15), 2)
        price_naira = int(price_usd * rng.choice([1450, 1550, 1600]))
        durability = round(
            min(5.0, max(1.0, durability_rating + rng.uniform(-0.3, 0.3))), 1
        )
        sentiment = _pick_sentiment(rng)
        quality = rng.choice(["sturdy", "decent", "premium-feel", "flimsy"])
        tags = [
            category.lower(),
            "durability_" + str(int(durability)),
            "remote_work" if is_tech else "home_essentials",
            "lagos_commute" if is_tech and rng.random() > 0.5 else "student_budget",
        ]

        tpl = rng.choice(_AMAZON_REVIEW_TEMPLATES)
        text = tpl.format(
            durability=durability,
            sentiment=sentiment,
            price_naira=price_naira,
            price_verdict=rng.choice(["fair", "steep", "a steal"]),
            product=product_name,
            quality=quality,
            weeks=rng.randint(2, 12),
            category=category,
            tags=", ".join(tags),
        )

        item_domain = "tech" if is_tech else "services"
        row = {
            "source": "amazon",
            "user_id": f"amazon_user_{i:04d}",
            "item_id": f"amazon_{category.lower()}_{i:05d}",
            "item_name": product_name,
            "item_domain": item_domain,
            "text": text,
            "rating": _stars_from_sentiment(sentiment),
            "price_tier": _price_tier_from_naira(price_naira),
            "tags": tags,
            "domain_record": {
                "item_id": f"amazon_{category.lower()}_{i:05d}",
                "product_name": product_name,
                "category": category,
                "durability_rating": durability,
                "price": {"usd": price_usd, "ngn": price_naira},
            },
        }
        rows.append(row)
    return rows


def generate_goodreads_rows(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        is_media = i % 4 == 3
        category = "Media" if is_media else "Books"
        pool = _MEDIA_TITLES if is_media else _GOODREADS_BOOKS
        title, narrative_style, thematic_tags = pool[i % len(pool)]
        if i >= len(pool) and not is_media:
            title = f"{title} - reader edition {i // len(pool) + 1}"

        sentiment = _pick_sentiment(rng)
        mood = rng.choice(["slow", "page-turner", "reflective", "weekend binge"])
        tpl = rng.choice(_GOODREADS_TEMPLATES)
        text = tpl.format(
            style=narrative_style.replace("_", " "),
            themes=", ".join(thematic_tags),
            sentiment=sentiment,
            mood=mood,
            title=title,
            tags=", ".join(thematic_tags),
            theme_primary=thematic_tags[0],
            context=rng.choice(["Lagos", "Nigeria", "diaspora", "West Africa"]),
            tag=rng.choice(thematic_tags),
        )

        item_domain = "books" if category == "Books" else "movies"
        row = {
            "source": "goodreads",
            "user_id": f"goodreads_user_{i:04d}",
            "item_id": f"gr_{category.lower()}_{i:05d}",
            "item_name": title,
            "item_domain": item_domain,
            "text": text,
            "rating": _stars_from_sentiment(sentiment),
            "price_tier": rng.choice(["budget", "mid"]),
            "tags": thematic_tags + [narrative_style, category.lower()],
            "domain_record": {
                "item_id": f"gr_{category.lower()}_{i:05d}",
                "book_title": title,
                "category": category,
                "narrative_style": narrative_style,
                "thematic_tags": thematic_tags,
            },
        }
        rows.append(row)
    return rows


def build_corpus(*, per_domain: int = 1000, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    yelp = generate_yelp_rows(per_domain, rng)
    amazon = generate_amazon_rows(per_domain, rng)
    goodreads = generate_goodreads_rows(per_domain, rng)
    records = yelp + amazon + goodreads
    rng.shuffle(records)

    return {
        "corpus_version": "1.0",
        "description": "NaijaSense synthetic large corpus - Yelp / Amazon / Goodreads domains",
        "total_rows": len(records),
        "domains": {
            "yelp": len(yelp),
            "amazon": len(amazon),
            "goodreads": len(goodreads),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data/large_corpus.json (3000 rows).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-domain", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Also write data/processed/corpus_index.json for fast Stage-1 lookup.",
    )
    args = parser.parse_args()

    payload = build_corpus(per_domain=args.per_domain, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {payload['total_rows']} rows -> {out}")
    print(f"  Domains: {payload['domains']}")

    if args.build_index:
        from scripts.build_large_corpus import build_index
        from utils.config import settings

        index_path = PROJECT_ROOT / settings.corpus_index_path
        build_index(out, index_path)
        print(f"  Index -> {index_path}")


if __name__ == "__main__":
    main()
