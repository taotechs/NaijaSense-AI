"""Build a normalized review corpus from available public datasets."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAGGLE_AMAZON_SLUG = "yacharki/amazon-reviews-for-sa-binary-negative-positive-csv"
DEFAULT_KAGGLE_YELP_SLUG = "luisfredgs/yelp-reviews-csv"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.normalize import normalize_amazon, normalize_goodreads, normalize_yelp
from data_pipeline.schema import NormalizedReviewRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized review corpus JSONL.")
    parser.add_argument("--output", default="data/processed/review_corpus.jsonl")
    parser.add_argument("--limit", type=int, default=500, help="Rows per source (best effort).")
    parser.add_argument("--use_hf", action="store_true", help="Pull datasets from HuggingFace hub.")
    parser.add_argument(
        "--hf_sources",
        default="yelp,amazon",
        help="When using --use_hf: comma-separated subset of yelp, amazon (skips large Yelp download if you pass amazon only).",
    )
    parser.add_argument(
        "--extra_jsonl",
        action="append",
        default=None,
        help="JSONL file of NormalizedReviewRecord rows (repeat flag for multiple files). Works offline.",
    )
    parser.add_argument(
        "--use_kaggle",
        action="store_true",
        help="Download CSV datasets from Kaggle (needs API credentials and dataset access on kaggle.com).",
    )
    parser.add_argument(
        "--kaggle_sources",
        default="amazon,yelp",
        help="With --use_kaggle: comma-separated subset of amazon, yelp.",
    )
    parser.add_argument("--kaggle_cache", default="data/raw/kaggle", help="Directory for downloaded Kaggle files.")
    parser.add_argument(
        "--kaggle_force_download",
        action="store_true",
        help="Re-download Kaggle datasets even if a CSV cache already exists.",
    )
    parser.add_argument(
        "--kaggle_amazon_slug",
        default=DEFAULT_KAGGLE_AMAZON_SLUG,
        help="Kaggle dataset slug (owner/name) for Amazon-style reviews.",
    )
    parser.add_argument(
        "--kaggle_yelp_slug",
        default=DEFAULT_KAGGLE_YELP_SLUG,
        help="Kaggle dataset slug (owner/name) for Yelp-style reviews.",
    )
    parser.add_argument(
        "--kaggle_amazon_dir",
        default=None,
        help="Folder already containing Amazon CSV(s) (e.g. manual browser download). No Kaggle API used.",
    )
    parser.add_argument(
        "--kaggle_yelp_dir",
        default=None,
        help="Folder already containing Yelp CSV(s) (e.g. manual browser download). No Kaggle API used.",
    )
    args = parser.parse_args()

    records: List[NormalizedReviewRecord] = []
    for path_str in args.extra_jsonl or []:
        records.extend(_load_extra_jsonl(Path(path_str)))

    hf_keys = {s.strip().lower() for s in args.hf_sources.split(",") if s.strip()} & {"yelp", "amazon"}
    if args.use_hf:
        if not hf_keys:
            print("No valid --hf_sources (use yelp and/or amazon); skipping HuggingFace ingest.")
        else:
            records.extend(_fetch_hf_records(limit=args.limit, hf_sources=hf_keys))

    kg_keys = {s.strip().lower() for s in args.kaggle_sources.split(",") if s.strip()} & {"amazon", "yelp"}
    if args.use_kaggle:
        if not kg_keys:
            print("No valid --kaggle_sources (use amazon and/or yelp); skipping Kaggle ingest.")
        else:
            records.extend(
                _fetch_kaggle_records(
                    limit=args.limit,
                    kaggle_sources=kg_keys,
                    cache=Path(args.kaggle_cache),
                    amazon_slug=args.kaggle_amazon_slug.strip(),
                    yelp_slug=args.kaggle_yelp_slug.strip(),
                    force_download=args.kaggle_force_download,
                    amazon_dir=Path(args.kaggle_amazon_dir) if args.kaggle_amazon_dir else None,
                    yelp_dir=Path(args.kaggle_yelp_dir) if args.kaggle_yelp_dir else None,
                )
            )

    # Always append tiny local seed corpus to ensure non-empty behavior.
    records.extend(_local_seed_records())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} normalized records to {output_path}")


def _load_extra_jsonl(path: Path) -> List[NormalizedReviewRecord]:
    if not path.is_file():
        print(f"Extra JSONL not found, skipping: {path}")
        return []
    out: List[NormalizedReviewRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(NormalizedReviewRecord.model_validate(json.loads(line)))
            except Exception:
                continue
    print(f"Loaded {len(out)} records from {path}")
    return out


def _hf_stream_rows(dataset_name: str, split: str, limit: int) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    yield from itertools.islice(ds, limit)


def _coerce_yelp_hf_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    r = dict(row)
    if "stars" not in r and "label" in r:
        r["stars"] = float(r["label"]) + 1.0
    r.setdefault("user_id", f"hf_yelp_{idx}")
    r.setdefault("business_id", f"hf_yelp_{idx}")
    r.setdefault("name", "yelp_review")
    return r


def _coerce_amazon_hf_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    r = dict(row)
    if not str(r.get("text", "")).strip() and r.get("content"):
        r["text"] = r["content"]
    if "rating" not in r and "overall" not in r and "label" in r:
        r["rating"] = 5.0 if int(r["label"]) == 1 else 1.0
    r.setdefault("user_id", f"hf_amazon_{idx}")
    r.setdefault("asin", f"hf_amazon_{idx}")
    r.setdefault("title", str(r.get("title", "amazon_review")))
    r.setdefault("category", "general")
    return r


def _fetch_hf_records(limit: int, hf_sources: set[str]) -> List[NormalizedReviewRecord]:
    try:
        import datasets  # noqa: F401
    except Exception:
        print("datasets package unavailable; skipping HuggingFace ingest.")
        return []

    out: List[NormalizedReviewRecord] = []
    specs: list[
        tuple[str, str, str, Callable[[Dict[str, Any], int], Dict[str, Any]], Callable[[Dict[str, Any]], NormalizedReviewRecord]]
    ] = [
        ("yelp", "yelp_review_full", "train", _coerce_yelp_hf_row, normalize_yelp),
        ("amazon", "amazon_polarity", "train", _coerce_amazon_hf_row, normalize_amazon),
    ]
    for key, dataset_name, split, coerce, normalizer in specs:
        if key not in hf_sources:
            continue
        n_before = len(out)
        try:
            for idx, row in enumerate(_hf_stream_rows(dataset_name, split, limit)):
                try:
                    rec = normalizer(coerce(dict(row), idx))
                    if rec.text:
                        out.append(rec)
                except Exception:
                    continue
            print(f"Ingested {dataset_name}: {len(out) - n_before} records (streaming).")
        except Exception:
            print(f"Could not ingest {dataset_name}; continuing.")
            continue

    return out


def _slug_cache_dir(cache_root: Path, slug: str) -> Path:
    safe = slug.replace("/", "_").replace("\\", "_")
    return cache_root / safe


def _kaggle_dataset_has_csv(dir_path: Path) -> bool:
    return dir_path.is_dir() and any(dir_path.rglob("*.csv"))


def _ensure_kaggle_dataset(slug: str, cache_dir: Path, force_download: bool) -> bool:
    if "/" not in slug or not slug.strip():
        print(f"Invalid Kaggle slug (expected owner/name): {slug!r}")
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not force_download and _kaggle_dataset_has_csv(cache_dir):
        return True
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print(
            "kaggle package unavailable; pip install kaggle and configure ~/.kaggle/kaggle.json, "
            "or unzip a manual download into "
            f"{cache_dir} (see --kaggle_amazon_dir / --kaggle_yelp_dir)."
        )
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(cache_dir), unzip=True, quiet=False)
    except Exception as exc:
        print(f"Kaggle download failed for {slug}: {exc}")
        return False
    if not _kaggle_dataset_has_csv(cache_dir):
        print(f"No CSV files found after download for {slug} (check dataset layout).")
        return False
    return True


def _pick_primary_csv(root: Path) -> Optional[Path]:
    files = [p for p in root.rglob("*.csv") if p.is_file()]
    if not files:
        return None
    train = [f for f in files if "train" in f.name.lower()]
    pool = train if train else files
    return max(pool, key=lambda p: p.stat().st_size)


def _read_csv_rows(path: Path, limit: int) -> List[Dict[str, Any]]:
    import pandas as pd

    last_err: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, nrows=limit, encoding=encoding, on_bad_lines="skip")
            if df.empty:
                return []
            return df.to_dict(orient="records")
        except Exception as exc:
            last_err = exc
            continue
    print(f"Could not read CSV {path}: {last_err}")
    return []


def _find_column(columns: Sequence[str], *candidates: str) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _binary_label_to_rating(raw: Any) -> float:
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return 3.0
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in ("positive", "pos", "good", "true"):
            return 5.0
        if v in ("negative", "neg", "bad", "false"):
            return 1.0
    try:
        f = float(raw)
    except Exception:
        return 3.0
    if 1.0 <= f <= 5.0:
        return float(f)
    n = int(f)
    if n in (0, 1):
        return 5.0 if n == 1 else 1.0
    if n in (1, 2):
        return 5.0 if n == 2 else 1.0
    if n in (-1, 1):
        return 5.0 if n == 1 else 1.0
    return 3.0


def _stars_from_value(raw: Any) -> float:
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return 3.0
    try:
        s = float(raw)
    except Exception:
        return 3.0
    if 0.0 <= s <= 4.0 and abs(s - round(s)) < 1e-9:
        s = s + 1.0
    return max(1.0, min(5.0, s))


def _ingest_amazon_csv_folder(dest: Path, limit: int, label: str) -> List[NormalizedReviewRecord]:
    out: List[NormalizedReviewRecord] = []
    csv_path = _pick_primary_csv(dest)
    if not csv_path:
        print(f"No CSV found under {dest} ({label}).")
        return out
    rows = _read_csv_rows(csv_path, limit)
    for idx, row in enumerate(rows):
        try:
            rec = _kaggle_row_to_amazon(row, idx)
            if rec.text:
                out.append(rec)
        except Exception:
            continue
    print(f"Ingested {label}: {len(out)} records from {csv_path.name}.")
    return out


def _ingest_yelp_csv_folder(dest: Path, limit: int, label: str) -> List[NormalizedReviewRecord]:
    out: List[NormalizedReviewRecord] = []
    csv_path = _pick_primary_csv(dest)
    if not csv_path:
        print(f"No CSV found under {dest} ({label}).")
        return out
    rows = _read_csv_rows(csv_path, limit)
    for idx, row in enumerate(rows):
        try:
            rec = _kaggle_row_to_yelp(row, idx)
            if rec.text:
                out.append(rec)
        except Exception:
            continue
    print(f"Ingested {label}: {len(out)} records from {csv_path.name}.")
    return out


def _fetch_kaggle_records(
    limit: int,
    kaggle_sources: set[str],
    cache: Path,
    amazon_slug: str,
    yelp_slug: str,
    force_download: bool,
    amazon_dir: Optional[Path],
    yelp_dir: Optional[Path],
) -> List[NormalizedReviewRecord]:
    out: List[NormalizedReviewRecord] = []
    if "amazon" in kaggle_sources:
        if amazon_dir is not None:
            dest = amazon_dir.expanduser().resolve()
            if _kaggle_dataset_has_csv(dest):
                out.extend(_ingest_amazon_csv_folder(dest, limit, f"Amazon CSV ({dest})"))
            else:
                print(f"No CSV under --kaggle_amazon_dir {dest}")
        else:
            dest = _slug_cache_dir(cache, amazon_slug)
            if _ensure_kaggle_dataset(amazon_slug, dest, force_download):
                out.extend(_ingest_amazon_csv_folder(dest, limit, f"Kaggle Amazon ({amazon_slug})"))
    if "yelp" in kaggle_sources:
        if yelp_dir is not None:
            dest = yelp_dir.expanduser().resolve()
            if _kaggle_dataset_has_csv(dest):
                out.extend(_ingest_yelp_csv_folder(dest, limit, f"Yelp CSV ({dest})"))
            else:
                print(f"No CSV under --kaggle_yelp_dir {dest}")
        else:
            dest = _slug_cache_dir(cache, yelp_slug)
            if _ensure_kaggle_dataset(yelp_slug, dest, force_download):
                out.extend(_ingest_yelp_csv_folder(dest, limit, f"Kaggle Yelp ({yelp_slug})"))
    return out


def _kaggle_row_to_amazon(row: Dict[str, Any], idx: int) -> NormalizedReviewRecord:
    cols = [str(k) for k in row.keys()]
    text_col = _find_column(cols, "text", "review", "content", "review_text", "body", "sentence")
    if not text_col:
        raise ValueError("no text column")
    label_col = _find_column(cols, "label", "sentiment", "polarity", "class", "target", "rating")
    title_col = _find_column(cols, "title", "summary", "product_title", "headline")
    text = str(row.get(text_col, "")).strip()
    rating = _binary_label_to_rating(row.get(label_col)) if label_col else 3.0
    title = str(row[title_col]).strip() if title_col and row.get(title_col) is not None else "amazon_review"
    return normalize_amazon(
        {
            "user_id": f"kg_amazon_{idx}",
            "asin": f"kg_amazon_{idx}",
            "title": title or "amazon_review",
            "text": text,
            "rating": rating,
            "category": "general",
        }
    )


def _kaggle_row_to_yelp(row: Dict[str, Any], idx: int) -> NormalizedReviewRecord:
    cols = [str(k) for k in row.keys()]
    text_col = _find_column(cols, "text", "review", "content", "review_text", "body")
    if not text_col:
        raise ValueError("no text column")
    stars_col = _find_column(cols, "stars", "star_rating", "rating", "label")
    text = str(row.get(text_col, "")).strip()
    stars = _stars_from_value(row.get(stars_col)) if stars_col else 3.0
    name_col = _find_column(cols, "name", "business_name", "business")
    name = str(row[name_col]).strip() if name_col and row.get(name_col) is not None else "yelp_review"
    return normalize_yelp(
        {
            "user_id": f"kg_yelp_{idx}",
            "business_id": f"kg_yelp_{idx}",
            "name": name or "yelp_review",
            "text": text,
            "stars": stars,
        }
    )


def _local_seed_records() -> List[NormalizedReviewRecord]:
    return [
        normalize_yelp(
            {
                "user_id": "y_seed_1",
                "business_id": "b_amala_1",
                "name": "Iya Aladura Amala Spot",
                "text": "Amala and gbegiri were rich and tasty, portion was fair.",
                "stars": 4.0,
            }
        ),
        normalize_yelp(
            {
                "user_id": "y_seed_2",
                "business_id": "b_buka_2",
                "name": "Budget Buka",
                "text": "Affordable meal but service was slow and soup arrived cold.",
                "stars": 2.0,
            }
        ),
        normalize_amazon(
            {
                "user_id": "a_seed_1",
                "asin": "earbud_1",
                "title": "Budget Earbuds",
                "text": "Battery life impressed me and bass was clean for the price.",
                "rating": 4.5,
                "category": "tech",
            }
        ),
        normalize_goodreads(
            {
                "user_id": "gr_u_1",
                "book_id": "gr_b_1",
                "title": "Half of a Yellow Sun",
                "review_text": "Emotional story with powerful character development.",
                "rating": 5,
            }
        ),
    ]


if __name__ == "__main__":
    main()

