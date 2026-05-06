"""CLI script for running NaijaSense evaluation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task A/Task B evaluation.")
    parser.add_argument(
        "--dataset",
        default="data/sample_evaluation_dataset.json",
        help="Path to evaluation dataset JSON file.",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k for ranking metrics.")
    args = parser.parse_args()

    results = run_evaluation(dataset_path=args.dataset, k=args.k)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

