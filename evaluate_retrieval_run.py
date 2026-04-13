from __future__ import annotations

import argparse
from pathlib import Path

from chunked_pooling.retrieval_evaluation import (
    DEFAULT_EVALUATION_ROOT,
    DEFAULT_K_VALUES,
    LABEL_SOURCE_CHOICES,
    evaluate_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an existing late-chunking retrieval run and write compact, comparable metrics artifacts."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Path to the existing retrieval run directory.")
    parser.add_argument("--labels-file", required=True, help="Path to the labels/relevance JSON or JSONL file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional override for the evaluation output directory. "
            f"By default, the script mirrors the run path under {DEFAULT_EVALUATION_ROOT}/..."
        ),
    )
    parser.add_argument("--method-name", default=None, help="Method label to write into the output artifacts.")
    parser.add_argument("--dataset-name", default=None, help="Dataset label to write into the output artifacts.")
    parser.add_argument("--split", default=None, help="Dataset split label to write into the output artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional override for the run name recorded in outputs.")
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="k values to evaluate, for example: --ks 5 10",
    )
    parser.add_argument(
        "--primary-relevance",
        choices=LABEL_SOURCE_CHOICES,
        default="auto",
        help="Primary relevance field to use. Defaults to auto.",
    )
    parser.add_argument(
        "--raw-results-file",
        default=None,
        help="Optional explicit raw retrieval file inside or outside the run directory.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = evaluate_run(
        run_dir=Path(args.run_dir),
        labels_file=Path(args.labels_file),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        method_name=args.method_name,
        dataset_name=args.dataset_name,
        split=args.split,
        ks=args.ks,
        run_name=args.run_name,
        primary_relevance=args.primary_relevance,
        raw_results_file=Path(args.raw_results_file) if args.raw_results_file else None,
    )
    print(result["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
