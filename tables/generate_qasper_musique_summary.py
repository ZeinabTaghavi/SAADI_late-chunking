#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


QASPER_DATASET = "qasper"
MUSIQUE_DATASETS = ("musique_2hop", "musique_3hop", "musique_4hop")
DEFAULT_RETRIEVERS = ("jina-v3", "qwen", "contriever", "bm25", "bge-m3")
RETRIEVER_LABELS = {
    "jina-v3": "Jina-v3",
    "qwen": "Qwen",
    "contriever": "Contriever",
    "bm25": "BM25",
    "bge-m3": "BGE-M3",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str


RANKING_METRICS = (
    MetricSpec("gold_ndcg@10", "NDCG@10"),
    MetricSpec("gold_recall@10", "Recall@10"),
    MetricSpec("silver_loose_ndcg@10", "NDCG@10"),
    MetricSpec("silver_loose_recall@10", "Recall@10"),
    MetricSpec("union_loose_ndcg@10", "NDCG@10"),
    MetricSpec("union_loose_recall@10", "Recall@10"),
)
BINARY_METRICS = (
    MetricSpec("gold_hit@5", "HR@5"),
    MetricSpec("gold_hit@10", "HR@10"),
    MetricSpec("silver_strict_hit@5", "HR@5"),
    MetricSpec("silver_strict_hit@10", "HR@10"),
    MetricSpec("strict_union_hit@5", "HR@5"),
    MetricSpec("strict_union_hit@10", "HR@10"),
)
ALL_METRICS = RANKING_METRICS + BINARY_METRICS


@dataclass(frozen=True)
class EvaluationSource:
    dataset: str
    retriever: str
    n_queries: int
    rows: Tuple[Dict[str, object], ...]
    summary_path: Path
    per_query_path: Path


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> Tuple[Dict[str, object], ...]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            rows.append(payload)
    return tuple(rows)


def _load_evaluation_source(
    input_root: Path,
    *,
    dataset: str,
    retriever: str,
    chunk_folder: str,
) -> EvaluationSource:
    evaluation_dir = input_root / dataset / retriever / chunk_folder
    summary_path = evaluation_dir / "metrics_summary.json"
    per_query_path = evaluation_dir / "metrics_per_query.jsonl"
    missing_paths = [
        path for path in (summary_path, per_query_path) if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing evaluation artifact(s):\n"
            + "\n".join(f"  {path}" for path in missing_paths)
        )

    summary = _read_json(summary_path)
    rows = _read_jsonl(per_query_path)
    declared_n_queries = summary.get("n_queries")
    if declared_n_queries is None:
        raise ValueError(f"Missing n_queries in {summary_path}")
    try:
        expected_n_queries = int(declared_n_queries)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid n_queries in {summary_path}") from exc
    if expected_n_queries != len(rows):
        raise ValueError(
            f"Query-count mismatch for {dataset}/{retriever}/{chunk_folder}: "
            f"{summary_path} declares {expected_n_queries}, but "
            f"{per_query_path} contains {len(rows)} rows."
        )

    return EvaluationSource(
        dataset=dataset,
        retriever=retriever,
        n_queries=len(rows),
        rows=rows,
        summary_path=summary_path,
        per_query_path=per_query_path,
    )


def _mean_metric(
    rows: Sequence[Dict[str, object]],
    metric_key: str,
) -> Tuple[Optional[float], int]:
    values: List[float] = []
    for row in rows:
        raw_value = row.get(metric_key)
        if raw_value is None:
            continue
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metric {metric_key!r} has a non-numeric value: {raw_value!r}"
            ) from exc
    if not values:
        return None, 0
    return sum(values) / float(len(values)), len(values)


def _chunk_method_label(chunk_folder: str) -> str:
    match = re.fullmatch(r"c(?P<size>\d+)_o(?P<overlap>\d+)", chunk_folder)
    if match is None:
        return f"Late Chunking ({chunk_folder})"
    return (
        "Late Chunking "
        f"({match.group('size')}/{match.group('overlap')})"
    )


def _relative_source_path(path: Path, input_root: Path) -> str:
    try:
        return str(path.relative_to(input_root))
    except ValueError:
        return str(path)


def _build_result_row(
    *,
    dataset_label: str,
    dataset_key: str,
    retriever: str,
    chunk_folder: str,
    sources: Sequence[EvaluationSource],
    input_root: Path,
    aggregation: str,
) -> Dict[str, object]:
    combined_rows = [
        row
        for source in sources
        for row in source.rows
    ]
    metrics: Dict[str, Optional[float]] = {}
    metric_counts: Dict[str, int] = {}
    for metric in ALL_METRICS:
        value, count = _mean_metric(combined_rows, metric.key)
        metrics[metric.key] = value
        metric_counts[metric.key] = count

    return {
        "dataset": dataset_key,
        "dataset_label": dataset_label,
        "retriever": retriever,
        "retriever_label": RETRIEVER_LABELS.get(retriever, retriever),
        "method": _chunk_method_label(chunk_folder),
        "chunk_folder": chunk_folder,
        "aggregation": aggregation,
        "n_queries": len(combined_rows),
        "source_query_counts": {
            source.dataset: source.n_queries for source in sources
        },
        "source_files": [
            _relative_source_path(source.per_query_path, input_root)
            for source in sources
        ],
        "metrics": metrics,
        "metric_counts": metric_counts,
    }


def build_report(
    input_root: Path,
    *,
    chunk_folder: str,
    retrievers: Sequence[str],
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for retriever in retrievers:
        qasper_source = _load_evaluation_source(
            input_root,
            dataset=QASPER_DATASET,
            retriever=retriever,
            chunk_folder=chunk_folder,
        )
        rows.append(
            _build_result_row(
                dataset_label="QASPER",
                dataset_key=QASPER_DATASET,
                retriever=retriever,
                chunk_folder=chunk_folder,
                sources=(qasper_source,),
                input_root=input_root,
                aggregation="per-query mean",
            )
        )

    for retriever in retrievers:
        musique_sources = tuple(
            _load_evaluation_source(
                input_root,
                dataset=dataset,
                retriever=retriever,
                chunk_folder=chunk_folder,
            )
            for dataset in MUSIQUE_DATASETS
        )
        rows.append(
            _build_result_row(
                dataset_label="MuSiQue (2–4 Hop Aggregate)",
                dataset_key="musique",
                retriever=retriever,
                chunk_folder=chunk_folder,
                sources=musique_sources,
                input_root=input_root,
                aggregation=(
                    "micro-average over available per-query metric values "
                    "from musique_2hop, musique_3hop, and musique_4hop"
                ),
            )
        )

    return {
        "schema_version": 1,
        "report": "QASPER and aggregated MuSiQue late-chunking retrieval",
        "chunk_folder": chunk_folder,
        "retrievers": list(retrievers),
        "aggregation_notes": [
            (
                "QASPER metrics are means over the available per-query values "
                "for that retriever."
            ),
            (
                "Each MuSiQue row concatenates the 2-hop, 3-hop, and 4-hop "
                "per-query evaluation rows, then computes one query-weighted "
                "mean per metric."
            ),
            (
                "Null metric values are excluded independently for each metric; "
                "metric_counts records every displayed metric's denominator."
            ),
        ],
        "rows": rows,
    }


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "–": "--",
    }
    return "".join(replacements.get(char, char) for char in text)


def _format_metric(value: object) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):.1f}"


def build_latex_table(report: Dict[str, object]) -> str:
    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Report has no result rows")

    lines = [
        "% Auto-generated by tables/generate_qasper_musique_summary.py",
        (
            "% MuSiQue is a query-weighted micro-average over the 2-hop, "
            "3-hop, and 4-hop per-query metrics."
        ),
        (
            "% The N column is the total number of evaluated query rows. "
            "Metric-specific denominators are recorded in the JSON sidecar."
        ),
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{QASPER and MuSiQue late-chunking retrieval results. "
            r"MuSiQue combines 2-hop, 3-hop, and 4-hop queries into one "
            r"query-weighted row per retriever. Values are percentages.}"
        ),
        r"\label{tab:qasper-musique-c250-retrieval}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrlrrrrrrrrrrrr}",
        r"\toprule",
        (
            r"Dataset & Retriever & N & Method & "
            r"\multicolumn{6}{c}{Ranking Metrics} & "
            r"\multicolumn{6}{c}{Binary Metrics} \\"
        ),
        r"\cmidrule(lr){5-10}\cmidrule(lr){11-16}",
        (
            r"& & & & \multicolumn{2}{c}{Gold} & "
            r"\multicolumn{2}{c}{Silver-L} & "
            r"\multicolumn{2}{c}{Union-L} & "
            r"\multicolumn{2}{c}{Gold} & "
            r"\multicolumn{2}{c}{Silver-S} & "
            r"\multicolumn{2}{c}{Union-S} \\"
        ),
        (
            r"& & & & NDCG@10 & Recall@10 & NDCG@10 & Recall@10 & "
            r"NDCG@10 & Recall@10 & HR@5 & HR@10 & HR@5 & HR@10 & "
            r"HR@5 & HR@10 \\"
        ),
        r"\midrule",
    ]

    previous_dataset: Optional[str] = None
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("Report row must be a JSON object")
        dataset_label = str(raw_row.get("dataset_label") or "")
        display_dataset = (
            dataset_label if dataset_label != previous_dataset else ""
        )
        metrics = raw_row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Report row is missing its metrics object")
        cells = [
            _latex_escape(display_dataset),
            _latex_escape(str(raw_row.get("retriever_label") or "")),
            str(int(raw_row.get("n_queries") or 0)),
            _latex_escape(str(raw_row.get("method") or "")),
        ]
        cells.extend(_format_metric(metrics.get(metric.key)) for metric in ALL_METRICS)
        lines.append(" & ".join(cells) + r" \\")
        previous_dataset = dataset_label

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _default_input_root(script_path: Path) -> Path:
    return script_path.parent.parent / "late_chunk_evaluations"


def _default_output_tex(script_path: Path) -> Path:
    return (
        script_path.parent.parent
        / "docs"
        / "qasper_musique_c250_retrieval_table.tex"
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a commit-ready QASPER and aggregated-MuSiQue LaTeX "
            "retrieval table plus a JSON audit file."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_default_input_root(script_path),
        help="Root containing late_chunk_evaluations.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=_default_output_tex(script_path),
        help="Commit-ready LaTeX output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="JSON audit output path (default: output-tex with .json suffix).",
    )
    parser.add_argument(
        "--chunk-folder",
        default="c250_o0",
        help="Run folder to summarize, such as c250_o0.",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=list(DEFAULT_RETRIEVERS),
        help="Retriever directory names in desired table order.",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print the generated LaTeX table to standard output.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.resolve()
    output_tex = args.output_tex.resolve()
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else output_tex.with_suffix(".json")
    )

    report = build_report(
        input_root,
        chunk_folder=args.chunk_folder,
        retrievers=args.retrievers,
    )
    latex = build_latex_table(report)

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(latex, encoding="utf-8")
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote LaTeX table: {output_tex}")
    print(f"Wrote JSON audit: {output_json}")
    print(f"Loaded {len(report['rows'])} result rows from {input_root}")
    if args.print_table:
        print()
        print(latex, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
