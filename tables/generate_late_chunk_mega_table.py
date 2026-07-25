#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RANKING_VIEW_ORDER = ("gold", "silver_loose", "union_loose")
RANKING_VIEW_LABELS = {
    "gold": "Gold",
    "silver_loose": "Silver-L",
    "union_loose": "Union-L",
}
BINARY_VIEW_ORDER = ("gold_hit", "silver_strict_hit", "strict_union_hit")
BINARY_VIEW_LABELS = {
    "gold_hit": "Gold",
    "silver_strict_hit": "Silver-S",
    "strict_union_hit": "Union-S",
}
RANKING_METRIC_ORDER = (
    ("mrr@10", "MRR@10"),
    ("ndcg@10", "NDCG@10"),
    ("recall@10", "R@10"),
)
BINARY_METRIC_ORDER = (
    ("hit_rate@5", "HR@5"),
    ("hit_rate@10", "HR@10"),
)
DATASET_ORDER = {
    "qasper": 0,
    "loogle": 1,
    "narrativeqa": 2,
    "quality": 3,
    "novelhopqa": 4,
    "novelqa": 4,
    "musique-2hop": 5,
    "musique-3hop": 6,
    "musique-4hop": 7,
}
DATASET_LABELS = {
    "musique_2hop": "MuSiQue-2Hop",
    "musique_3hop": "MuSiQue-3Hop",
    "musique_4hop": "MuSiQue-4Hop",
}
RETRIEVER_ORDER = {
    "jina": 0,
    "qwen": 1,
}
RUN_PATTERN = re.compile(r"^(?P<prefix>[a-z])(?P<size>\d+)_o(?P<overlap>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RunKey:
    dataset: str
    retriever: str
    chunk_size_label: str
    chunk_size_sort: Tuple[int, str]
    overlap_label: str
    overlap_sort: int


@dataclass
class RunCell:
    source_path: Path
    ranking_metrics_by_view: Dict[str, Dict[str, Optional[float]]]
    hit_rate_metrics_by_view: Dict[str, Dict[str, Optional[float]]]


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
    }
    return "".join(replacements.get(char, char) for char in text)


def _format_metric(value: Optional[float], missing_token: str) -> str:
    if value is None:
        return missing_token
    return f"{value * 100.0:.1f}"


def _parse_run_identity(summary_path: Path, payload: Dict[str, object]) -> RunKey:
    dataset = str(payload.get("dataset_name") or summary_path.parts[-4]).strip()
    dataset = DATASET_LABELS.get(dataset.lower(), dataset)
    run_name = str(payload.get("run_name") or "").strip()

    retriever = summary_path.parts[-3]
    run_leaf = summary_path.parts[-2]
    if run_name:
        parts = [part for part in run_name.split("/") if part]
        if len(parts) >= 2:
            retriever = parts[-2]
            run_leaf = parts[-1]
        elif len(parts) == 1:
            run_leaf = parts[0]

    match = RUN_PATTERN.match(run_leaf)
    if match:
        prefix = match.group("prefix").lower()
        size = match.group("size")
        overlap = match.group("overlap")
        if prefix == "c":
            chunk_size_label = size
            chunk_size_sort = (0, f"{int(size):08d}")
        else:
            chunk_size_label = f"{prefix}{size}"
            chunk_size_sort = (1, f"{prefix}{int(size):08d}")
        overlap_label = overlap
        overlap_sort = int(overlap)
    else:
        chunk_size_label = run_leaf
        chunk_size_sort = (9, run_leaf)
        overlap_label = "--"
        overlap_sort = 10**9

    return RunKey(
        dataset=dataset,
        retriever=retriever,
        chunk_size_label=chunk_size_label,
        chunk_size_sort=chunk_size_sort,
        overlap_label=overlap_label,
        overlap_sort=overlap_sort,
    )


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _coerce_metric_dict(raw_value: object, *, path: Path, key_name: str) -> Dict[str, Optional[float]]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError(f"Expected {key_name} to be a JSON object in {path}")
    metrics: Dict[str, Optional[float]] = {}
    for metric_name, metric_value in raw_value.items():
        if metric_value is None:
            metrics[str(metric_name)] = None
        else:
            metrics[str(metric_name)] = float(metric_value)
    return metrics


def _extract_ranking_metrics(payload: Dict[str, object], summary_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    raw_block = payload.get("ranking_metrics_by_view")
    if raw_block is None:
        raw_block = payload.get("retrieval_metrics_by_view")
    if raw_block is None:
        raise ValueError(f"Missing ranking metrics block in {summary_path}")
    if not isinstance(raw_block, dict):
        raise ValueError(f"Expected ranking metrics block to be a JSON object in {summary_path}")

    ranking_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for view_name in RANKING_VIEW_ORDER:
        source_view_name = view_name
        if source_view_name not in raw_block and view_name == "union_loose" and "loose_union" in raw_block:
            source_view_name = "loose_union"
        ranking_metrics[view_name] = _coerce_metric_dict(
            raw_block.get(source_view_name),
            path=summary_path,
            key_name=f"ranking_metrics_by_view[{source_view_name!r}]",
        )
    return ranking_metrics


def _extract_hit_metrics(payload: Dict[str, object], summary_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    raw_block = payload.get("hit_rate_metrics_by_view")
    if raw_block is None:
        raw_block = payload.get("retrieval_metrics_by_view")
    if raw_block is None:
        raise ValueError(f"Missing hit-rate metrics block in {summary_path}")
    if not isinstance(raw_block, dict):
        raise ValueError(f"Expected hit-rate metrics block to be a JSON object in {summary_path}")

    hit_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for view_name in BINARY_VIEW_ORDER:
        hit_metrics[view_name] = _coerce_metric_dict(
            raw_block.get(view_name),
            path=summary_path,
            key_name=f"hit_rate_metrics_by_view[{view_name!r}]",
        )
    return hit_metrics


def load_table_records(input_root: Path) -> Dict[RunKey, RunCell]:
    rows: Dict[RunKey, RunCell] = {}
    summary_paths = sorted(input_root.rglob("metrics_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No metrics_summary.json files were found under {input_root}")

    for summary_path in summary_paths:
        payload = _load_json(summary_path)
        run_key = _parse_run_identity(summary_path, payload)
        if run_key in rows:
            raise ValueError(
                "Duplicate metrics for the same run:\n"
                f"  {rows[run_key].source_path}\n"
                f"  {summary_path}\n"
                f"Resolved key: dataset={run_key.dataset}, retriever={run_key.retriever}, "
                f"chunk_size={run_key.chunk_size_label}, overlap={run_key.overlap_label}"
            )

        rows[run_key] = RunCell(
            source_path=summary_path,
            ranking_metrics_by_view=_extract_ranking_metrics(payload, summary_path),
            hit_rate_metrics_by_view=_extract_hit_metrics(payload, summary_path),
        )

    return rows


def _dataset_sort_key(dataset_name: str) -> Tuple[int, str]:
    lowered = dataset_name.strip().lower()
    return (DATASET_ORDER.get(lowered, 99), lowered)


def _retriever_sort_key(retriever_name: str) -> Tuple[int, str]:
    lowered = retriever_name.strip().lower()
    return (RETRIEVER_ORDER.get(lowered, 99), lowered)


def sorted_run_keys(rows: Dict[RunKey, RunCell]) -> List[RunKey]:
    return sorted(
        rows,
        key=lambda key: (
            _dataset_sort_key(key.dataset),
            _retriever_sort_key(key.retriever),
            key.chunk_size_sort,
            key.overlap_sort,
        ),
    )


def _column_spec() -> str:
    metric_columns = "".join(
        "r" for _ in range(len(RANKING_VIEW_ORDER) * len(RANKING_METRIC_ORDER) + len(BINARY_VIEW_ORDER) * len(BINARY_METRIC_ORDER))
    )
    return "llcc" + metric_columns


def _build_header_lines() -> List[str]:
    top_header = [
        "Dataset",
        "Retriever",
        "Chunk Size",
        "Overlap",
        r"\multicolumn{9}{c}{Ranking Metrics}",
        r"\multicolumn{6}{c}{Binary Metrics}",
    ]

    second_header = ["", "", "", ""]
    for view_name in RANKING_VIEW_ORDER:
        second_header.append(rf"\multicolumn{{3}}{{c}}{{{RANKING_VIEW_LABELS[view_name]}}}")
    for view_name in BINARY_VIEW_ORDER:
        second_header.append(rf"\multicolumn{{2}}{{c}}{{{BINARY_VIEW_LABELS[view_name]}}}")

    third_header = ["", "", "", ""]
    for _view_name in RANKING_VIEW_ORDER:
        for _metric_name, metric_label in RANKING_METRIC_ORDER:
            third_header.append(metric_label)
    for _view_name in BINARY_VIEW_ORDER:
        for _metric_name, metric_label in BINARY_METRIC_ORDER:
            third_header.append(metric_label)

    cmidrules_top = [
        r"\cmidrule(lr){5-13}",
        r"\cmidrule(lr){14-19}",
    ]
    cmidrules_second = []
    column_start = 5
    for _view_name in RANKING_VIEW_ORDER:
        cmidrules_second.append(rf"\cmidrule(lr){{{column_start}-{column_start + 2}}}")
        column_start += 3
    for _view_name in BINARY_VIEW_ORDER:
        cmidrules_second.append(rf"\cmidrule(lr){{{column_start}-{column_start + 1}}}")
        column_start += 2

    return [
        " & ".join(top_header) + r" \\",
        "".join(cmidrules_top),
        " & ".join(second_header) + r" \\",
        "".join(cmidrules_second),
        " & ".join(third_header) + r" \\",
        r"\midrule",
    ]


def build_latex_table(
    rows: Dict[RunKey, RunCell],
    *,
    caption: str,
    label: str,
    missing_token: str,
) -> str:
    sorted_keys = sorted_run_keys(rows)
    total_columns = 19
    header_lines = _build_header_lines()

    lines: List[str] = [
        "% Auto-generated by generate_late_chunk_mega_table.py",
        "% Requires: \\usepackage{booktabs,longtable,pdflscape}",
        "% Ranking metrics use Gold / Silver-L / Union-L with MRR@10, NDCG@10, and Recall@10.",
        "% Binary metrics use Gold / Silver-S / Union-S with HR@5 and HR@10.",
        "",
        "\\begin{landscape}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\scriptsize",
        rf"\begin{{longtable}}{{{_column_spec()}}}",
        rf"\caption{{{_latex_escape(caption)}}}\label{{{_latex_escape(label)}}} \\",
        r"\toprule",
    ]
    lines.extend(header_lines)
    lines.extend(
        [
            r"\endfirsthead",
            rf"\multicolumn{{{total_columns}}}{{l}}{{\tablename\ \thetable\ -- continued from previous page}} \\",
            r"\toprule",
        ]
    )
    lines.extend(header_lines)
    lines.extend(
        [
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{total_columns}}}{{r}}{{\emph{{Continued on next page}}}} \\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )

    previous_dataset: Optional[str] = None
    previous_retriever: Optional[str] = None
    for run_key in sorted_keys:
        run_cell = rows[run_key]
        dataset_cell = run_key.dataset if run_key.dataset != previous_dataset else ""
        retriever_cell = (
            run_key.retriever
            if run_key.dataset != previous_dataset or run_key.retriever != previous_retriever
            else ""
        )

        row_cells = [
            _latex_escape(dataset_cell),
            _latex_escape(retriever_cell),
            _latex_escape(run_key.chunk_size_label),
            _latex_escape(run_key.overlap_label),
        ]

        for view_name in RANKING_VIEW_ORDER:
            metrics = run_cell.ranking_metrics_by_view.get(view_name, {})
            for metric_name, _metric_label in RANKING_METRIC_ORDER:
                row_cells.append(_format_metric(metrics.get(metric_name), missing_token))

        for view_name in BINARY_VIEW_ORDER:
            metrics = run_cell.hit_rate_metrics_by_view.get(view_name, {})
            for metric_name, _metric_label in BINARY_METRIC_ORDER:
                row_cells.append(_format_metric(metrics.get(metric_name), missing_token))

        lines.append(" & ".join(row_cells) + r" \\")
        previous_dataset = run_key.dataset
        previous_retriever = run_key.retriever

    lines.extend(
        [
            r"\end{longtable}",
            r"\end{landscape}",
            "",
        ]
    )
    return "\n".join(lines)


def _default_input_root(script_path: Path) -> Path:
    return script_path.parent.parent / "late_chunk_evaluations"


def _default_output_path(script_path: Path) -> Path:
    return script_path.parent / "late_chunking_mega_table.txt"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX mega-table for late chunking ablations from compact "
            "evaluation outputs."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_default_input_root(script_path),
        help="Directory containing copied late_chunk_evaluations results.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=_default_output_path(script_path),
        help="Where to write the LaTeX table as a .txt file.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Late chunking retrieval ablation results across datasets, retrievers, "
            "chunk sizes, and overlaps."
        ),
        help="LaTeX caption for the generated table.",
    )
    parser.add_argument(
        "--label",
        default="tab:late-chunking-mega-results",
        help="LaTeX label for the generated table.",
    )
    parser.add_argument(
        "--missing-token",
        default="--",
        help="Token used when a metric is not available.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.resolve()
    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = load_table_records(input_root)
    latex = build_latex_table(
        rows,
        caption=args.caption,
        label=args.label,
        missing_token=args.missing_token,
    )
    output_file.write_text(latex, encoding="utf-8")

    print(f"Wrote LaTeX table to {output_file}")
    print(f"Loaded {len(rows)} unique run rows from {input_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
