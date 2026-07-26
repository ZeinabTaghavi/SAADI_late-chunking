from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tables"
    / "generate_qasper_musique_summary.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_qasper_musique_summary",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
summary_generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_generator
SPEC.loader.exec_module(summary_generator)


def _write_evaluation(
    root: Path,
    *,
    dataset: str,
    values: list[float],
    retriever: str = "qwen",
    chunk_folder: str = "c250_o0",
    null_first_recall: bool = False,
) -> None:
    output_dir = root / dataset / retriever / chunk_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, value in enumerate(values):
        row = {"query_id": f"{dataset}-q{index}"}
        row.update(
            {
                metric.key: value
                for metric in summary_generator.ALL_METRICS
            }
        )
        if null_first_recall and index == 0:
            row["gold_recall@10"] = None
        rows.append(row)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset,
                "run_name": f"{retriever}/{chunk_folder}",
                "n_queries": len(rows),
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics_per_query.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _complete_inputs(
    root: Path,
    *,
    chunk_folder: str = "c250_o0",
    value_offset: float = 0.0,
) -> None:
    _write_evaluation(
        root,
        dataset="qasper",
        values=[0.2 + value_offset, 0.4 + value_offset],
        chunk_folder=chunk_folder,
    )
    _write_evaluation(
        root,
        dataset="musique_2hop",
        values=[0.1 + value_offset],
        chunk_folder=chunk_folder,
    )
    _write_evaluation(
        root,
        dataset="musique_3hop",
        values=[0.2 + value_offset, 0.4 + value_offset],
        chunk_folder=chunk_folder,
    )
    _write_evaluation(
        root,
        dataset="musique_4hop",
        values=[0.9 + value_offset],
        chunk_folder=chunk_folder,
        null_first_recall=True,
    )


def test_musique_is_one_query_weighted_row_per_retriever(tmp_path: Path):
    evaluation_root = tmp_path / "late_chunk_evaluations"
    _complete_inputs(evaluation_root)

    report = summary_generator.build_report(
        evaluation_root,
        chunk_folder="c250_o0",
        retrievers=["qwen"],
    )

    assert len(report["rows"]) == 2
    qasper_row, musique_row = report["rows"]
    assert qasper_row["dataset"] == "qasper"
    assert qasper_row["n_queries"] == 2
    assert qasper_row["metrics"]["gold_ndcg@10"] == pytest.approx(0.3)

    assert musique_row["dataset"] == "musique"
    assert musique_row["n_queries"] == 4
    assert musique_row["source_query_counts"] == {
        "musique_2hop": 1,
        "musique_3hop": 2,
        "musique_4hop": 1,
    }
    # Micro-average: (0.1 + 0.2 + 0.4 + 0.9) / 4, rather than
    # averaging the three hop-level means.
    assert musique_row["metrics"]["gold_ndcg@10"] == pytest.approx(0.4)
    assert musique_row["metric_counts"]["gold_ndcg@10"] == 4
    assert musique_row["metric_counts"]["gold_recall@10"] == 3

    latex = summary_generator.build_latex_table(report)
    assert latex.count("MuSiQue (2--4 Hop Aggregate) &") == 1
    assert "MuSiQue-2hop" not in latex
    assert "MuSiQue-3hop" not in latex
    assert "MuSiQue-4hop" not in latex
    assert " & 250 & 0 & 4 & 40.0" in latex


def test_multiple_overlaps_are_separate_rows_but_hops_stay_aggregated(
    tmp_path: Path,
):
    evaluation_root = tmp_path / "late_chunk_evaluations"
    _complete_inputs(evaluation_root, chunk_folder="c250_o25")
    _complete_inputs(
        evaluation_root,
        chunk_folder="c250_o50",
        value_offset=0.05,
    )

    report = summary_generator.build_report(
        evaluation_root,
        chunk_folders=["c250_o25", "c250_o50"],
        retrievers=["qwen"],
    )

    assert [
        (row["dataset"], row["overlap"])
        for row in report["rows"]
    ] == [
        ("qasper", 25),
        ("qasper", 50),
        ("musique", 25),
        ("musique", 50),
    ]
    assert report["rows"][2]["n_queries"] == 4
    assert report["rows"][3]["n_queries"] == 4
    latex = summary_generator.build_latex_table(report)
    assert latex.count("MuSiQue (2--4 Hop Aggregate) &") == 1
    assert " & Qwen & 250 & 25 & 4 & 40.0" in latex
    assert " & Qwen & 250 & 50 & 4 & 45.0" in latex

    output_tex = tmp_path / "docs" / "multi_overlap.tex"
    output_json = tmp_path / "docs" / "multi_overlap.json"
    result = summary_generator.main(
        [
            "--input-root",
            str(evaluation_root),
            "--output-tex",
            str(output_tex),
            "--output-json",
            str(output_json),
            "--chunk-folders",
            "c250_o25",
            "c250_o50",
            "--retrievers",
            "qwen",
        ]
    )
    assert result == 0
    audit = json.loads(output_json.read_text(encoding="utf-8"))
    assert audit["chunk_folders"] == ["c250_o25", "c250_o50"]
    assert len(audit["rows"]) == 4


def test_main_writes_commit_ready_tex_and_json(tmp_path: Path):
    evaluation_root = tmp_path / "late_chunk_evaluations"
    _complete_inputs(evaluation_root)
    output_tex = tmp_path / "docs" / "summary.tex"
    output_json = tmp_path / "docs" / "summary.json"

    result = summary_generator.main(
        [
            "--input-root",
            str(evaluation_root),
            "--output-tex",
            str(output_tex),
            "--output-json",
            str(output_json),
            "--retrievers",
            "qwen",
        ]
    )

    assert result == 0
    assert output_tex.is_file()
    audit = json.loads(output_json.read_text(encoding="utf-8"))
    assert [row["dataset"] for row in audit["rows"]] == ["qasper", "musique"]
    assert audit["rows"][1]["metric_counts"]["gold_recall@10"] == 3


def test_query_count_mismatch_is_rejected(tmp_path: Path):
    evaluation_root = tmp_path / "late_chunk_evaluations"
    _write_evaluation(evaluation_root, dataset="qasper", values=[0.2])
    summary_path = (
        evaluation_root
        / "qasper"
        / "qwen"
        / "c250_o0"
        / "metrics_summary.json"
    )
    summary_path.write_text(
        json.dumps({"dataset_name": "qasper", "n_queries": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Query-count mismatch"):
        summary_generator.build_report(
            evaluation_root,
            chunk_folder="c250_o0",
            retrievers=["qwen"],
        )
