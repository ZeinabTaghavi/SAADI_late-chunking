import json
from pathlib import Path

import pytest

from chunked_pooling.retrieval_evaluation import evaluate_run


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def test_evaluate_run_writes_compact_artifacts(tmp_path):
    run_dir = tmp_path / "late_chunk_runs" / "qasper" / "jina" / "c300_o0"
    retrieval_dir = run_dir / "retrieval"
    _write_jsonl(
        retrieval_dir / "retrieval_payloads__jina__late_chunking__per_document.jsonl",
        [
            {
                "query_id": "q1",
                "doc_id": "doc-1",
                "question": "Where is the answer?",
                "retrieved_chunk_ids": ["c2", "c1", "c1", "c3"],
                "scores": [0.9, 0.8, 0.7, 0.1],
            },
            {
                "query_id": "q2",
                "doc_id": "doc-2",
                "question": "Who is relevant?",
                "retrieved_chunk_ids": ["c9", "c8", "c7"],
                "scores": [0.6, 0.4, 0.2],
            },
        ],
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "dataset_name": "qasper",
            "run_name": "jina/c300_o0",
            "artifact_paths": {
                "retrieval_payloads_jina": "retrieval/retrieval_payloads__jina__late_chunking__per_document.jsonl",
            },
        },
    )

    labels_path = tmp_path / "labels.json"
    _write_json(
        labels_path,
        [
            {
                "query_id": "q1",
                "doc_id": "doc-1",
                "question": "Where is the answer?",
                "gold_chunk_ids": ["c1", "c4"],
            },
            {
                "query_id": "q2",
                "doc_id": "doc-2",
                "question": "Who is relevant?",
                "gold_chunk_ids": ["c8"],
            },
        ],
    )

    output_dir = run_dir / "evaluation"
    result = evaluate_run(
        run_dir=run_dir,
        labels_file=labels_path,
        output_dir=output_dir,
        method_name="late_chunking",
        dataset_name="qasper",
        split="test",
        ks=[5, 10],
        command="python3 evaluate_retrieval_run.py ...",
    )

    metrics_summary = json.loads((output_dir / "metrics_summary.json").read_text())
    leaderboard_row = json.loads((output_dir / "leaderboard_row.json").read_text())
    manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
    per_query_rows = [
        json.loads(line)
        for line in (output_dir / "metrics_per_query.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert result["metrics_summary"]["primary_relevance"] == "gold_chunk_ids"
    assert metrics_summary["method_name"] == "late_chunking"
    assert metrics_summary["retrieval_metrics"]["recall@5"] == pytest.approx(0.75)
    assert metrics_summary["retrieval_metrics"]["mrr@5"] == pytest.approx(0.5)
    assert metrics_summary["retrieval_metrics"]["hit_rate@10"] == 1.0
    assert leaderboard_row["ndcg@10"] == metrics_summary["retrieval_metrics"]["ndcg@10"]
    assert per_query_rows[0]["retrieved_ids_top10"] == ["c2", "c1", "c3"]
    assert per_query_rows[0]["relevant_ids"] == ["c1", "c4"]
    assert per_query_rows[0]["recall@5"] == pytest.approx(0.5)
    assert manifest["input_files_used"]["labels_file"].endswith("labels.json")
    assert manifest["join_summary"]["n_raw_queries"] == 2


def test_evaluate_run_reports_null_metrics_when_only_grouped_silver_exists(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "retrieval" / "retrieval_payloads__bm25__late_chunking__per_document.jsonl",
        [
            {
                "query_id": "q1",
                "retrieved_chunk_ids": ["c1", "c2"],
                "scores": [1.0, 0.5],
            }
        ],
    )
    labels_path = tmp_path / "labels.json"
    _write_json(
        labels_path,
        [
            {
                "query_id": "q1",
                "silver_chunk_groups": [["c1", "c2"]],
            }
        ],
    )

    output_dir = tmp_path / "evaluation"
    evaluate_run(
        run_dir=run_dir,
        labels_file=labels_path,
        output_dir=output_dir,
        method_name="late_chunking",
        dataset_name="toy",
        split="test",
    )

    metrics_summary = json.loads((output_dir / "metrics_summary.json").read_text())
    manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
    per_query_rows = [
        json.loads(line)
        for line in (output_dir / "metrics_per_query.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert metrics_summary["primary_relevance"] is None
    assert metrics_summary["retrieval_metrics"]["recall@5"] is None
    assert per_query_rows[0]["relevant_ids"] == []
    assert "silver_chunk_groups" in " ".join(manifest["assumptions"])
