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

    result = evaluate_run(
        run_dir=run_dir,
        labels_file=labels_path,
        method_name="late_chunking",
        dataset_name="qasper",
        split="test",
        ks=[5, 10],
        command="python3 evaluate_retrieval_run.py ...",
    )

    output_dir = Path(result["output_dir"])
    metrics_summary = json.loads((output_dir / "metrics_summary.json").read_text())
    leaderboard_row = json.loads((output_dir / "leaderboard_row.json").read_text())
    manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
    per_query_rows = [
        json.loads(line)
        for line in (output_dir / "metrics_per_query.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert result["metrics_summary"]["primary_relevance"] == "gold_chunk_ids"
    assert output_dir == tmp_path / "late_chunk_evaluations" / "qasper" / "jina" / "c300_o0"
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

    result = evaluate_run(
        run_dir=run_dir,
        labels_file=labels_path,
        output_dir=tmp_path / "evaluation_override",
        method_name="late_chunking",
        dataset_name="toy",
        split="test",
    )

    output_dir = Path(result["output_dir"])
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


def test_evaluate_run_generates_qasper_labels_from_run_artifacts(tmp_path):
    run_dir = tmp_path / "late_chunk_runs" / "qasper" / "jina" / "c200_o0"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "dataset_name": "qasper",
            "run_name": "jina/c200_o0",
            "artifact_paths": {
                "retrieval_payloads_jina": "retrieval/retrieval_payloads__jina__late_chunking__per_document.jsonl",
            },
        },
    )
    _write_json(
        run_dir / "selection" / "qa_entries.json",
        [
            {
                "query_id": "qasper_0",
                "doc_id": "doc-1",
                "document_id": "doc-1",
                "question": "What supports the answer?",
                "answers": ["Answer text"],
                "retrieval_spans": ["Evidence span text"],
            }
        ],
    )
    _write_jsonl(
        run_dir / "chunking" / "doc-1" / "chunks.jsonl",
        [
            {
                "doc_id": "doc-1",
                "chunk_id": "c1",
                "chunk_index": 0,
                "raw_text": "Intro chunk.",
            },
            {
                "doc_id": "doc-1",
                "chunk_id": "c2",
                "chunk_index": 1,
                "raw_text": "Evidence span text appears here.",
            },
        ],
    )
    _write_jsonl(
        run_dir / "retrieval" / "retrieval_payloads__jina__late_chunking__per_document.jsonl",
        [
            {
                "query_id": "qasper_0",
                "doc_id": "doc-1",
                "question": "What supports the answer?",
                "retrieved_chunk_ids": ["c1", "c2"],
                "scores": [0.8, 0.7],
            }
        ],
    )

    result = evaluate_run(
        run_dir=run_dir,
        method_name="late_chunking",
        dataset_name="qasper",
        split="test",
        ks=[5, 10],
    )

    output_dir = Path(result["output_dir"])
    metrics_summary = json.loads((output_dir / "metrics_summary.json").read_text())
    manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
    per_query_rows = [
        json.loads(line)
        for line in (output_dir / "metrics_per_query.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert metrics_summary["primary_relevance"] == "gold_chunk_ids"
    assert metrics_summary["retrieval_metrics"]["mrr@5"] == pytest.approx(0.5)
    assert per_query_rows[0]["relevant_ids"] == ["c2"]
    assert manifest["relevance_source_used"]["labels_source"] == "generated_from_run"


def test_evaluate_run_generates_loogle_labels_from_run_artifacts(tmp_path):
    run_dir = tmp_path / "late_chunk_runs" / "loogle" / "qwen" / "c300_o0"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "dataset_name": "loogle",
            "run_name": "qwen/c300_o0",
            "artifact_paths": {
                "retrieval_payloads_qwen": "retrieval/retrieval_payloads__qwen__late_chunking__per_document.jsonl",
            },
        },
    )
    _write_json(
        run_dir / "selection" / "qa_entries.json",
        [
            {
                "query_id": "loogle_0",
                "doc_id": "doc-7",
                "document_id": "doc-7",
                "question": "Which passage supports the answer?",
                "answers": ["Final answer"],
                "retrieval_spans": ["LooGLE evidence span"],
            }
        ],
    )
    _write_jsonl(
        run_dir / "chunking" / "doc-7" / "chunks.jsonl",
        [
            {
                "doc_id": "doc-7",
                "chunk_id": "l1",
                "chunk_index": 0,
                "raw_text": "Opening context.",
            },
            {
                "doc_id": "doc-7",
                "chunk_id": "l2",
                "chunk_index": 1,
                "raw_text": "LooGLE evidence span is located in this chunk.",
            },
        ],
    )
    _write_jsonl(
        run_dir / "retrieval" / "retrieval_payloads__qwen__late_chunking__per_document.jsonl",
        [
            {
                "query_id": "loogle_0",
                "doc_id": "doc-7",
                "question": "Which passage supports the answer?",
                "retrieved_chunk_ids": ["l1", "l2"],
                "scores": [0.75, 0.7],
            }
        ],
    )

    result = evaluate_run(
        run_dir=run_dir,
        method_name="late_chunking",
        dataset_name="loogle",
        split="test",
        ks=[5, 10],
    )

    output_dir = Path(result["output_dir"])
    metrics_summary = json.loads((output_dir / "metrics_summary.json").read_text())
    manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
    per_query_rows = [
        json.loads(line)
        for line in (output_dir / "metrics_per_query.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert metrics_summary["primary_relevance"] == "gold_chunk_ids"
    assert metrics_summary["retrieval_metrics"]["mrr@5"] == pytest.approx(0.5)
    assert per_query_rows[0]["relevant_ids"] == ["l2"]
    assert manifest["relevance_source_used"]["labels_source"] == "generated_from_run"


def test_evaluate_run_generates_narrativeqa_labels_from_answer_text(tmp_path):
    run_dir = tmp_path / "late_chunk_runs" / "narrativeqa" / "jina" / "c300_o0"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "dataset_name": "narrativeqa",
            "run_name": "jina/c300_o0",
            "artifact_paths": {
                "retrieval_payloads_jina": "retrieval/retrieval_payloads__jina__late_chunking__per_document.jsonl",
            },
        },
    )
    _write_json(
        run_dir / "selection" / "qa_entries.json",
        [
            {
                "query_id": "narrativeqa_0",
                "doc_id": "story-1",
                "document_id": "story-1",
                "question": "Who was the captain?",
                "answers": ["Captain Aster"],
                "retrieval_spans": [],
            }
        ],
    )
    _write_jsonl(
        run_dir / "chunking" / "story-1" / "chunks.jsonl",
        [
            {
                "doc_id": "story-1",
                "chunk_id": "n1",
                "chunk_index": 0,
                "raw_text": "The voyage began in winter.",
            },
            {
                "doc_id": "story-1",
                "chunk_id": "n2",
                "chunk_index": 1,
                "raw_text": "Captain Aster led the crew through the storm.",
            },
        ],
    )
    _write_jsonl(
        run_dir / "retrieval" / "retrieval_payloads__jina__late_chunking__per_document.jsonl",
        [
            {
                "query_id": "narrativeqa_0",
                "doc_id": "story-1",
                "question": "Who was the captain?",
                "retrieved_chunk_ids": ["n1", "n2"],
                "scores": [0.82, 0.8],
            }
        ],
    )

    result = evaluate_run(
        run_dir=run_dir,
        method_name="late_chunking",
        dataset_name="narrativeqa",
        split="test",
        ks=[5, 10],
    )

    output_dir = Path(result["output_dir"])
    metrics_summary = json.loads((output_dir / "metrics_summary.json").read_text())
    manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
    per_query_rows = [
        json.loads(line)
        for line in (output_dir / "metrics_per_query.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert metrics_summary["primary_relevance"] == "gold_chunk_ids"
    assert metrics_summary["retrieval_metrics"]["mrr@5"] == pytest.approx(0.5)
    assert per_query_rows[0]["relevant_ids"] == ["n2"]
    assert "answer text" in " ".join(manifest["assumptions"]).lower()


def test_evaluate_run_generates_novelhopqa_window_labels_from_run_artifacts(tmp_path):
    run_dir = tmp_path / "late_chunk_runs" / "novelqa" / "qwen" / "c500_o0"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "dataset_name": "novelqa",
            "run_name": "qwen/c500_o0",
            "artifact_paths": {
                "retrieval_payloads_qwen": "retrieval/retrieval_payloads__qwen__late_chunking__per_document.jsonl",
            },
        },
    )
    _write_json(
        run_dir / "selection" / "qa_entries.json",
        [
            {
                "query_id": "hop_1:q0",
                "doc_id": "book:B30",
                "document_id": "book:B30",
                "question": "Who discovered the clue?",
                "answers": ["Alice"],
                "retrieval_spans": ["Context 30"],
                "retrieval_span_mode": "window",
            }
        ],
    )
    _write_jsonl(
        run_dir / "chunking" / "book:B30" / "chunks.jsonl",
        [
            {
                "doc_id": "book:B30",
                "chunk_id": "h1",
                "chunk_index": 0,
                "raw_text": "Context 30 begins here and continues.",
            },
            {
                "doc_id": "book:B30",
                "chunk_id": "h2",
                "chunk_index": 1,
                "raw_text": "More of Context 30 appears in this next chunk.",
            },
        ],
    )
    _write_jsonl(
        run_dir / "retrieval" / "retrieval_payloads__qwen__late_chunking__per_document.jsonl",
        [
            {
                "query_id": "hop_1:q0",
                "doc_id": "book:B30",
                "question": "Who discovered the clue?",
                "retrieved_chunk_ids": ["x0", "h1", "h2"],
                "scores": [0.9, 0.8, 0.7],
            }
        ],
    )

    result = evaluate_run(
        run_dir=run_dir,
        method_name="late_chunking",
        dataset_name="novelqa",
        split="test",
        ks=[5, 10],
    )

    output_dir = Path(result["output_dir"])
    metrics_summary = json.loads((output_dir / "metrics_summary.json").read_text())
    manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
    per_query_rows = [
        json.loads(line)
        for line in (output_dir / "metrics_per_query.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert metrics_summary["primary_relevance"] == "silver_chunk_ids"
    assert metrics_summary["retrieval_metrics"]["mrr@5"] == pytest.approx(0.5)
    assert per_query_rows[0]["relevant_ids"] == ["h1", "h2"]
    assert manifest["relevance_source_used"]["labels_source"] == "generated_from_run"
