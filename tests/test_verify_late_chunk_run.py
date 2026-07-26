from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from verify_late_chunk_run import verify_run


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _complete_run(tmp_path: Path, retriever_name: str = "qwen") -> Path:
    run_dir = tmp_path / "late_chunk_runs" / "musique_2hop" / retriever_name / "c250_o0"
    query_rows = [{"query_id": "q1"}]
    chunk_rows = [
        {"chunk_id": "doc-1__chunk_00000"},
        {"chunk_id": "doc-1__chunk_00001"},
    ]

    _write_json(
        run_dir / "run_manifest.json",
        {"retrievers_used": [retriever_name]},
    )
    config_path = run_dir / "config" / "late_chunking_run.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("resume: true\n", encoding="utf-8")
    _write_json(run_dir / "selection" / "selected_doc_ids.json", ["doc-1"])
    _write_json(run_dir / "selection" / "qa_entries.json", query_rows)
    _write_json(run_dir / "indexing" / "index_manifest.json", {})
    _write_json(
        run_dir / "indexing" / retriever_name / "index_stats.json",
        {},
    )
    _write_jsonl(run_dir / "chunking" / "doc-1" / "chunks.jsonl", chunk_rows)
    _write_jsonl(
        run_dir
        / "retrieval"
        / f"retrieval_payloads__{retriever_name}__late_chunking__per_document.jsonl",
        query_rows,
    )
    _write_json(
        run_dir
        / "retrieval"
        / f"retrieval_results_raw__{retriever_name}__late_chunking__per_document.json",
        query_rows,
    )

    if retriever_name != "bm25":
        matrix_path = (
            run_dir / "indexing" / retriever_name / "chunk_embeddings.npy"
        )
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(matrix_path, np.ones((2, 3), dtype=np.float32))
        _write_json(
            run_dir / "indexing" / retriever_name / "chunk_ids.json",
            [row["chunk_id"] for row in chunk_rows],
        )
    return run_dir


def test_verify_run_accepts_complete_dense_artifacts(tmp_path: Path):
    run_dir = _complete_run(tmp_path)
    assert verify_run(run_dir, "qwen") == []


def test_verify_run_rejects_dense_row_count_mismatch(tmp_path: Path):
    run_dir = _complete_run(tmp_path)
    np.save(
        run_dir / "indexing" / "qwen" / "chunk_embeddings.npy",
        np.ones((1, 3), dtype=np.float32),
    )
    errors = verify_run(run_dir, "qwen")
    assert any("row count" in error for error in errors)


def test_verify_run_accepts_complete_bm25_artifacts(tmp_path: Path):
    run_dir = _complete_run(tmp_path, retriever_name="bm25")
    assert verify_run(run_dir, "bm25") == []
