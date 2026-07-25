from __future__ import annotations

import json
from pathlib import Path

from chunked_pooling.experiment_datasets import load_dataset_bundle
from chunked_pooling.retrieval_labeling import generate_label_rows_for_run


def _bundle(hop: int, qa_n: object = "all"):
    return load_dataset_bundle(
        {
            "type": "task_registry",
            "dataset_name": f"musique_{hop}hop",
            "split": "validation",
            "config_name": f"{hop}hop",
            "qa_n": qa_n,
            "qa_selection_method": "first",
            "prepend_title": False,
        }
    )


def test_packaged_musique_counts_and_integrity() -> None:
    expected_groups = {2: 65, 3: 64, 4: 66}
    for hop in (2, 3, 4):
        bundle = _bundle(hop)
        assert len(bundle.qa_entries) == 300
        assert len(bundle.documents) == expected_groups[hop]
        for entry in bundle.qa_entries:
            document = str(bundle.documents[entry["doc_id"]]["text"])
            assert entry["retrieval_spans"]
            assert all(span in document for span in entry["retrieval_spans"])


def test_musique_subsampling_keeps_referenced_documents() -> None:
    bundle = _bundle(2, qa_n=3)
    assert len(bundle.qa_entries) == 3
    assert {entry["doc_id"] for entry in bundle.qa_entries} == set(bundle.documents)


def test_automatic_labels_keep_distant_spans_independent(tmp_path: Path) -> None:
    run_dir = tmp_path / "musique_2hop" / "jina" / "c250_o0"
    selection = run_dir / "selection"
    chunks_dir = run_dir / "chunking" / "group"
    selection.mkdir(parents=True)
    chunks_dir.mkdir(parents=True)
    (selection / "qa_entries.json").write_text(
        json.dumps(
            [
                {
                    "query_id": "q",
                    "doc_id": "group",
                    "document_id": "group",
                    "question": "question?",
                    "retrieval_spans": [
                        "first supporting paragraph",
                        "second supporting paragraph",
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    chunks = [
        {"chunk_id": "group:0", "raw_text": "first supporting paragraph"},
        {"chunk_id": "group:1", "raw_text": "irrelevant middle material"},
        {"chunk_id": "group:2", "raw_text": "second supporting paragraph"},
    ]
    (chunks_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks),
        encoding="utf-8",
    )

    labels = generate_label_rows_for_run(run_dir, dataset_name="musique_2hop")
    assert labels["q"].gold_chunk_ids == ["group:0", "group:2"]
    assert "group:1" not in labels["q"].gold_chunk_ids
    assert "group:1" not in labels["q"].silver_chunk_ids
