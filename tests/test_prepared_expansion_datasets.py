from __future__ import annotations

from collections import Counter

import pytest

from chunked_pooling.experiment_datasets import load_task_registry_bundle


@pytest.mark.parametrize(
    ("dataset_name", "split", "expected_documents", "expected_queries", "first_doc_id"),
    (
        ("qasper_64k", "test", 23, 1372, "group_013"),
        ("musique_32k", "validation", 45, 900, "group_036"),
    ),
)
def test_prepared_expansions_match_main_saadi(
    dataset_name: str,
    split: str,
    expected_documents: int,
    expected_queries: int,
    first_doc_id: str,
) -> None:
    bundle = load_task_registry_bundle(
        {
            "dataset_name": dataset_name,
            "split": split,
            "qa_n": "all",
            "qa_selection_method": "first",
        }
    )

    assert len(bundle.documents) == expected_documents
    assert len(bundle.qa_entries) == expected_queries
    assert bundle.qa_entries[0]["query_id"] == "0"
    assert bundle.qa_entries[0]["doc_id"] == first_doc_id
    assert bundle.metadata["loader_type"] == "prepared_expansion"
    assert bundle.metadata["dataset_variant"] == dataset_name


def test_musique_32k_preserves_the_saadi_300_per_hop_fraction() -> None:
    bundle = load_task_registry_bundle(
        {
            "dataset_name": "musique_32k",
            "split": "validation",
            "qa_n": "all",
            "qa_selection_method": "first",
        }
    )

    assert Counter(row["hop"] for row in bundle.qa_entries) == {2: 300, 3: 300, 4: 300}
    assert all(row["retrieval_spans"] for row in bundle.qa_entries)


def test_prepared_expansion_selection_is_deterministic() -> None:
    config = {
        "dataset_name": "qasper_64k",
        "split": "test",
        "qa_n": 5,
        "qa_selection_method": "random",
    }
    first = load_task_registry_bundle(config)
    second = load_task_registry_bundle(config)

    assert [row["query_id"] for row in first.qa_entries] == [
        row["query_id"] for row in second.qa_entries
    ]
