from chunked_pooling.experiment_config import parse_retriever_specs, resolve_run_config
from chunked_pooling.experiment_datasets import load_dataset_bundle, select_dataset_subset


def test_qasper_yaml_mapping_matches_reference_defaults():
    default_experiment = {
        "output_dir": "outputs/qasper_ablation",
        "dataset": {
            "name": "qasper",
            "split": "test",
            "config_name": "default",
            "qa_n": "all",
            "qa_selection_method": "first",
        },
        "ingest": {
            "enabled": True,
            "strategy": "hierarchical",
            "chunk_size": 200,
            "chunk_overlap": 0,
        },
        "retrieval": {
            "retriever": "bm25",
            "retrieve_k": 5,
            "scope": "per_document",
        },
        "sample": {
            "max_documents": 25,
        },
    }
    retrievers = parse_retriever_specs([], default_experiment)
    assert retrievers[0]["name"] == "bm25"
    resolved_config, notes = resolve_run_config(
        dataset_name="qasper",
        default_experiment=default_experiment,
        retrievers=retrievers,
        resume=True,
    )

    assert resolved_config["dataset_loader"]["dataset_name"] == "qasper"
    assert resolved_config["dataset_loader"]["split"] == "test"
    assert resolved_config["dataset_loader"]["config_name"] == "default"
    assert resolved_config["dataset_loader"]["qa_n"] == "all"
    assert resolved_config["dataset_loader"]["qa_selection_method"] == "first"
    assert resolved_config["dataset_loader"]["max_docs"] == 25
    assert resolved_config["chunking"]["strategy"] == "fixed"
    assert resolved_config["chunking"]["chunk_size"] == 200
    assert resolved_config["chunking"]["overlap"] == 0
    assert resolved_config["retrieval"]["scope"] == "per_document"
    assert resolved_config["retrieval"]["retrieve_k"] == 5
    assert any("ingest.strategy='hierarchical'" in note for note in notes)


def test_qasper_loader_and_sampling_follow_reference_order(monkeypatch):
    fake_dataset = {
        "test": [
            {
                "id": "doc-1",
                "full_text": {
                    "paragraphs": [
                        ["Section one paragraph A.", "Section one paragraph B."],
                    ]
                },
                "qas": {
                    "question": ["What does section one describe?"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "extractive_spans": ["section one"],
                                    "evidence": ["Section one paragraph A."],
                                    "unanswerable": False,
                                }
                            ]
                        }
                    ],
                },
            },
            {
                "id": "doc-2",
                "full_text": {
                    "paragraphs": ["Second document paragraph."]
                },
                "qas": {
                    "question": ["What is in the second document?"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "free_form_answer": "A paragraph.",
                                    "highlighted_evidence": ["Second document paragraph."],
                                    "unanswerable": False,
                                }
                            ]
                        }
                    ],
                },
            },
            {
                "id": "doc-3",
                "full_text": {
                    "paragraphs": ["Third document only."]
                },
                "qas": {
                    "question": [],
                    "answers": [],
                },
            },
        ]
    }

    monkeypatch.setattr(
        "chunked_pooling.experiment_datasets._resolve_qasper_config",
        lambda requested, dataset_name="allenai/qasper": requested or "default",
    )
    monkeypatch.setattr(
        "chunked_pooling.experiment_datasets._load_qasper_dataset",
        lambda cfg=None: fake_dataset,
    )

    bundle = load_dataset_bundle(
        {
            "type": "task_registry",
            "dataset_name": "qasper",
            "split": "test",
            "config_name": "default",
            "docs_config_name": "default",
            "qa_config_name": "default",
            "qa_n": "all",
            "qa_selection_method": "first",
        }
    )

    assert list(bundle.documents.keys()) == ["doc-1", "doc-2", "doc-3"]
    assert bundle.documents["doc-1"]["text"] == "Section one paragraph A.\nSection one paragraph B."
    assert len(bundle.qa_entries) == 2
    assert bundle.qa_entries[0]["query_id"] == "qasper_0"
    assert bundle.qa_entries[0]["doc_id"] == "doc-1"
    assert bundle.qa_entries[0]["reference_answers"] == ["section one"]
    assert bundle.qa_entries[0]["retrieval_spans"] == ["Section one paragraph A."]

    subset = select_dataset_subset(
        bundle=bundle,
        max_docs=3,
        max_questions=None,
    )
    assert list(subset.documents.keys()) == ["doc-1", "doc-2", "doc-3"]
    assert [entry["doc_id"] for entry in subset.qa_entries] == ["doc-1", "doc-2"]

    filtered_subset = select_dataset_subset(
        bundle=bundle,
        max_docs=1,
        max_questions=None,
    )
    assert list(filtered_subset.documents.keys()) == ["doc-1"]
    assert [entry["doc_id"] for entry in filtered_subset.qa_entries] == ["doc-1"]
