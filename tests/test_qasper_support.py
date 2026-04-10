import pytest

from chunked_pooling.experiment_config import parse_retriever_specs, resolve_run_config
from chunked_pooling.experiment_datasets import load_dataset_bundle, select_dataset_subset
from chunked_pooling.experiment_retrievers import (
    _build_model_load_kwargs,
    _validate_runtime_requirements,
)
from chunked_pooling.wrappers import load_tokenizer
from chunked_pooling.wrappers import _ensure_remote_code_compat


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


def test_loogle_yaml_mapping_matches_reference_defaults():
    default_experiment = {
        "output_dir": "outputs/loogle_ablation",
        "dataset": {
            "name": "loogle",
            "split": "test",
            "config_name": "shortdep_qa",
            "qa_n": "all",
            "qa_selection_method": "first",
        },
        "ingest": {
            "enabled": True,
            "strategy": "hierarchical",
            "chunk_size": 300,
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
    resolved_config, notes = resolve_run_config(
        dataset_name="loogle",
        default_experiment=default_experiment,
        retrievers=retrievers,
        resume=True,
    )

    assert resolved_config["dataset_loader"]["dataset_name"] == "loogle"
    assert resolved_config["dataset_loader"]["split"] == "test"
    assert resolved_config["dataset_loader"]["config_name"] == "shortdep_qa"
    assert resolved_config["dataset_loader"]["qa_n"] == "all"
    assert resolved_config["dataset_loader"]["qa_selection_method"] == "first"
    assert resolved_config["dataset_loader"]["max_docs"] == 25
    assert resolved_config["chunking"]["chunk_size"] == 300
    assert resolved_config["chunking"]["overlap"] == 0
    assert resolved_config["retrieval"]["scope"] == "per_document"
    assert resolved_config["retrieval"]["retrieve_k"] == 5
    assert any("ingest.strategy='hierarchical'" in note for note in notes)


def test_qwen_retriever_alias_is_available():
    retrievers = parse_retriever_specs(["qwen"], {})
    assert retrievers[0]["name"] == "qwen"
    assert retrievers[0]["model_name"] == "Qwen/Qwen3-Embedding-8B"
    assert retrievers[0]["pooling"] == "last_token"
    assert retrievers[0]["min_transformers_version"] == "4.51.0"
    assert retrievers[0]["shard_across_available_gpus"] is True


def test_jina_retriever_alias_rejects_transformers_5(monkeypatch):
    retrievers = parse_retriever_specs(["jina"], {})
    monkeypatch.setattr(
        "chunked_pooling.experiment_retrievers.transformers.__version__",
        "5.0.0",
    )

    with pytest.raises(RuntimeError, match="requires transformers<5.0.0"):
        _validate_runtime_requirements(retrievers[0])


def test_qwen_runtime_validation_reports_old_transformers(monkeypatch):
    retrievers = parse_retriever_specs(["qwen"], {})
    monkeypatch.setattr(
        "chunked_pooling.experiment_retrievers.transformers.__version__",
        "4.43.4",
    )

    with pytest.raises(RuntimeError, match="requires transformers>=4.51.0"):
        _validate_runtime_requirements(retrievers[0])


def test_qwen_builds_auto_device_map_when_multiple_gpus_are_available(monkeypatch):
    retrievers = parse_retriever_specs(["qwen"], {})
    monkeypatch.setattr("chunked_pooling.experiment_retrievers.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("chunked_pooling.experiment_retrievers.torch.cuda.device_count", lambda: 4)
    monkeypatch.setattr(
        "chunked_pooling.experiment_retrievers.importlib.util.find_spec",
        lambda name: object() if name == "accelerate" else None,
    )

    kwargs = _build_model_load_kwargs(retrievers[0])

    assert kwargs["device_map"] == "auto"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["torch_dtype"] == "auto"


def test_qwen_multi_gpu_requires_accelerate(monkeypatch):
    retrievers = parse_retriever_specs(["qwen"], {})
    monkeypatch.setattr("chunked_pooling.experiment_retrievers.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("chunked_pooling.experiment_retrievers.torch.cuda.device_count", lambda: 4)
    monkeypatch.setattr(
        "chunked_pooling.experiment_retrievers.importlib.util.find_spec",
        lambda name: None,
    )

    with pytest.raises(RuntimeError, match="requires the 'accelerate' package"):
        _build_model_load_kwargs(retrievers[0])


def test_load_tokenizer_prefers_standard_path_before_remote_code(monkeypatch):
    calls = []

    class DummyTokenizer:
        pass

    def fake_from_pretrained(name, trust_remote_code=False, **kwargs):
        calls.append((name, trust_remote_code))
        if trust_remote_code:
            raise AssertionError("remote-code fallback should not be used here")
        return DummyTokenizer()

    monkeypatch.setattr(
        "chunked_pooling.wrappers.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    tokenizer = load_tokenizer("jinaai/jina-embeddings-v2-small-en")

    assert isinstance(tokenizer, DummyTokenizer)
    assert calls == [("jinaai/jina-embeddings-v2-small-en", False)]


def test_jina_remote_code_installs_transformers_onnx_compat(monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "transformers.onnx", raising=False)

    _ensure_remote_code_compat("jinaai/jina-embeddings-v2-small-en")

    import transformers.onnx as transformers_onnx

    assert hasattr(transformers_onnx, "OnnxConfig")
    assert hasattr(transformers_onnx, "OnnxConfigWithPast")
    assert hasattr(transformers_onnx, "PatchingSpec")


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


def test_loogle_loader_and_sampling_follow_reference_order(monkeypatch):
    fake_dataset = {
        "test": [
            {
                "title": "Doc One",
                "context": "Document one context.",
                "qa_pairs": '[{"Q": "Question one?", "A": "Answer one", "S": "Evidence one"}]',
            },
            {
                "doc_id": "doc-2",
                "context": "Document two context.",
                "question": "Question two?",
                "answer": "Answer two",
                "evidence": "Evidence two",
            },
        ]
    }

    monkeypatch.setattr(
        "chunked_pooling.experiment_datasets._load_loogle_dataset",
        lambda cfg=None: fake_dataset,
    )

    bundle = load_dataset_bundle(
        {
            "type": "task_registry",
            "dataset_name": "loogle",
            "split": "test",
            "config_name": "shortdep_qa",
            "qa_n": "all",
            "qa_selection_method": "first",
        }
    )

    assert list(bundle.documents.keys()) == ["Doc One", "doc-2"]
    assert bundle.documents["Doc One"]["text"] == "Document one context."
    assert len(bundle.qa_entries) == 2
    assert bundle.qa_entries[0]["query_id"] == "loogle_0"
    assert bundle.qa_entries[0]["doc_id"] == "Doc One"
    assert bundle.qa_entries[0]["reference_answers"] == ["Answer one"]
    assert bundle.qa_entries[0]["retrieval_spans"] == ["Evidence one"]
    assert bundle.qa_entries[1]["doc_id"] == "doc-2"
    assert bundle.qa_entries[1]["reference_answers"] == ["Answer two"]

    filtered_subset = select_dataset_subset(
        bundle=bundle,
        max_docs=1,
        max_questions=None,
    )
    assert list(filtered_subset.documents.keys()) == ["Doc One"]
    assert [entry["doc_id"] for entry in filtered_subset.qa_entries] == ["Doc One"]


def test_resolve_run_config_applies_explicit_overrides():
    default_experiment = {
        "dataset": {
            "name": "qasper",
            "split": "test",
            "config_name": "default",
            "qa_n": "all",
            "qa_selection_method": "first",
        },
        "chunking": {
            "strategy": "fixed",
            "chunk_size": 200,
            "overlap": 0,
            "tokenizer_name": "jinaai/jina-embeddings-v2-small-en",
        },
        "retrieval": {
            "retriever": "jina",
            "retrieve_k": 5,
            "scope": "per_document",
        },
        "sample": {
            "max_documents": 25,
        },
        "late_chunking": {
            "max_tokens_per_forward": 8192,
            "window_overlap_tokens": 256,
        },
    }
    retrievers = parse_retriever_specs([], default_experiment)
    resolved_config, notes = resolve_run_config(
        dataset_name="qasper",
        default_experiment=default_experiment,
        retrievers=retrievers,
        resume=True,
        overrides={
            "chunk_size": 384,
            "chunk_overlap": 32,
            "retrieve_k": 7,
            "max_docs": 12,
            "late_max_tokens_per_forward": 4096,
        },
    )

    assert resolved_config["chunking"]["chunk_size"] == 384
    assert resolved_config["chunking"]["overlap"] == 32
    assert resolved_config["retrieval"]["retrieve_k"] == 7
    assert resolved_config["dataset_loader"]["max_docs"] == 12
    assert resolved_config["late_chunking"]["max_tokens_per_forward"] == 4096
    assert any("chunking.chunk_size overridden" in note for note in notes)
