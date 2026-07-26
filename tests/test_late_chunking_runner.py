import json
import re
from pathlib import Path

import numpy as np
import pytest

from chunked_pooling.experiment_chunking import build_chunk_records, make_chunking_signature
from chunked_pooling.experiment_config import (
    load_yaml_file,
    parse_retriever_specs,
    resolve_run_config,
)
from chunked_pooling.late_chunk_runner import run_late_chunking_experiment


class SimpleTokenizer:
    model_max_length = 512

    @staticmethod
    def _matches(text):
        return list(re.finditer(r"\w+|[^\w\s]", text))

    def __call__(
        self,
        text,
        return_offsets_mapping=False,
        add_special_tokens=False,
        **_,
    ):
        matches = self._matches(text)
        payload = {
            "input_ids": list(range(len(matches))),
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return payload

    def encode_plus(self, text, return_offsets_mapping=False, add_special_tokens=False, **kwargs):
        return self(
            text,
            return_offsets_mapping=return_offsets_mapping,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_build_chunk_records_is_deterministic():
    tokenizer = SimpleTokenizer()
    text = "Berlin is the capital of Germany."
    chunking_config = {
        "strategy": "fixed",
        "chunk_size": 3,
        "overlap": 1,
        "n_sentences": 1,
        "sentence_overlap": 0,
        "semantic_model_name": None,
        "tokenizer_name": "simple",
    }
    signature = make_chunking_signature(chunking_config)

    first, _ = build_chunk_records(
        doc_id="doc-1",
        text=text,
        tokenizer=tokenizer,
        chunking_config=chunking_config,
        chunking_signature=signature,
    )
    second, _ = build_chunk_records(
        doc_id="doc-1",
        text=text,
        tokenizer=tokenizer,
        chunking_config=chunking_config,
        chunking_signature=signature,
    )

    assert [row["chunk_id"] for row in first] == [row["chunk_id"] for row in second]
    assert [row["char_start"] for row in first] == [row["char_start"] for row in second]
    assert [row["char_end"] for row in first] == [row["char_end"] for row in second]


def test_local_json_bm25_run_writes_required_artifacts(tmp_path, monkeypatch):
    documents_path = tmp_path / "documents.jsonl"
    qa_path = tmp_path / "qa.json"
    yaml_path = tmp_path / "default_experiment.yaml"

    _write_jsonl(
        documents_path,
        [
            {
                "doc_id": "doc-1",
                "text": "Berlin is the capital of Germany and has 3.8 million people.",
            },
            {
                "doc_id": "doc-2",
                "text": "Lucerne is a city in Switzerland by the lake.",
            },
        ],
    )
    with open(qa_path, "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "query_id": "q1",
                    "doc_id": "doc-1",
                    "question": "Which city has 3.8 million people?",
                    "reference_answers": ["Berlin"],
                },
                {
                    "query_id": "q2",
                    "doc_id": "doc-2",
                    "question": "Which city is by the lake?",
                    "reference_answers": ["Lucerne"],
                },
            ],
            handle,
            indent=2,
        )

    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""
dataset_loader:
  type: local_json
  local_json:
    documents_path: {documents_path}
    qa_path: {qa_path}
chunking:
  strategy: fixed
  chunk_size: 4
  overlap: 1
  tokenizer_name: simple-tokenizer
retrieval:
  scope: per_document
  retrieve_k: 2
"""
        )

    monkeypatch.setattr(
        "chunked_pooling.late_chunk_runner.load_tokenizer",
        lambda *args, **kwargs: SimpleTokenizer(),
    )

    default_experiment = load_yaml_file(str(yaml_path))
    retrievers = parse_retriever_specs(["bm25"], default_experiment)
    resolved_config, notes = resolve_run_config(
        dataset_name="local_qa",
        default_experiment=default_experiment,
        retrievers=retrievers,
        output_root_override=str(tmp_path / "late_chunk_runs"),
        resume=False,
    )
    run_dir = run_late_chunking_experiment(
        resolved_config=resolved_config,
        default_experiment_path=str(yaml_path),
        notes=notes,
    )

    selected_doc_ids = json.loads((run_dir / "selection" / "selected_doc_ids.json").read_text())
    assert selected_doc_ids == ["doc-1", "doc-2"]

    chunk_rows = [
        json.loads(line)
        for line in (run_dir / "chunking" / "doc-1" / "chunks.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert chunk_rows
    assert chunk_rows[0]["chunk_id"].startswith("doc-1__chunk_")

    encoding_map = json.loads(
        (run_dir / "chunking" / "doc-1" / "encoding_map.json").read_text()
    )
    assert encoding_map["doc_id"] == "doc-1"
    assert encoding_map["encoders"] == {}

    retrieval_payload_rows = [
        json.loads(line)
        for line in (
            run_dir
            / "retrieval"
            / "retrieval_payloads__bm25__late_chunking__per_document.jsonl"
        )
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert len(retrieval_payload_rows) == 2
    assert retrieval_payload_rows[0]["indexed_doc_ids"] == ["doc-1"]
    assert retrieval_payload_rows[1]["indexed_doc_ids"] == ["doc-2"]
    assert retrieval_payload_rows[0]["retrieved_chunks"]

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert run_manifest["selected_documents_count"] == 2
    assert "No gold labels" in " ".join(run_manifest["notes"])


class FakeDenseRetriever:
    name = "fake"
    model_name = "fake-model"
    tokenizer_name = "fake-tokenizer"
    pooling = "mean"
    normalize = True
    distance_metric = "cosine"
    document_instruction = ""

    def __init__(self, fail_on_text=None):
        self.fail_on_text = fail_on_text
        self.encoded_documents = []
        self.tokenizer = SimpleTokenizer()

    def document_instruction_token_count(self):
        return 0

    def effective_max_tokens_per_forward(self, requested):
        return int(requested or 512)

    def encode_late_chunks(
        self,
        text,
        model_token_spans,
        max_tokens_per_forward,
        window_overlap_tokens,
    ):
        self.encoded_documents.append(text)
        if self.fail_on_text and self.fail_on_text in text:
            raise RuntimeError("intentional interruption")
        vectors = np.ones((len(model_token_spans), 2), dtype=np.float32)
        return vectors, {
            "effective_max_tokens_per_forward": max_tokens_per_forward,
            "segmentation_or_windowing_strategy": "single_forward",
            "full_model_input_token_count": len(text.split()),
            "encoder_windows": [],
        }

    def encode_queries(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)


def test_dense_resume_reuses_completed_document_checkpoints(tmp_path, monkeypatch):
    documents_path = tmp_path / "documents.jsonl"
    qa_path = tmp_path / "qa.json"
    yaml_path = tmp_path / "default_experiment.yaml"
    _write_jsonl(
        documents_path,
        [
            {"doc_id": "doc-1", "text": "Berlin is the capital of Germany."},
            {"doc_id": "doc-2", "text": "Lucerne is a city in Switzerland."},
        ],
    )
    qa_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "doc_id": "doc-1",
                    "question": "What is Germany's capital?",
                },
                {
                    "query_id": "q2",
                    "doc_id": "doc-2",
                    "question": "Which Swiss city is mentioned?",
                },
            ]
        ),
        encoding="utf-8",
    )
    yaml_path.write_text(
        f"""
dataset_loader:
  type: local_json
  local_json:
    documents_path: {documents_path}
    qa_path: {qa_path}
chunking:
  strategy: fixed
  chunk_size: 4
  overlap: 0
  tokenizer_name: simple-tokenizer
retrieval:
  scope: per_document
  retrieve_k: 2
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "chunked_pooling.late_chunk_runner.load_tokenizer",
        lambda *args, **kwargs: SimpleTokenizer(),
    )

    default_experiment = load_yaml_file(str(yaml_path))
    retrievers = parse_retriever_specs(
        [
            "name=fake,type=dense,model_name=fake-model,"
            "tokenizer_name=fake-tokenizer,normalize=true"
        ],
        default_experiment,
    )
    output_root = tmp_path / "late_chunk_runs"
    interrupted_retriever = FakeDenseRetriever(fail_on_text="Lucerne")
    monkeypatch.setattr(
        "chunked_pooling.late_chunk_runner.DenseRetriever.from_config",
        lambda *_args, **_kwargs: interrupted_retriever,
    )
    interrupted_config, interrupted_notes = resolve_run_config(
        dataset_name="local_qa",
        default_experiment=default_experiment,
        retrievers=retrievers,
        run_name_override="fake/c4_o0",
        output_root_override=str(output_root),
        resume=True,
    )

    with pytest.raises(RuntimeError, match="intentional interruption"):
        run_late_chunking_experiment(
            resolved_config=interrupted_config,
            default_experiment_path=str(yaml_path),
            notes=interrupted_notes,
        )

    checkpoint_dir = (
        output_root
        / "local_qa"
        / "fake"
        / "c4_o0"
        / "indexing"
        / "fake"
        / "document_checkpoints"
    )
    assert len(list(checkpoint_dir.glob("*.json"))) == 1
    assert len(interrupted_retriever.encoded_documents) == 2

    resumed_retriever = FakeDenseRetriever()
    monkeypatch.setattr(
        "chunked_pooling.late_chunk_runner.DenseRetriever.from_config",
        lambda *_args, **_kwargs: resumed_retriever,
    )
    resumed_config, resumed_notes = resolve_run_config(
        dataset_name="local_qa",
        default_experiment=default_experiment,
        retrievers=retrievers,
        run_name_override="fake/c4_o0",
        output_root_override=str(output_root),
        resume=True,
    )
    run_dir = run_late_chunking_experiment(
        resolved_config=resumed_config,
        default_experiment_path=str(yaml_path),
        notes=resumed_notes,
    )

    assert resumed_retriever.encoded_documents == [
        "Lucerne is a city in Switzerland."
    ]
    assert len(list(checkpoint_dir.glob("*.json"))) == 2
    assert (run_dir / "indexing" / "fake" / "chunk_embeddings.npy").is_file()
    assert (run_dir / "run_manifest.json").is_file()
