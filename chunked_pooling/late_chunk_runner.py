from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from chunked_pooling.experiment_chunking import (
    build_chunk_records,
    build_encoder_chunk_mappings,
    chunk_records_to_index_maps,
    iter_flatten_chunks,
    make_chunking_signature,
)
from chunked_pooling.experiment_datasets import (
    load_dataset_bundle,
    select_dataset_subset,
)
from chunked_pooling.experiment_retrievers import BM25Index, DenseRetriever


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_yaml(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _ru_maxrss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _gpu_memory_snapshot() -> Tuple[Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return None, None
    device = torch.cuda.current_device()
    current = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
    peak = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    return float(current), float(peak)


def _resource_usage_record(
    phase: str,
    doc_id: Optional[str] = None,
    query_id: Optional[str] = None,
    retriever_name: Optional[str] = None,
) -> Dict[str, object]:
    gpu_memory, gpu_peak_memory = _gpu_memory_snapshot()
    return {
        "phase": phase,
        "doc_id": doc_id,
        "query_id": query_id,
        "retriever_name": retriever_name,
        "cpu_memory_mb": round(_ru_maxrss_mb(), 4),
        "gpu_memory_mb": None if gpu_memory is None else round(gpu_memory, 4),
        "gpu_peak_memory_mb": (
            None if gpu_peak_memory is None else round(gpu_peak_memory, 4)
        ),
    }


def _package_versions(packages: Sequence[str]) -> Dict[str, Optional[str]]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _hardware_summary() -> Dict[str, object]:
    gpu_devices = []
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            gpu_devices.append(
                {
                    "device_index": device_index,
                    "name": torch.cuda.get_device_name(device_index),
                }
            )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "gpu_devices": gpu_devices,
    }


def _relative_path(path: Path, run_dir: Path) -> str:
    return str(path.relative_to(run_dir))


def _serialize_qa_entry(entry: Dict[str, object]) -> Dict[str, object]:
    return json.loads(json.dumps(entry))


def _document_row(doc_id: str, record: Dict[str, object], token_count: int) -> Dict[str, object]:
    return {
        "doc_id": doc_id,
        "text": record["text"],
        "character_count": len(str(record["text"])),
        "token_count": token_count,
    }


def _load_or_build_chunks(
    run_dir: Path,
    documents: "OrderedDict[str, Dict[str, object]]",
    chunking_config: Dict[str, object],
    resume: bool,
) -> Tuple[
    Dict[str, List[Dict[str, object]]],
    Dict[str, int],
    str,
    str,
]:
    chunking_signature = make_chunking_signature(chunking_config)
    tokenizer_name = str(chunking_config["tokenizer_name"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    chunks_by_doc: Dict[str, List[Dict[str, object]]] = {}
    token_counts: Dict[str, int] = {}

    for doc_id, record in documents.items():
        chunk_dir = run_dir / "chunking" / doc_id
        chunk_file = chunk_dir / "chunks.jsonl"
        encoding_map_file = chunk_dir / "encoding_map.json"

        if resume and chunk_file.exists():
            rows = _read_jsonl(chunk_file)
            chunks_by_doc[doc_id] = rows
            token_counts[doc_id] = max((int(row["token_end"]) for row in rows), default=0)
            if not encoding_map_file.exists():
                _write_json(
                    encoding_map_file,
                    {
                        "doc_id": doc_id,
                        "canonical_chunking_tokenizer_name": tokenizer_name,
                        "canonical_full_document_token_count": token_counts[doc_id],
                        "encoders": {},
                    },
                )
            continue

        _ensure_directory(chunk_dir)
        chunks, token_count = build_chunk_records(
            doc_id=doc_id,
            text=str(record["text"]),
            tokenizer=tokenizer,
            chunking_config=chunking_config,
            chunking_signature=chunking_signature,
        )
        _write_jsonl(chunk_file, chunks)
        _write_json(
            encoding_map_file,
            {
                "doc_id": doc_id,
                "canonical_chunking_tokenizer_name": tokenizer_name,
                "canonical_full_document_token_count": token_count,
                "encoders": {},
            },
        )
        chunks_by_doc[doc_id] = chunks
        token_counts[doc_id] = token_count

    return chunks_by_doc, token_counts, chunking_signature, tokenizer_name


def _update_encoding_map_file(
    run_dir: Path,
    doc_id: str,
    retriever_name: str,
    encoder_payload: Dict[str, object],
) -> None:
    path = run_dir / "chunking" / doc_id / "encoding_map.json"
    if path.exists():
        payload = _read_json(path)
    else:
        payload = {"doc_id": doc_id, "encoders": {}}
    payload.setdefault("encoders", {})[retriever_name] = encoder_payload
    _write_json(path, payload)


def _retrieval_scope_for_query(
    query: Dict[str, object],
    retrieval_scope: str,
    selected_doc_ids: Sequence[str],
) -> List[str]:
    if retrieval_scope == "global":
        return list(selected_doc_ids)

    if retrieval_scope == "per_document":
        if query.get("doc_id") is not None:
            return [str(query["doc_id"])]
        relevant_doc_ids = [str(doc_id) for doc_id in query.get("relevant_doc_ids", [])]
        if len(relevant_doc_ids) == 1:
            return relevant_doc_ids
        if relevant_doc_ids:
            return relevant_doc_ids
    return list(selected_doc_ids)


def run_late_chunking_experiment(
    resolved_config: Dict[str, object],
    default_experiment_path: str,
    notes: Sequence[str],
) -> Path:
    dataset_name = str(resolved_config["dataset_name"])
    run_name = str(resolved_config["run_name"])
    run_dir = Path(str(resolved_config["output_root"])) / dataset_name / run_name

    config_dir = run_dir / "config"
    selection_dir = run_dir / "selection"
    corpus_dir = run_dir / "corpus"
    indexing_dir = run_dir / "indexing"
    retrieval_dir = run_dir / "retrieval"
    profiling_dir = run_dir / "profiling"

    for directory in (
        config_dir,
        selection_dir,
        corpus_dir,
        indexing_dir,
        retrieval_dir,
        profiling_dir,
    ):
        _ensure_directory(directory)

    with open(default_experiment_path, "r", encoding="utf-8") as handle:
        saved_yaml_text = handle.read()
    with open(config_dir / "default_experiment.yaml", "w", encoding="utf-8") as handle:
        handle.write(saved_yaml_text)

    _write_yaml(config_dir / "late_chunking_run.yaml", resolved_config)

    dataset_bundle = load_dataset_bundle(resolved_config["dataset_loader"])
    selected_bundle = select_dataset_subset(
        bundle=dataset_bundle,
        max_docs=resolved_config["dataset_loader"].get("max_docs"),
        max_questions=resolved_config["dataset_loader"].get("max_questions"),
        selected_doc_ids=resolved_config["dataset_loader"].get("selected_doc_ids"),
        selected_query_ids=resolved_config["dataset_loader"].get("selected_query_ids"),
    )

    selected_doc_ids = list(selected_bundle.documents.keys())
    _write_json(selection_dir / "selected_doc_ids.json", selected_doc_ids)
    _write_json(
        selection_dir / "qa_entries.json",
        [_serialize_qa_entry(entry) for entry in selected_bundle.qa_entries],
    )

    resume = bool(resolved_config.get("resume", True))
    (
        chunks_by_doc,
        canonical_token_counts,
        chunking_signature,
        canonical_tokenizer_name,
    ) = _load_or_build_chunks(
        run_dir=run_dir,
        documents=selected_bundle.documents,
        chunking_config=resolved_config["chunking"],
        resume=resume,
    )

    document_rows = [
        _document_row(doc_id, selected_bundle.documents[doc_id], canonical_token_counts[doc_id])
        for doc_id in selected_doc_ids
    ]
    _write_jsonl(corpus_dir / "documents.jsonl", document_rows)

    flattened_chunks = iter_flatten_chunks(chunks_by_doc, selected_doc_ids)
    chunk_ids, _, doc_to_indices = chunk_records_to_index_maps(flattened_chunks)

    document_encoding_path = profiling_dir / "document_encoding_times.jsonl"
    index_build_path = profiling_dir / "index_build_times.jsonl"
    resource_usage_path = profiling_dir / "resource_usage.jsonl"
    document_encoding_rows = (
        _read_jsonl(document_encoding_path)
        if resume and document_encoding_path.exists()
        else []
    )
    index_build_rows = (
        _read_jsonl(index_build_path) if resume and index_build_path.exists() else []
    )
    resource_usage_rows = (
        _read_jsonl(resource_usage_path) if resume and resource_usage_path.exists() else []
    )

    retriever_summaries = []
    index_manifest = {
        "dataset_name": dataset_name,
        "run_name": run_name,
        "retrieval_scope": resolved_config["retrieval"]["scope"],
        "selected_doc_ids": selected_doc_ids,
        "chunking_signature": chunking_signature,
        "retrievers": retriever_summaries,
    }

    dense_indexes: Dict[str, Dict[str, object]] = {}
    lexical_indexes: Dict[str, BM25Index] = {}

    for retriever_config in resolved_config["retrievers"]:
        retriever_name = str(retriever_config["name"])
        retriever_dir = indexing_dir / retriever_name
        _ensure_directory(retriever_dir)

        build_start = time.perf_counter()
        build_time_ms = 0.0
        index_stats_path = retriever_dir / "index_stats.json"
        indexed_chunk_count = len(flattened_chunks)
        peak_cpu_memory_mb = None
        peak_gpu_memory_mb = None

        document_encoding_rows = [
            row
            for row in document_encoding_rows
            if row.get("retriever_name") != retriever_name
        ]
        index_build_rows = [
            row for row in index_build_rows if row.get("retriever_name") != retriever_name
        ]
        resource_usage_rows = [
            row
            for row in resource_usage_rows
            if row.get("retriever_name") != retriever_name
            or row.get("phase") not in {"document_encoding", "index_build", "query_retrieval"}
        ]

        if retriever_config["type"] == "bm25":
            bm25_index = BM25Index(flattened_chunks)
            lexical_indexes[retriever_name] = bm25_index
            build_time_ms = (time.perf_counter() - build_start) * 1000.0
            peak_cpu_memory_mb = _ru_maxrss_mb()
            _, peak_gpu_memory_mb = _gpu_memory_snapshot()
            index_stats = {
                "retriever_name": retriever_name,
                "retriever_model": retriever_config["model_name"],
                "indexed_doc_count": len(selected_doc_ids),
                "indexed_chunk_count": indexed_chunk_count,
                "build_time_ms": round(build_time_ms, 4),
                "embedding_dimension": None,
                "normalization": False,
                "distance_metric": "bm25",
                "peak_cpu_memory_mb": None if peak_cpu_memory_mb is None else round(peak_cpu_memory_mb, 4),
                "peak_gpu_memory_mb": None if peak_gpu_memory_mb is None else round(peak_gpu_memory_mb, 4),
            }
        else:
            retriever = DenseRetriever.from_config(retriever_config)
            dense_matrix_file = retriever_dir / "chunk_embeddings.npy"
            dense_chunk_ids_file = retriever_dir / "chunk_ids.json"
            dense_indexes[retriever_name] = {}
            reused_dense_index = resume and dense_matrix_file.exists() and dense_chunk_ids_file.exists()

            if reused_dense_index:
                embedding_matrix = np.load(dense_matrix_file)
                stored_chunk_ids = _read_json(dense_chunk_ids_file)
                if stored_chunk_ids != chunk_ids:
                    raise ValueError(
                        f"Stored chunk_ids for retriever '{retriever_name}' do not match "
                        "the current chunk order."
                    )
                build_time_ms = 0.0
            else:
                vectors = []
                instruction_token_count = retriever.document_instruction_token_count()
                for doc_id in selected_doc_ids:
                    doc_start = time.perf_counter()
                    model_token_spans, per_chunk_metadata, encoder_token_count = build_encoder_chunk_mappings(
                        chunk_records=chunks_by_doc[doc_id],
                        text=str(selected_bundle.documents[doc_id]["text"]),
                        tokenizer=retriever.tokenizer,
                        instruction_token_count=instruction_token_count,
                    )

                    pooling_start = time.perf_counter()
                    doc_vectors, window_metadata = retriever.encode_late_chunks(
                        text=str(selected_bundle.documents[doc_id]["text"]),
                        model_token_spans=model_token_spans,
                        max_tokens_per_forward=resolved_config["late_chunking"].get(
                            "max_tokens_per_forward"
                        )
                        or retriever.max_length,
                        window_overlap_tokens=int(
                            resolved_config["late_chunking"].get("window_overlap_tokens") or 0
                        ),
                    )
                    pooling_time_ms = (time.perf_counter() - pooling_start) * 1000.0
                    document_time_ms = (time.perf_counter() - doc_start) * 1000.0

                    vectors.append(doc_vectors)
                    document_encoding_rows.append(
                        {
                            "doc_id": doc_id,
                            "retriever_name": retriever_name,
                            "document_encoding_time_ms": round(document_time_ms, 4),
                            "late_chunk_pooling_time_ms": round(pooling_time_ms, 4),
                        }
                    )
                    resource_usage_rows.append(
                        _resource_usage_record(
                            phase="document_encoding",
                            doc_id=doc_id,
                            retriever_name=retriever_name,
                        )
                    )
                    encoder_payload = {
                        "doc_id": doc_id,
                        "encoder_model": retriever.model_name,
                        "tokenizer_name": retriever.tokenizer_name,
                        "full_document_token_count": encoder_token_count,
                        "max_tokens_per_forward": resolved_config["late_chunking"].get(
                            "max_tokens_per_forward"
                        )
                        or retriever.max_length,
                        "segmentation_or_windowing_strategy": window_metadata[
                            "segmentation_or_windowing_strategy"
                        ],
                        "stride_or_overlap_tokens": int(
                            resolved_config["late_chunking"].get("window_overlap_tokens") or 0
                        ),
                        "instruction_prefix": retriever.document_instruction,
                        "instruction_token_count": instruction_token_count,
                        "full_model_input_token_count": window_metadata[
                            "full_model_input_token_count"
                        ],
                        "encoder_windows": window_metadata["encoder_windows"],
                        "chunk_pooling_map": per_chunk_metadata,
                    }
                    _update_encoding_map_file(
                        run_dir=run_dir,
                        doc_id=doc_id,
                        retriever_name=retriever_name,
                        encoder_payload=encoder_payload,
                    )

                embedding_matrix = np.vstack(vectors) if vectors else np.zeros((0, 0))
                np.save(dense_matrix_file, embedding_matrix)
                _write_json(dense_chunk_ids_file, chunk_ids)
                build_time_ms = (time.perf_counter() - build_start) * 1000.0

            dense_indexes[retriever_name]["retriever"] = retriever
            dense_indexes[retriever_name]["embedding_matrix"] = embedding_matrix
            peak_cpu_memory_mb = _ru_maxrss_mb()
            _, peak_gpu_memory_mb = _gpu_memory_snapshot()
            index_stats = {
                "retriever_name": retriever_name,
                "retriever_model": retriever.model_name,
                "indexed_doc_count": len(selected_doc_ids),
                "indexed_chunk_count": indexed_chunk_count,
                "build_time_ms": round(build_time_ms, 4),
                "embedding_dimension": (
                    None if embedding_matrix.size == 0 else int(embedding_matrix.shape[1])
                ),
                "normalization": retriever.normalize,
                "distance_metric": retriever.distance_metric,
                "peak_cpu_memory_mb": None if peak_cpu_memory_mb is None else round(peak_cpu_memory_mb, 4),
                "peak_gpu_memory_mb": None if peak_gpu_memory_mb is None else round(peak_gpu_memory_mb, 4),
            }

        _write_json(index_stats_path, index_stats)
        index_build_rows.append(
            {
                "retriever_name": retriever_name,
                "build_time_ms": round(build_time_ms, 4),
            }
        )
        resource_usage_rows.append(
            _resource_usage_record(
                phase="index_build",
                retriever_name=retriever_name,
            )
        )
        retriever_summaries.append(
            {
                "retriever_name": retriever_name,
                "retriever_model": retriever_config["model_name"],
                "index_type": "bm25" if retriever_config["type"] == "bm25" else "dense_matrix",
                "indexed_doc_count": len(selected_doc_ids),
                "indexed_chunk_count": indexed_chunk_count,
                "chunk_id_ordering_rule": "selected_doc_ids order, then chunk_index ascending",
                "index_stats_path": _relative_path(index_stats_path, run_dir),
            }
        )

    _write_json(indexing_dir / "index_manifest.json", index_manifest)
    _write_jsonl(document_encoding_path, document_encoding_rows)
    _write_jsonl(index_build_path, index_build_rows)

    artifact_paths = {
        "default_experiment_yaml": _relative_path(config_dir / "default_experiment.yaml", run_dir),
        "late_chunking_run_yaml": _relative_path(config_dir / "late_chunking_run.yaml", run_dir),
        "selected_doc_ids_json": _relative_path(selection_dir / "selected_doc_ids.json", run_dir),
        "qa_entries_json": _relative_path(selection_dir / "qa_entries.json", run_dir),
        "documents_jsonl": _relative_path(corpus_dir / "documents.jsonl", run_dir),
        "index_manifest_json": _relative_path(indexing_dir / "index_manifest.json", run_dir),
        "document_encoding_times_jsonl": _relative_path(
            profiling_dir / "document_encoding_times.jsonl",
            run_dir,
        ),
        "index_build_times_jsonl": _relative_path(
            profiling_dir / "index_build_times.jsonl",
            run_dir,
        ),
    }

    for retriever_config in resolved_config["retrievers"]:
        retriever_name = str(retriever_config["name"])
        payload_path = (
            retrieval_dir
            / f"retrieval_payloads__{retriever_name}__late_chunking__per_document.jsonl"
        )
        raw_results_path = (
            retrieval_dir
            / f"retrieval_results_raw__{retriever_name}__late_chunking__per_document.json"
        )
        query_times_path = profiling_dir / f"query_times__{retriever_name}.jsonl"

        retrieval_payload_rows = []
        raw_result_rows = []
        query_time_rows = []

        if retriever_config["type"] == "bm25":
            index = lexical_indexes[retriever_name]
        else:
            dense_index = dense_indexes[retriever_name]
            retriever = dense_index["retriever"]
            embedding_matrix = dense_index["embedding_matrix"]

        for query in selected_bundle.qa_entries:
            start = time.perf_counter()
            indexed_doc_ids = _retrieval_scope_for_query(
                query=query,
                retrieval_scope=str(resolved_config["retrieval"]["scope"]),
                selected_doc_ids=selected_doc_ids,
            )
            candidate_indices = []
            for doc_id in indexed_doc_ids:
                candidate_indices.extend(doc_to_indices.get(doc_id, []))

            if retriever_config["type"] == "bm25":
                ranked_indices, scores = index.search(
                    query_text=str(query["question"]),
                    top_k=int(resolved_config["retrieval"]["retrieve_k"]),
                    candidate_indices=candidate_indices,
                )
            else:
                query_embedding = retriever.encode_queries([str(query["question"])])[0]
                if candidate_indices:
                    candidate_matrix = embedding_matrix[candidate_indices]
                    scores_array = candidate_matrix @ query_embedding
                    local_order = np.argsort(-scores_array)
                    top_local = local_order[
                        : min(int(resolved_config["retrieval"]["retrieve_k"]), len(local_order))
                    ]
                    ranked_indices = [candidate_indices[int(index)] for index in top_local]
                    scores = [float(scores_array[int(index)]) for index in top_local]
                else:
                    ranked_indices, scores = [], []

            latency_ms = (time.perf_counter() - start) * 1000.0
            query_time_rows.append(
                {
                    "query_id": str(query["query_id"]),
                    "doc_id": query.get("doc_id"),
                    "retriever_name": retriever_name,
                    "retrieval_latency_ms": round(latency_ms, 4),
                }
            )
            resource_usage_rows.append(
                _resource_usage_record(
                    phase="query_retrieval",
                    query_id=str(query["query_id"]),
                    retriever_name=retriever_name,
                )
            )

            retrieved_chunks = []
            retrieved_chunk_ids = []
            retrieved_indices = []
            for global_index, score in zip(ranked_indices, scores):
                chunk = flattened_chunks[global_index]
                retrieved_chunk_ids.append(str(chunk["chunk_id"]))
                retrieved_indices.append(int(global_index))
                retrieved_chunks.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["chunk_index"],
                        "raw_text": chunk["raw_text"],
                        "score": float(score),
                    }
                )

            retrieval_payload_rows.append(
                {
                    "query_id": str(query["query_id"]),
                    "doc_id": query.get("doc_id"),
                    "question": str(query["question"]),
                    "retriever_name": retriever_name,
                    "retrieval_scope": str(resolved_config["retrieval"]["scope"]),
                    "indexed_doc_ids": indexed_doc_ids,
                    "indexed_chunk_count": len(candidate_indices),
                    "retrieve_k": int(resolved_config["retrieval"]["retrieve_k"]),
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "retrieved_indices": retrieved_indices,
                    "scores": [float(score) for score in scores],
                    "retrieved_chunks": retrieved_chunks,
                    "retrieval_latency_ms": round(latency_ms, 4),
                }
            )
            raw_result_rows.append(
                {
                    "query_id": str(query["query_id"]),
                    "doc_id": query.get("doc_id"),
                    "question": str(query["question"]),
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "retrieved_indices": retrieved_indices,
                    "scores": [float(score) for score in scores],
                }
            )

        _write_jsonl(payload_path, retrieval_payload_rows)
        _write_json(raw_results_path, raw_result_rows)
        _write_jsonl(query_times_path, query_time_rows)

        artifact_paths[f"retrieval_payloads_{retriever_name}"] = _relative_path(
            payload_path, run_dir
        )
        artifact_paths[f"retrieval_results_raw_{retriever_name}"] = _relative_path(
            raw_results_path, run_dir
        )
        artifact_paths[f"query_times_{retriever_name}"] = _relative_path(
            query_times_path, run_dir
        )

    _write_jsonl(resource_usage_path, resource_usage_rows)
    artifact_paths["resource_usage_jsonl"] = _relative_path(
        resource_usage_path,
        run_dir,
    )

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "run_name": run_name,
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "package_versions": _package_versions(
            [
                "datasets",
                "transformers",
                "torch",
                "numpy",
                "pyyaml",
                "mteb",
            ]
        ),
        "hardware_summary": _hardware_summary(),
        "retrievers_used": [retriever["name"] for retriever in resolved_config["retrievers"]],
        "selected_documents_count": len(selected_doc_ids),
        "selected_questions_count": len(selected_bundle.qa_entries),
        "artifact_paths": artifact_paths,
        "config_paths": {
            "default_experiment_yaml": artifact_paths["default_experiment_yaml"],
            "late_chunking_run_yaml": artifact_paths["late_chunking_run_yaml"],
        },
        "notes": list(notes)
        + [
            f"Canonical chunking tokenizer: {canonical_tokenizer_name}",
            "No gold labels, retrieval recall metrics, or QA metrics were computed.",
            "Multi-retriever encoding_map.json files store retriever-specific pooling spans under chunking/<doc_id>/encoding_map.json -> encoders.<retriever_name>.",
        ],
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)
    return run_dir
