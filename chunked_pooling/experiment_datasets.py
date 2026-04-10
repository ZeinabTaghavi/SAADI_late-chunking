from __future__ import annotations

import logging
import json
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import datasets


DATASET_PRESETS = {
    "scifact": {
        "path": "mteb/scifact",
        "revision": "0228b52cf27578f30900b9e5271d331663a030d7",
    },
    "scifactchunked": {
        "path": "mteb/scifact",
        "revision": "0228b52cf27578f30900b9e5271d331663a030d7",
    },
    "narrativeqa": {
        "path": "narrativeqa",
        "revision": "2e643e7363944af1c33a652d1c87320d0871c4e4",
        "name": "NarrativeQARetrieval",
    },
    "narrativeqachunked": {
        "path": "narrativeqa",
        "revision": "2e643e7363944af1c33a652d1c87320d0871c4e4",
        "name": "NarrativeQARetrieval",
    },
    "nfcorpus": {
        "path": "mteb/nfcorpus",
        "revision": "ec0fa4fe99da2ff19ca1214b7966684033a58814",
    },
    "nfcorpuschunked": {
        "path": "mteb/nfcorpus",
        "revision": "ec0fa4fe99da2ff19ca1214b7966684033a58814",
    },
    "quora": {
        "path": "mteb/quora",
        "revision": "e4e08e0b7dbe3c8700f0daef558ff32256715259",
    },
    "quorachunked": {
        "path": "mteb/quora",
        "revision": "e4e08e0b7dbe3c8700f0daef558ff32256715259",
    },
    "fiqa2018": {
        "path": "mteb/fiqa",
        "revision": "27a168819829fe9bcd655c2df245fb19452e8e06",
    },
    "fiqa2018chunked": {
        "path": "mteb/fiqa",
        "revision": "27a168819829fe9bcd655c2df245fb19452e8e06",
    },
    "treccovid": {
        "path": "mteb/trec-covid",
        "revision": "bb9466bac8153a0349341eb1b22e06409e78ef4e",
    },
    "treccovidchunked": {
        "path": "mteb/trec-covid",
        "revision": "bb9466bac8153a0349341eb1b22e06409e78ef4e",
    },
    "lembwikimqaretrieval": {
        "path": "dwzhu/LongEmbed",
        "revision": "10039a580487dacecf79db69166e17ace3ede392",
        "name": "2wikimqa",
    },
    "lembwikimqaretrievalchunked": {
        "path": "dwzhu/LongEmbed",
        "revision": "10039a580487dacecf79db69166e17ace3ede392",
        "name": "2wikimqa",
    },
    "lembsummscreenfdretrieval": {
        "path": "dwzhu/LongEmbed",
        "revision": "10039a580487dacecf79db69166e17ace3ede392",
        "name": "summ_screen_fd",
    },
    "lembsummscreenfdretrievalchunked": {
        "path": "dwzhu/LongEmbed",
        "revision": "10039a580487dacecf79db69166e17ace3ede392",
        "name": "summ_screen_fd",
    },
    "lembqmsumretrieval": {
        "path": "dwzhu/LongEmbed",
        "revision": "10039a580487dacecf79db69166e17ace3ede392",
        "name": "qmsum",
    },
    "lembqmsumretrievalchunked": {
        "path": "dwzhu/LongEmbed",
        "revision": "10039a580487dacecf79db69166e17ace3ede392",
        "name": "qmsum",
    },
    "lembneedleretrieval": {
        "path": "dwzhu/LongEmbed",
        "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
        "name": "needle",
        "split_mode": "context_length",
    },
    "lembneedleretrievalchunked": {
        "path": "dwzhu/LongEmbed",
        "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
        "name": "needle",
        "split_mode": "context_length",
    },
    "lembpasskeyretrieval": {
        "path": "dwzhu/LongEmbed",
        "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
        "name": "passkey",
        "split_mode": "context_length",
    },
    "lembpasskeyretrievalchunked": {
        "path": "dwzhu/LongEmbed",
        "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
        "name": "passkey",
        "split_mode": "context_length",
    },
}

logger = logging.getLogger(__name__)
QASPER_DATASET_NAMES = {"qasper", "allenai/qasper"}


@dataclass
class DatasetBundle:
    documents: "OrderedDict[str, Dict[str, object]]"
    qa_entries: List[Dict[str, object]]
    metadata: Dict[str, object]


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _load_json_rows(path: Path) -> List[Dict[str, object]]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected a list payload in {path}.")


def _normalize_reference_answers(value) -> Optional[List[object]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _coerce_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_coerce_to_text(item) for item in value if _coerce_to_text(item))
    if isinstance(value, dict):
        return "\n".join(
            _coerce_to_text(item)
            for item in value.values()
            if _coerce_to_text(item)
        )
    return str(value)


def _datasets_version_major() -> Optional[int]:
    try:
        raw = str(pkg_version("datasets")).strip()
        if not raw:
            return None
        return int(raw.split(".", 1)[0])
    except Exception:
        return None


def _resolve_qasper_config(requested: Optional[str], dataset_name: str = "allenai/qasper") -> Optional[str]:
    major = _datasets_version_major()
    kwargs = {}
    if major is None or major < 4:
        kwargs["trust_remote_code"] = True
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            configs = datasets.get_dataset_config_names(dataset_name, **kwargs) or []
        except TypeError:
            configs = datasets.get_dataset_config_names(dataset_name) or []
    except Exception:
        configs = []
    if not configs:
        return None
    if requested in configs:
        return requested
    return configs[0]


def _load_qasper_dataset(*, cfg: Optional[str]):
    major = _datasets_version_major()
    if major is not None and major >= 4:
        modern_kwargs = {"revision": "refs/convert/parquet"}
        try:
            return (
                datasets.load_dataset("allenai/qasper", name=cfg, **modern_kwargs)
                if cfg
                else datasets.load_dataset("allenai/qasper", **modern_kwargs)
            )
        except TypeError:
            return (
                datasets.load_dataset("allenai/qasper", name=cfg)
                if cfg
                else datasets.load_dataset("allenai/qasper")
            )
        except Exception as exc:
            msg = str(exc)
            if "Dataset scripts are no longer supported" in msg and "qasper.py" in msg:
                raise RuntimeError(
                    "Loading 'allenai/qasper' failed with datasets>=4 because script execution "
                    "is disabled and parquet fallback could not be resolved."
                ) from exc
            raise

    legacy_kwargs = {"trust_remote_code": True}
    try:
        try:
            return (
                datasets.load_dataset("allenai/qasper", name=cfg, **legacy_kwargs)
                if cfg
                else datasets.load_dataset("allenai/qasper", **legacy_kwargs)
            )
        except TypeError:
            return (
                datasets.load_dataset("allenai/qasper", name=cfg)
                if cfg
                else datasets.load_dataset("allenai/qasper")
            )
    except RuntimeError as exc:
        if "trust_remote_code" in str(exc):
            raise RuntimeError(
                "Loading 'allenai/qasper' requires script loading. Use `datasets<4` with "
                "`trust_remote_code=True`, or rely on the parquet-converted branch."
            ) from exc
        raise


def _qasper_documents(
    *,
    split: str,
    config_name: Optional[str],
) -> OrderedDict:
    cfg = _resolve_qasper_config(config_name)
    logger.info("Loading QASPER documents split=%s config=%s", split, cfg)
    dataset_dict = _load_qasper_dataset(cfg=cfg)
    documents = OrderedDict()
    for row in dataset_dict[split]:
        doc_id = str(row.get("id"))
        full_text = (row.get("full_text") or {}).get("paragraphs", "")
        if isinstance(full_text, list):
            if full_text and isinstance(full_text[0], list):
                text = "\n".join(
                    paragraph
                    for section in full_text
                    for paragraph in section
                    if isinstance(paragraph, str)
                )
            else:
                text = "\n".join(paragraph for paragraph in full_text if isinstance(paragraph, str))
        else:
            text = _coerce_to_text(full_text)
        documents[doc_id] = {
            **row,
            "doc_id": doc_id,
            "text": text,
        }
    return documents


def _qasper_qa_entries(
    *,
    split: str,
    n,
    selection_method: str,
    config_name: Optional[str],
) -> List[Dict[str, object]]:
    import random

    cfg = _resolve_qasper_config(config_name)
    logger.info(
        "Loading QASPER QA entries split=%s config=%s n=%s selection=%s",
        split,
        cfg,
        n,
        selection_method,
    )
    dataset_dict = _load_qasper_dataset(cfg=cfg)
    qa_entries = []
    for row in dataset_dict[split]:
        doc_id = str(row.get("id"))
        qas = row.get("qas") or {}
        questions = qas.get("question") or []
        answers_all = qas.get("answers") or []
        for question_index, (question, answer_dict) in enumerate(zip(questions, answers_all)):
            answer_texts: List[str] = []
            retrieval_spans: List[str] = []
            raw_answer_annotations = []
            for answer_index, answer in enumerate((answer_dict or {}).get("answer", []) or []):
                if not isinstance(answer, dict) or answer.get("unanswerable"):
                    continue
                if answer.get("extractive_spans"):
                    text = " ".join(
                        span
                        for span in answer.get("extractive_spans", [])
                        if isinstance(span, str)
                    )
                    answer_type = "extractive"
                elif isinstance(answer.get("free_form_answer"), str) and answer.get(
                    "free_form_answer", ""
                ).strip():
                    text = answer["free_form_answer"].strip()
                    answer_type = "free_form"
                elif answer.get("yes_no") is not None:
                    text = "Yes" if bool(answer["yes_no"]) else "No"
                    answer_type = "yes_no"
                else:
                    text = ""
                    answer_type = "unknown"
                evidence = answer.get("evidence") or answer.get("highlighted_evidence") or []
                if text:
                    answer_texts.append(text)
                if isinstance(evidence, str) and evidence.strip():
                    retrieval_spans.append(evidence.strip())
                elif isinstance(evidence, list):
                    retrieval_spans.extend(
                        span.strip()
                        for span in evidence
                        if isinstance(span, str) and span.strip()
                    )
                raw_answer_annotations.append(
                    {
                        "answer_index": answer_index,
                        "answer_type": answer_type,
                        "extractive_spans": answer.get("extractive_spans") or [],
                        "free_form_answer": answer.get("free_form_answer"),
                        "yes_no": answer.get("yes_no"),
                        "evidence": answer.get("evidence"),
                        "highlighted_evidence": answer.get("highlighted_evidence"),
                    }
                )
            if answer_texts or retrieval_spans:
                qa_entries.append(
                    {
                        "question": str(question).strip(),
                        "document_id": doc_id,
                        "answers": answer_texts,
                        "retrieval_spans": retrieval_spans,
                        "question_index": question_index,
                        "raw_answer_annotations": raw_answer_annotations,
                    }
                )

    total = len(qa_entries)
    if isinstance(n, str):
        n = total if n.lower() == "all" else int(n)
    n = min(int(n), total)
    indices = list(range(total))
    if n < total:
        indices = random.sample(indices, n) if selection_method == "random" else indices[:n]

    output = []
    for qa_index in indices:
        row = qa_entries[qa_index]
        output.append(
            {
                "query_id": f"qasper_{qa_index}",
                "source_qa_index": qa_index,
                "doc_id": row["document_id"],
                "document_id": row["document_id"],
                "question": row["question"],
                "reference_answers": list(row["answers"]),
                "answers": list(row["answers"]),
                "retrieval_spans": list(row["retrieval_spans"]),
                "evidence_spans": list(row["retrieval_spans"]),
                "question_index": row["question_index"],
                "raw_answer_annotations": list(row["raw_answer_annotations"]),
                "relevant_doc_ids": [row["document_id"]],
            }
        )
    return output


def load_qasper_bundle(loader_config: Dict[str, object]) -> DatasetBundle:
    split = str(loader_config.get("split") or "test")
    docs_config_name = loader_config.get("docs_config_name") or loader_config.get("config_name")
    qa_config_name = loader_config.get("qa_config_name") or loader_config.get("config_name")
    qa_n = loader_config.get("qa_n", "all")
    qa_selection_method = str(loader_config.get("qa_selection_method") or "first")

    documents = _qasper_documents(split=split, config_name=docs_config_name)
    qa_entries = _qasper_qa_entries(
        split=split,
        n=qa_n,
        selection_method=qa_selection_method,
        config_name=qa_config_name,
    )

    return DatasetBundle(
        documents=documents,
        qa_entries=qa_entries,
        metadata={
            "loader_type": "qasper",
            "split": split,
            "docs_config_name": docs_config_name,
            "qa_config_name": qa_config_name,
            "qa_n": qa_n,
            "qa_selection_method": qa_selection_method,
        },
    )


def load_local_json_bundle(loader_config: Dict[str, object]) -> DatasetBundle:
    local_json = loader_config["local_json"]
    documents_path = Path(str(local_json["documents_path"]))
    qa_path = Path(str(local_json["qa_path"]))

    document_id_field = str(local_json.get("document_id_field", "doc_id"))
    document_text_field = str(local_json.get("document_text_field", "text"))
    query_id_field = str(local_json.get("query_id_field", "query_id"))
    query_text_field = str(local_json.get("query_text_field", "question"))
    qa_doc_id_field = str(local_json.get("qa_doc_id_field", "doc_id"))
    reference_answers_field = str(
        local_json.get("reference_answers_field", "reference_answers")
    )

    documents = OrderedDict()
    for row in _load_json_rows(documents_path):
        doc_id = str(row[document_id_field])
        documents[doc_id] = {
            **row,
            "doc_id": doc_id,
            "text": str(row[document_text_field]),
        }

    qa_entries = []
    for row in _load_json_rows(qa_path):
        doc_id = row.get(qa_doc_id_field)
        entry = {
            **row,
            "query_id": str(row[query_id_field]),
            "question": str(row[query_text_field]),
            "doc_id": str(doc_id) if doc_id is not None else None,
            "reference_answers": _normalize_reference_answers(
                row.get(reference_answers_field)
            ),
        }
        if entry["doc_id"] is not None:
            entry["relevant_doc_ids"] = [entry["doc_id"]]
        qa_entries.append(entry)

    return DatasetBundle(
        documents=documents,
        qa_entries=qa_entries,
        metadata={"loader_type": "local_json"},
    )


def _load_hf_rows(spec: Dict[str, object], split_name: str) -> List[Dict[str, object]]:
    dataset_dict = datasets.load_dataset(
        path=spec["path"],
        name=spec.get("name"),
        revision=spec.get("revision"),
    )
    split = dataset_dict[split_name]

    if spec.get("split_mode") == "context_length":
        context_length = int(str(spec["requested_split"]).split("_")[1])
        split = split.filter(lambda row: row["context_length"] == context_length)

    return [dict(row) for row in split]


def load_task_registry_bundle(loader_config: Dict[str, object]) -> DatasetBundle:
    dataset_name = str(loader_config["dataset_name"]).lower()
    if dataset_name in QASPER_DATASET_NAMES:
        return load_qasper_bundle(loader_config)

    if dataset_name not in DATASET_PRESETS:
        supported = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(
            f"Unsupported dataset '{loader_config['dataset_name']}'. "
            f"Supported presets: {supported}, qasper"
        )

    spec = dict(DATASET_PRESETS[dataset_name])
    spec["requested_split"] = loader_config["split"]

    query_rows = _load_hf_rows(spec, "queries")
    corpus_rows = _load_hf_rows(spec, "corpus")
    qrel_rows = _load_hf_rows(spec, "qrels")

    qrels_by_query: Dict[str, OrderedDict] = OrderedDict()
    for row in qrel_rows:
        query_id = str(row.get("qid") or row.get("query_id"))
        doc_id = str(row["doc_id"])
        score = row.get("score", row.get("relevance", 1))
        qrels_by_query.setdefault(query_id, OrderedDict())[doc_id] = score

    documents = OrderedDict()
    prepend_title = bool(loader_config.get("prepend_title", True))
    for row in corpus_rows:
        doc_id = str(row["doc_id"])
        text = str(row["text"])
        title = row.get("title")
        final_text = text
        if prepend_title and title:
            final_text = f"{title} {text}".strip()
        documents[doc_id] = {
            **row,
            "doc_id": doc_id,
            "text": final_text,
            "source_text": text,
        }

    qa_entries = []
    for row in query_rows:
        query_id = str(row.get("qid") or row.get("query_id"))
        question = str(row.get("text") or row.get("question"))
        relevant_doc_ids = list(qrels_by_query.get(query_id, OrderedDict()).keys())
        doc_id = None
        if len(relevant_doc_ids) == 1:
            doc_id = relevant_doc_ids[0]
        entry = {
            **row,
            "query_id": query_id,
            "question": question,
            "doc_id": doc_id,
            "reference_answers": _normalize_reference_answers(
                row.get("reference_answers")
                or row.get("answers")
                or row.get("answer")
            ),
            "relevant_doc_ids": relevant_doc_ids,
            "qrels": dict(qrels_by_query.get(query_id, OrderedDict())),
        }
        qa_entries.append(entry)

    return DatasetBundle(
        documents=documents,
        qa_entries=qa_entries,
        metadata={
            "loader_type": "task_registry",
            "dataset_spec": spec,
        },
    )


def load_dataset_bundle(loader_config: Dict[str, object]) -> DatasetBundle:
    loader_type = str(loader_config["type"])
    if loader_type == "local_json":
        return load_local_json_bundle(loader_config)
    if loader_type == "task_registry":
        return load_task_registry_bundle(loader_config)
    raise ValueError(f"Unsupported dataset loader type: {loader_type}")


def select_dataset_subset(
    bundle: DatasetBundle,
    max_docs: Optional[int] = None,
    max_questions: Optional[int] = None,
    selected_doc_ids: Optional[Sequence[str]] = None,
    selected_query_ids: Optional[Sequence[str]] = None,
) -> DatasetBundle:
    query_filter = set(str(item) for item in selected_query_ids or [])
    doc_filter = set(str(item) for item in selected_doc_ids or [])

    qa_entries = []
    for entry in bundle.qa_entries:
        if query_filter and str(entry["query_id"]) not in query_filter:
            continue
        if doc_filter:
            direct_doc_id = entry.get("doc_id")
            relevant_doc_ids = [str(doc_id) for doc_id in entry.get("relevant_doc_ids", [])]
            if direct_doc_id is not None and str(direct_doc_id) in doc_filter:
                qa_entries.append(entry)
                continue
            if doc_filter.intersection(relevant_doc_ids):
                qa_entries.append(entry)
                continue
            continue
        qa_entries.append(entry)

    ordered_doc_ids = []
    if doc_filter:
        ordered_doc_ids.extend([doc_id for doc_id in selected_doc_ids if doc_id in bundle.documents])

    if not ordered_doc_ids:
        candidate_doc_ids = []
        for entry in qa_entries:
            if entry.get("doc_id") is not None:
                candidate_doc_ids.append(str(entry["doc_id"]))
            else:
                candidate_doc_ids.extend(
                    str(doc_id) for doc_id in entry.get("relevant_doc_ids", [])
                )
        ordered_doc_ids = _ordered_unique(doc_id for doc_id in candidate_doc_ids if doc_id in bundle.documents)

    if not ordered_doc_ids:
        ordered_doc_ids = list(bundle.documents.keys())

    if max_docs is not None:
        selected_doc_ids_ordered = list(ordered_doc_ids[:max_docs])
        selected_doc_ids_set = set(selected_doc_ids_ordered)
        if not doc_filter and len(selected_doc_ids_ordered) < max_docs:
            for doc_id in bundle.documents.keys():
                if doc_id in selected_doc_ids_set:
                    continue
                selected_doc_ids_ordered.append(doc_id)
                selected_doc_ids_set.add(doc_id)
                if len(selected_doc_ids_ordered) >= max_docs:
                    break
        ordered_doc_ids = selected_doc_ids_ordered
        allowed_docs = set(ordered_doc_ids)
        filtered_entries = []
        for entry in qa_entries:
            direct_doc_id = entry.get("doc_id")
            relevant_doc_ids = [str(doc_id) for doc_id in entry.get("relevant_doc_ids", [])]
            if direct_doc_id is not None and str(direct_doc_id) in allowed_docs:
                filtered_entries.append(entry)
                continue
            if not relevant_doc_ids:
                filtered_entries.append(entry)
                continue
            if allowed_docs.intersection(relevant_doc_ids):
                filtered_entries.append(entry)
        qa_entries = filtered_entries

    if max_questions is not None:
        qa_entries = qa_entries[:max_questions]

    documents = OrderedDict(
        (doc_id, bundle.documents[doc_id]) for doc_id in ordered_doc_ids if doc_id in bundle.documents
    )
    return DatasetBundle(documents=documents, qa_entries=qa_entries, metadata=bundle.metadata)
