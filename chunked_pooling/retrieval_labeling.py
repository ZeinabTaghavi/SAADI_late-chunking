from __future__ import annotations

import json
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


QASPER_DATASET_NAMES = {"qasper", "allenai/qasper"}
LOOGLE_DATASET_NAMES = {"loogle", "bigai-nlco/loogle", "bigainlco/loogle"}
NARRATIVEQA_DATASET_NAMES = {"narrativeqa", "deepmind/narrativeqa"}
NOVELHOPQA_DATASET_NAMES = {
    "novelqa",
    "novelhopqa",
    "abhaygupta1266/novelhopqa",
}


@dataclass
class LabelRow:
    query_id: str
    doc_id: Optional[str]
    question: Optional[str]
    gold_chunk_ids: List[str]
    silver_chunk_ids: List[str]
    silver_chunk_groups: List[List[str]]
    relevant_ids: List[str]
    graded_relevance: "OrderedDict[str, float]"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalize_chunk_groups(raw: Any) -> List[List[str]]:
    groups: List[List[str]] = []
    if not isinstance(raw, list):
        return groups
    seen = set()
    for item in raw:
        if not isinstance(item, list):
            continue
        group = _ordered_unique(str(value) for value in item)
        if not group:
            continue
        key = tuple(group)
        if key in seen:
            continue
        seen.add(key)
        groups.append(group)
    return groups


def _as_id_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return _ordered_unique(str(value) for value in raw)
    if isinstance(raw, dict):
        positives = []
        for key, value in raw.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = 1.0 if value else 0.0
            if score > 0:
                positives.append(str(key))
        return _ordered_unique(positives)
    return _ordered_unique([str(raw)])


def _merge_graded_relevance(*mappings: "OrderedDict[str, float]") -> "OrderedDict[str, float]":
    merged: "OrderedDict[str, float]" = OrderedDict()
    for mapping in mappings:
        for key, value in mapping.items():
            if key not in merged or value > merged[key]:
                merged[key] = float(value)
    return merged


def _coerce_score_mapping(raw: Any) -> "OrderedDict[str, float]":
    scores: "OrderedDict[str, float]" = OrderedDict()
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score > 0:
                scores[str(key)] = score
    return scores


def _normalize_label_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        for key in ("rows", "queries", "examples", "data", "labels"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]

        dict_like_rows: List[Dict[str, Any]] = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            if "query_id" in value or "qid" in value or "id" in value:
                dict_like_rows.append(value)
            else:
                dict_like_rows.append(dict(value, query_id=str(key)))
        if dict_like_rows:
            return dict_like_rows

        if "query_id" in payload or "qid" in payload or "id" in payload:
            return [payload]

    raise ValueError(
        "Unsupported labels payload. Expected JSON/JSONL rows keyed by query id or a list of row objects."
    )


def load_label_rows(labels_file: Path) -> Dict[str, LabelRow]:
    if labels_file.suffix == ".jsonl":
        rows = _read_jsonl(labels_file)
    else:
        rows = _normalize_label_rows(_read_json(labels_file))

    labels_by_query: Dict[str, LabelRow] = {}
    for row in rows:
        query_id = str(
            row.get("query_id")
            or row.get("qid")
            or row.get("id")
            or ""
        ).strip()
        if not query_id:
            continue

        gold_chunk_ids = _as_id_list(row.get("gold_chunk_ids", row.get("gold")))
        silver_chunk_ids = _as_id_list(row.get("silver_chunk_ids", row.get("silver")))
        silver_chunk_groups = _normalize_chunk_groups(row.get("silver_chunk_groups"))
        relevant_ids = _as_id_list(
            row.get("relevant_ids")
            or row.get("relevant_chunk_ids")
            or row.get("chunk_ids")
            or row.get("relevant_doc_ids")
        )
        graded_relevance = _coerce_score_mapping(
            row.get("relevant_id_scores")
            or row.get("graded_relevance")
            or row.get("qrels")
        )

        if not relevant_ids and graded_relevance:
            relevant_ids = list(graded_relevance.keys())

        normalized = LabelRow(
            query_id=query_id,
            doc_id=(
                str(row["doc_id"]).strip()
                if row.get("doc_id") not in (None, "")
                else None
            ),
            question=(
                str(row["question"]).strip()
                if row.get("question") not in (None, "")
                else None
            ),
            gold_chunk_ids=gold_chunk_ids,
            silver_chunk_ids=silver_chunk_ids,
            silver_chunk_groups=silver_chunk_groups,
            relevant_ids=relevant_ids,
            graded_relevance=graded_relevance,
        )

        existing = labels_by_query.get(query_id)
        if existing is None:
            labels_by_query[query_id] = normalized
            continue

        merged_groups = existing.silver_chunk_groups + [
            group
            for group in normalized.silver_chunk_groups
            if group not in existing.silver_chunk_groups
        ]
        labels_by_query[query_id] = LabelRow(
            query_id=query_id,
            doc_id=existing.doc_id or normalized.doc_id,
            question=existing.question or normalized.question,
            gold_chunk_ids=_ordered_unique(existing.gold_chunk_ids + normalized.gold_chunk_ids),
            silver_chunk_ids=_ordered_unique(existing.silver_chunk_ids + normalized.silver_chunk_ids),
            silver_chunk_groups=merged_groups,
            relevant_ids=_ordered_unique(existing.relevant_ids + normalized.relevant_ids),
            graded_relevance=_merge_graded_relevance(
                existing.graded_relevance,
                normalized.graded_relevance,
            ),
        )

    return labels_by_query


def _tokenize_match_text(text: str) -> List[str]:
    normalized = (
        str(text or "")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", normalized.casefold())
    return [token for token in normalized.split() if token]


def _kmp_contains(text_tokens: List[str], pattern_tokens: List[str]) -> bool:
    if not pattern_tokens:
        return True
    prefix = [0] * len(pattern_tokens)
    j = 0
    for i in range(1, len(pattern_tokens)):
        while j and pattern_tokens[i] != pattern_tokens[j]:
            j = prefix[j - 1]
        if pattern_tokens[i] == pattern_tokens[j]:
            j += 1
            prefix[i] = j
    j = 0
    for token in text_tokens:
        while j and token != pattern_tokens[j]:
            j = prefix[j - 1]
        if token == pattern_tokens[j]:
            j += 1
            if j == len(pattern_tokens):
                return True
    return False


def classify_overlap(chunk_text: str, span_text: str) -> Tuple[str, int, str]:
    chunk_tokens = _tokenize_match_text(chunk_text)
    span_tokens = _tokenize_match_text(span_text)
    if not span_tokens:
        return ("full", 0, "")
    if _kmp_contains(chunk_tokens, span_tokens):
        return ("full", len(span_tokens), " ".join(span_tokens))

    max_end = min(len(chunk_tokens), len(span_tokens) - 1)
    overlap_at_end = max_end
    while overlap_at_end > 0 and chunk_tokens[-overlap_at_end:] != span_tokens[:overlap_at_end]:
        overlap_at_end -= 1

    max_start = min(len(chunk_tokens), len(span_tokens) - 1)
    overlap_at_start = max_start
    while overlap_at_start > 0 and chunk_tokens[:overlap_at_start] != span_tokens[-overlap_at_start:]:
        overlap_at_start -= 1

    if overlap_at_end > 0 or overlap_at_start > 0:
        if overlap_at_end >= overlap_at_start:
            return ("partial", overlap_at_end, " ".join(span_tokens[:overlap_at_end]))
        return ("partial", overlap_at_start, " ".join(span_tokens[-overlap_at_start:]))
    return ("none", 0, "")


def _normalize_retrieval_span_mode(value: Optional[str]) -> str:
    raw = str(value or "text").strip().lower()
    aliases = {
        "default": "text",
        "legacy": "text",
        "evidence": "text",
        "window_text": "window",
        "gold_window": "window",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"text", "window"}:
        raise ValueError(f"Unsupported retrieval_span_mode: {value!r}")
    return normalized


def _effective_retrieval_span_mode(qa_entry: Dict[str, Any]) -> str:
    override = qa_entry.get("retrieval_span_mode")
    if isinstance(override, str) and override.strip():
        return _normalize_retrieval_span_mode(override)
    return "text"


def _extract_answer_texts(qa_entry: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    raw = qa_entry.get("answers")
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out


def _extract_retrieval_spans(qa_entry: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    raw = qa_entry.get("retrieval_spans")
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    answers = qa_entry.get("answers")
    if not isinstance(answers, list):
        return out
    for item in answers:
        if not isinstance(item, dict):
            continue
        raw_span = item.get("span")
        if isinstance(raw_span, str):
            text = raw_span.strip()
            if text:
                out.append(text)
        elif isinstance(raw_span, list):
            out.extend(
                span.strip() for span in raw_span if isinstance(span, str) and span.strip()
            )
    return out


def _boundary_overlap_size(chunk_tokens: List[str], span_tokens: List[str]) -> int:
    max_end = min(len(chunk_tokens), len(span_tokens))
    overlap_at_end = max_end
    while overlap_at_end > 0 and chunk_tokens[-overlap_at_end:] != span_tokens[:overlap_at_end]:
        overlap_at_end -= 1

    max_start = min(len(chunk_tokens), len(span_tokens))
    overlap_at_start = max_start
    while overlap_at_start > 0 and chunk_tokens[:overlap_at_start] != span_tokens[-overlap_at_start:]:
        overlap_at_start -= 1
    return max(overlap_at_end, overlap_at_start)


def _window_overlaps_chunk(chunk_text: str, span_text: str) -> bool:
    chunk_tokens = _tokenize_match_text(chunk_text)
    span_tokens = _tokenize_match_text(span_text)
    if not chunk_tokens or not span_tokens:
        return False
    if _kmp_contains(chunk_tokens, span_tokens) or _kmp_contains(span_tokens, chunk_tokens):
        return True
    return _boundary_overlap_size(chunk_tokens, span_tokens) > 0


def _match_span_chunks(
    chunks: Sequence[Dict[str, Any]],
    span: str,
    *,
    retrieval_span_mode: str,
) -> Tuple[List[str], List[str]]:
    if retrieval_span_mode == "window":
        matched_ids = _ordered_unique(
            str(chunk["chunk_id"])
            for chunk in chunks
            if _window_overlaps_chunk(str(chunk.get("raw_text", "")), span)
        )
        if len(matched_ids) == 1:
            return matched_ids, []
        return [], matched_ids

    full_ids: List[str] = []
    partial_ids: List[str] = []
    for chunk in chunks:
        kind, _, _ = classify_overlap(str(chunk.get("raw_text", "")), span)
        if kind == "full":
            full_ids.append(str(chunk["chunk_id"]))
        elif kind == "partial":
            partial_ids.append(str(chunk["chunk_id"]))
    full_ids = _ordered_unique(full_ids)
    if full_ids:
        return full_ids, []
    return [], _ordered_unique(partial_ids)


def _build_support_targets(
    qa_entry: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[List[str]]]:
    retrieval_span_mode = _effective_retrieval_span_mode(qa_entry)
    spans = _extract_retrieval_spans(qa_entry)
    if not spans:
        spans = _extract_answer_texts(qa_entry)
        retrieval_span_mode = "text"
    if not spans:
        return [], [], []

    chunk_ids = {str(chunk["chunk_id"]) for chunk in chunks}
    gold_ids: List[str] = []
    silver_ids: List[str] = []
    silver_groups: List[List[str]] = []
    seen_groups: set[Tuple[str, ...]] = set()

    for span in spans:
        gold_matches, silver_group = _match_span_chunks(
            chunks,
            span,
            retrieval_span_mode=retrieval_span_mode,
        )
        gold_ids.extend(gold_matches)

        group = _ordered_unique(
            str(chunk_id) for chunk_id in silver_group if str(chunk_id) in chunk_ids
        )
        if not group:
            continue
        group_key = tuple(group)
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            silver_groups.append(group)
        silver_ids.extend(group)

    return _ordered_unique(gold_ids), _ordered_unique(silver_ids), silver_groups


def _chunks_by_doc_from_run(run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    chunk_root = run_dir / "chunking"
    if not chunk_root.exists():
        raise FileNotFoundError(f"Chunk directory not found under run: {chunk_root}")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for doc_dir in sorted(child for child in chunk_root.iterdir() if child.is_dir()):
        chunk_file = doc_dir / "chunks.jsonl"
        if not chunk_file.exists():
            continue
        out[doc_dir.name] = _read_jsonl(chunk_file)
    return out


def _qa_entries_from_run(run_dir: Path) -> List[Dict[str, Any]]:
    qa_entries_file = run_dir / "selection" / "qa_entries.json"
    if not qa_entries_file.exists():
        raise FileNotFoundError(
            f"Selected QA entries file not found under run: {qa_entries_file}"
        )
    payload = _read_json(qa_entries_file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {qa_entries_file}")
    return [row for row in payload if isinstance(row, dict)]


def infer_dataset_name_for_run(run_dir: Path) -> Optional[str]:
    run_manifest = run_dir / "run_manifest.json"
    if run_manifest.exists():
        payload = _read_json(run_manifest)
        dataset_name = payload.get("dataset_name")
        if isinstance(dataset_name, str) and dataset_name.strip():
            return dataset_name.strip()
    return None


def generate_label_rows_for_run(
    run_dir: Path,
    *,
    dataset_name: Optional[str] = None,
) -> Dict[str, LabelRow]:
    resolved_dataset_name = str(dataset_name or infer_dataset_name_for_run(run_dir) or "").strip().lower()
    if (
        resolved_dataset_name not in QASPER_DATASET_NAMES
        and resolved_dataset_name not in LOOGLE_DATASET_NAMES
        and resolved_dataset_name not in NARRATIVEQA_DATASET_NAMES
        and resolved_dataset_name not in NOVELHOPQA_DATASET_NAMES
    ):
        raise NotImplementedError(
            "Automatic in-process label generation is currently implemented only for qasper, loogle, narrativeqa, and novelhopqa."
        )

    chunks_by_doc = _chunks_by_doc_from_run(run_dir)
    qa_entries = _qa_entries_from_run(run_dir)

    labels_by_query: Dict[str, LabelRow] = {}
    for qa_entry in qa_entries:
        query_id = str(qa_entry.get("query_id") or "").strip()
        if not query_id:
            continue
        doc_id = str(
            qa_entry.get("document_id")
            or qa_entry.get("doc_id")
            or ""
        ).strip()
        doc_chunks = chunks_by_doc.get(doc_id, [])
        gold_ids, silver_ids, silver_groups = _build_support_targets(qa_entry, doc_chunks)
        relevant_ids = list(gold_ids or silver_ids)
        graded_relevance = OrderedDict((chunk_id, 1.0) for chunk_id in relevant_ids)
        labels_by_query[query_id] = LabelRow(
            query_id=query_id,
            doc_id=doc_id or None,
            question=(
                str(qa_entry["question"]).strip()
                if qa_entry.get("question") not in (None, "")
                else None
            ),
            gold_chunk_ids=list(gold_ids),
            silver_chunk_ids=list(silver_ids),
            silver_chunk_groups=[list(group) for group in silver_groups],
            relevant_ids=relevant_ids,
            graded_relevance=graded_relevance,
        )

    return labels_by_query
