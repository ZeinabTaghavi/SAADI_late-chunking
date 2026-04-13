from __future__ import annotations

import json
import math
import shlex
import sys
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_K_VALUES = (5, 10)
LABEL_SOURCE_CHOICES = (
    "auto",
    "gold_chunk_ids",
    "silver_chunk_ids",
    "relevant_ids",
)


@dataclass
class RawRunRow:
    query_id: str
    doc_id: Optional[str]
    question: Optional[str]
    retrieved_chunk_ids: List[str]
    scores: List[Optional[float]]
    source_path: str


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")


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
    if isinstance(raw, list):
        return _ordered_unique(str(value) for value in raw)
    if isinstance(raw, tuple):
        return _ordered_unique(str(value) for value in raw)
    if isinstance(raw, set):
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


def _as_ranked_id_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        ranked_ids: List[str] = []
        for value in raw:
            item = str(value).strip()
            if item:
                ranked_ids.append(item)
        return ranked_ids
    item = str(raw).strip()
    return [item] if item else []


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

        gold_chunk_ids = _as_id_list(
            row.get("gold_chunk_ids", row.get("gold"))
        )
        silver_chunk_ids = _as_id_list(
            row.get("silver_chunk_ids", row.get("silver"))
        )
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
            silver_chunk_ids=_ordered_unique(
                existing.silver_chunk_ids + normalized.silver_chunk_ids
            ),
            silver_chunk_groups=merged_groups,
            relevant_ids=_ordered_unique(existing.relevant_ids + normalized.relevant_ids),
            graded_relevance=_merge_graded_relevance(
                existing.graded_relevance,
                normalized.graded_relevance,
            ),
        )

    return labels_by_query


def _resolve_candidate_paths(run_dir: Path, run_manifest: Optional[Dict[str, Any]]) -> List[Path]:
    candidates: List[Path] = []
    if run_manifest:
        artifact_paths = run_manifest.get("artifact_paths") or {}
        if isinstance(artifact_paths, dict):
            for key, value in artifact_paths.items():
                if not isinstance(value, str):
                    continue
                if "retrieval_payloads_" not in key and "retrieval_results_raw_" not in key:
                    continue
                candidates.append(run_dir / value)

    retrieval_dir = run_dir / "retrieval"
    if retrieval_dir.exists():
        candidates.extend(sorted(retrieval_dir.glob("retrieval_payloads__*.jsonl")))
        candidates.extend(sorted(retrieval_dir.glob("retrieval_results_raw__*.json")))

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def select_raw_results_file(
    run_dir: Path,
    *,
    explicit_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.exists() else None

    if explicit_path is not None:
        raw_path = explicit_path if explicit_path.is_absolute() else run_dir / explicit_path
        if not raw_path.exists():
            raise FileNotFoundError("Raw results file not found: %s" % raw_path)
        return raw_path, run_manifest, notes

    candidates = _resolve_candidate_paths(run_dir, run_manifest)
    payload_candidates = [path for path in candidates if path.name.startswith("retrieval_payloads__")]
    raw_candidates = [path for path in candidates if path.name.startswith("retrieval_results_raw__")]

    if len(payload_candidates) == 1:
        notes.append(
            "Selected retrieval payload JSONL because it preserves ranked chunk ids, scores, and query metadata."
        )
        return payload_candidates[0], run_manifest, notes
    if len(payload_candidates) > 1:
        raise ValueError(
            "Multiple retrieval payload files were found under the run directory. "
            "Pass --raw-results-file to choose one."
        )
    if len(raw_candidates) == 1:
        notes.append(
            "Fell back to retrieval_results_raw JSON because no retrieval payload JSONL was available."
        )
        return raw_candidates[0], run_manifest, notes
    if len(raw_candidates) > 1:
        raise ValueError(
            "Multiple raw retrieval result files were found under the run directory. "
            "Pass --raw-results-file to choose one."
        )
    raise FileNotFoundError(
        "No retrieval payload or raw retrieval result file was found under the run directory."
    )


def load_raw_run_rows(raw_results_file: Path) -> List[RawRunRow]:
    if raw_results_file.suffix == ".jsonl":
        rows = _read_jsonl(raw_results_file)
    else:
        payload = _read_json(raw_results_file)
        if not isinstance(payload, list):
            raise ValueError(
                "Unsupported raw retrieval payload. Expected a JSON list or JSONL rows."
            )
        rows = [row for row in payload if isinstance(row, dict)]

    normalized_rows: List[RawRunRow] = []
    for row in rows:
        query_id = str(
            row.get("query_id")
            or row.get("qid")
            or row.get("id")
            or ""
        ).strip()
        if not query_id:
            continue
        raw_scores = row.get("scores", [])
        scores: List[Optional[float]] = []
        if isinstance(raw_scores, list):
            for value in raw_scores:
                try:
                    scores.append(float(value))
                except (TypeError, ValueError):
                    scores.append(None)

        normalized_rows.append(
            RawRunRow(
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
                retrieved_chunk_ids=_as_ranked_id_list(row.get("retrieved_chunk_ids", [])),
                scores=scores,
                source_path=str(raw_results_file),
            )
        )

    return normalized_rows


def _dedupe_ranked_ids(
    retrieved_ids: Sequence[str],
    scores: Sequence[Optional[float]],
) -> Tuple[List[str], List[Optional[float]]]:
    seen = set()
    deduped_ids: List[str] = []
    deduped_scores: List[Optional[float]] = []
    for index, chunk_id in enumerate(retrieved_ids):
        normalized = str(chunk_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_ids.append(normalized)
        deduped_scores.append(scores[index] if index < len(scores) else None)
    return deduped_ids, deduped_scores


def choose_primary_relevance(
    labels_by_query: Dict[str, LabelRow],
    primary_relevance: str,
) -> Tuple[Optional[str], List[str], Dict[str, int]]:
    counts = {
        "gold_chunk_ids": 0,
        "silver_chunk_ids": 0,
        "relevant_ids": 0,
        "silver_chunk_groups": 0,
    }
    for row in labels_by_query.values():
        if row.gold_chunk_ids:
            counts["gold_chunk_ids"] += 1
        if row.silver_chunk_ids:
            counts["silver_chunk_ids"] += 1
        if row.relevant_ids:
            counts["relevant_ids"] += 1
        if row.silver_chunk_groups:
            counts["silver_chunk_groups"] += 1

    notes: List[str] = []
    if primary_relevance != "auto":
        selected = primary_relevance
    elif counts["gold_chunk_ids"] > 0:
        selected = "gold_chunk_ids"
    elif counts["silver_chunk_ids"] > 0:
        selected = "silver_chunk_ids"
    elif counts["relevant_ids"] > 0:
        selected = "relevant_ids"
    else:
        selected = None

    if selected is None and counts["silver_chunk_groups"] > 0:
        notes.append(
            "Labels contained silver_chunk_groups but no flat gold/silver/relevant ids, so binary ranking metrics could not be computed faithfully."
        )
    elif selected == "gold_chunk_ids" and counts["silver_chunk_ids"] > 0:
        notes.append(
            "Primary relevance uses gold_chunk_ids. Queries without gold labels are left null rather than back-filled from silver labels."
        )
    elif selected == "silver_chunk_ids":
        notes.append(
            "Primary relevance uses silver_chunk_ids because gold_chunk_ids were unavailable."
        )
    elif selected == "relevant_ids":
        notes.append(
            "Primary relevance uses relevant_ids because no standardized gold_chunk_ids or silver_chunk_ids were available."
        )

    return selected, notes, counts


def _relevance_from_label(
    label: Optional[LabelRow],
    primary_relevance: Optional[str],
) -> Tuple[List[str], "OrderedDict[str, float]", Optional[str]]:
    if label is None:
        return [], OrderedDict(), "No matching label row for query_id."
    if primary_relevance is None:
        return [], OrderedDict(), "No supported primary relevance field was available."

    if primary_relevance == "gold_chunk_ids":
        relevant_ids = list(label.gold_chunk_ids)
        grades = OrderedDict((chunk_id, 1.0) for chunk_id in relevant_ids)
        if not relevant_ids:
            return [], grades, "gold_chunk_ids missing for this query."
        return relevant_ids, grades, None

    if primary_relevance == "silver_chunk_ids":
        relevant_ids = list(label.silver_chunk_ids)
        grades = OrderedDict((chunk_id, 1.0) for chunk_id in relevant_ids)
        if not relevant_ids:
            return [], grades, "silver_chunk_ids missing for this query."
        return relevant_ids, grades, None

    relevant_ids = list(label.relevant_ids)
    grades = label.graded_relevance or OrderedDict((chunk_id, 1.0) for chunk_id in relevant_ids)
    if not relevant_ids and grades:
        relevant_ids = list(grades.keys())
    if not relevant_ids:
        return [], grades, "relevant_ids missing for this query."
    if not grades:
        grades = OrderedDict((chunk_id, 1.0) for chunk_id in relevant_ids)
    return relevant_ids, grades, None


def _dcg_at_k(
    ranked_ids: Sequence[str],
    gains_by_id: "OrderedDict[str, float]",
    k: int,
) -> float:
    dcg = 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        gain = float(gains_by_id.get(item_id, 0.0))
        if gain <= 0:
            continue
        dcg += gain / math.log2(rank + 1.0)
    return dcg


def compute_query_metrics(
    ranked_ids: Sequence[str],
    relevant_ids: Sequence[str],
    graded_relevance: "OrderedDict[str, float]",
    k_values: Sequence[int],
) -> Dict[str, Optional[float]]:
    relevant_set = set(relevant_ids)
    if not relevant_set:
        metrics: Dict[str, Optional[float]] = {}
        for k in k_values:
            metrics["recall@%d" % k] = None
            metrics["mrr@%d" % k] = None
            metrics["ndcg@%d" % k] = None
            metrics["hit_rate@%d" % k] = None
        return metrics

    gains_by_id: "OrderedDict[str, float]" = OrderedDict()
    if graded_relevance:
        for item_id, gain in graded_relevance.items():
            if gain > 0:
                gains_by_id[item_id] = float(gain)
    else:
        for item_id in relevant_ids:
            gains_by_id[item_id] = 1.0

    binary_relevant_ids = [item_id for item_id in gains_by_id if gains_by_id[item_id] > 0]
    if not binary_relevant_ids:
        binary_relevant_ids = list(relevant_ids)

    metrics = {}
    for k in k_values:
        top_k = list(ranked_ids[:k])
        hit_positions = [
            index
            for index, item_id in enumerate(top_k, start=1)
            if item_id in relevant_set
        ]
        hits = len(hit_positions)
        metrics["recall@%d" % k] = hits / float(len(binary_relevant_ids))
        metrics["mrr@%d" % k] = (1.0 / float(hit_positions[0])) if hit_positions else 0.0
        metrics["hit_rate@%d" % k] = 1.0 if hit_positions else 0.0

        ideal_gains = sorted(
            (float(gains_by_id[item_id]) for item_id in binary_relevant_ids),
            reverse=True,
        )
        dcg = _dcg_at_k(top_k, gains_by_id, k)
        idcg = 0.0
        for rank, gain in enumerate(ideal_gains[:k], start=1):
            idcg += gain / math.log2(rank + 1.0)
        metrics["ndcg@%d" % k] = (dcg / idcg) if idcg > 0 else None

    return metrics


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / float(len(filtered))


def _package_versions(names: Sequence[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in names:
        try:
            versions[name] = str(pkg_version(name))
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def evaluate_run(
    *,
    run_dir: Path,
    labels_file: Path,
    output_dir: Path,
    method_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    ks: Sequence[int] = DEFAULT_K_VALUES,
    run_name: Optional[str] = None,
    primary_relevance: str = "auto",
    raw_results_file: Optional[Path] = None,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    if primary_relevance not in LABEL_SOURCE_CHOICES:
        raise ValueError(
            "Invalid primary relevance choice. Expected one of: %s"
            % ", ".join(LABEL_SOURCE_CHOICES)
        )

    normalized_k_values = sorted({int(value) for value in ks if int(value) > 0})
    if not normalized_k_values:
        raise ValueError("At least one positive k value is required.")

    selected_raw_file, run_manifest, selection_notes = select_raw_results_file(
        run_dir,
        explicit_path=raw_results_file,
    )
    raw_rows = load_raw_run_rows(selected_raw_file)
    labels_by_query = load_label_rows(labels_file)
    chosen_relevance, relevance_notes, relevance_counts = choose_primary_relevance(
        labels_by_query,
        primary_relevance=primary_relevance,
    )

    resolved_run_name = (
        run_name
        or (
            str(run_manifest.get("run_name")).strip()
            if isinstance(run_manifest, dict) and run_manifest.get("run_name")
            else ""
        )
        or run_dir.name
    )
    resolved_dataset_name = (
        dataset_name
        or (
            str(run_manifest.get("dataset_name")).strip()
            if isinstance(run_manifest, dict) and run_manifest.get("dataset_name")
            else ""
        )
        or "unknown"
    )
    resolved_method_name = method_name or "late_chunking"
    resolved_split = split or "unknown"

    missing_metric_reasons: Dict[str, str] = OrderedDict()
    assumptions: List[str] = [
        "Metrics are computed over deduplicated retrieved ids while preserving the first occurrence from the raw ranking.",
        "Scores are retained only for provenance; Recall, MRR, NDCG, and HitRate are computed from ranked order.",
        "Binary relevance is used unless graded relevance scores are explicitly present in the labels file.",
    ]
    assumptions.extend(selection_notes)
    assumptions.extend(relevance_notes)

    per_query_rows: List[Dict[str, Any]] = []
    aggregate_inputs: Dict[str, List[Optional[float]]] = OrderedDict()
    for k in normalized_k_values:
        aggregate_inputs["recall@%d" % k] = []
        aggregate_inputs["mrr@%d" % k] = []
        aggregate_inputs["ndcg@%d" % k] = []
        aggregate_inputs["hit_rate@%d" % k] = []

    unmatched_label_queries = 0
    queries_without_relevance = 0
    for raw_row in raw_rows:
        deduped_ids, _deduped_scores = _dedupe_ranked_ids(
            raw_row.retrieved_chunk_ids,
            raw_row.scores,
        )
        label = labels_by_query.get(raw_row.query_id)
        if label is None:
            unmatched_label_queries += 1
        relevant_ids, grades, reason = _relevance_from_label(label, chosen_relevance)
        if reason is not None:
            queries_without_relevance += 1
            missing_metric_reasons.setdefault(
                raw_row.query_id,
                reason,
            )

        metric_values = compute_query_metrics(
            ranked_ids=deduped_ids,
            relevant_ids=relevant_ids,
            graded_relevance=grades,
            k_values=normalized_k_values,
        )
        for metric_name, value in metric_values.items():
            aggregate_inputs[metric_name].append(value)

        per_query_row: Dict[str, Any] = OrderedDict()
        per_query_row["query_id"] = raw_row.query_id
        per_query_row["doc_id"] = raw_row.doc_id or (label.doc_id if label else None)
        per_query_row["question"] = raw_row.question or (label.question if label else None)
        for k in normalized_k_values:
            per_query_row["recall@%d" % k] = metric_values["recall@%d" % k]
        for k in normalized_k_values:
            per_query_row["mrr@%d" % k] = metric_values["mrr@%d" % k]
        for k in normalized_k_values:
            per_query_row["ndcg@%d" % k] = metric_values["ndcg@%d" % k]
        for k in normalized_k_values:
            per_query_row["hit_rate@%d" % k] = metric_values["hit_rate@%d" % k]
        per_query_row["retrieved_ids_top10"] = deduped_ids[:10]
        per_query_row["relevant_ids"] = list(relevant_ids)
        per_query_rows.append(per_query_row)

    retrieval_metrics: Dict[str, Optional[float]] = OrderedDict()
    for k in normalized_k_values:
        retrieval_metrics["recall@%d" % k] = _mean(aggregate_inputs["recall@%d" % k])
    for k in normalized_k_values:
        retrieval_metrics["mrr@%d" % k] = _mean(aggregate_inputs["mrr@%d" % k])
    for k in normalized_k_values:
        retrieval_metrics["ndcg@%d" % k] = _mean(aggregate_inputs["ndcg@%d" % k])
    for k in normalized_k_values:
        retrieval_metrics["hit_rate@%d" % k] = _mean(aggregate_inputs["hit_rate@%d" % k])

    for metric_name, value in retrieval_metrics.items():
        if value is None:
            missing_metric_reasons.setdefault(
                metric_name,
                "Metric could not be computed because no query had usable relevance labels for the selected primary relevance source.",
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_summary_path = output_dir / "metrics_summary.json"
    per_query_path = output_dir / "metrics_per_query.jsonl"
    leaderboard_path = output_dir / "leaderboard_row.json"
    manifest_path = output_dir / "evaluation_manifest.json"

    metrics_summary = OrderedDict(
        [
            ("method_name", resolved_method_name),
            ("dataset_name", resolved_dataset_name),
            ("split", resolved_split),
            ("run_name", resolved_run_name),
            ("n_queries", len(raw_rows)),
            ("k_values", list(normalized_k_values)),
            ("primary_relevance", chosen_relevance),
            ("retrieval_metrics", retrieval_metrics),
        ]
    )

    leaderboard_row = OrderedDict(
        [
            ("method_name", resolved_method_name),
            ("dataset_name", resolved_dataset_name),
            ("split", resolved_split),
            ("run_name", resolved_run_name),
        ]
    )
    for metric_name, value in retrieval_metrics.items():
        leaderboard_row[metric_name] = value

    manifest = OrderedDict(
        [
            (
                "input_files_used",
                OrderedDict(
                    [
                        ("run_dir", str(run_dir.resolve())),
                        ("raw_results_file", str(selected_raw_file.resolve())),
                        ("labels_file", str(labels_file.resolve())),
                        (
                            "run_manifest",
                            str((run_dir / "run_manifest.json").resolve())
                            if (run_dir / "run_manifest.json").exists()
                            else None,
                        ),
                    ]
                ),
            ),
            (
                "output_files_written",
                OrderedDict(
                    [
                        ("metrics_summary_json", str(metrics_summary_path.resolve())),
                        ("metrics_per_query_jsonl", str(per_query_path.resolve())),
                        ("leaderboard_row_json", str(leaderboard_path.resolve())),
                        ("evaluation_manifest_json", str(manifest_path.resolve())),
                    ]
                ),
            ),
            (
                "relevance_source_used",
                OrderedDict(
                    [
                        ("primary_relevance", chosen_relevance),
                        ("available_label_counts", relevance_counts),
                    ]
                ),
            ),
            ("primary_relevance_choice", chosen_relevance),
            ("assumptions", assumptions),
            ("missing_metrics_and_reasons", missing_metric_reasons),
            ("command_used", command or " ".join(shlex.quote(part) for part in sys.argv)),
            (
                "package_versions",
                _package_versions(
                    [
                        "late_chunking",
                        "click",
                        "numpy",
                        "datasets",
                        "torch",
                        "transformers",
                    ]
                ),
            ),
            (
                "join_summary",
                OrderedDict(
                    [
                        ("n_raw_queries", len(raw_rows)),
                        ("n_label_rows", len(labels_by_query)),
                        ("n_queries_missing_label_row", unmatched_label_queries),
                        ("n_queries_without_primary_relevance", queries_without_relevance),
                    ]
                ),
            ),
            (
                "raw_result_selection_notes",
                selection_notes,
            ),
            (
                "run_context",
                OrderedDict(
                    [
                        ("method_name", resolved_method_name),
                        ("dataset_name", resolved_dataset_name),
                        ("split", resolved_split),
                        ("run_name", resolved_run_name),
                    ]
                ),
            ),
        ]
    )

    _write_json(metrics_summary_path, metrics_summary)
    _write_jsonl(per_query_path, per_query_rows)
    _write_json(leaderboard_path, leaderboard_row)
    _write_json(manifest_path, manifest)

    return {
        "metrics_summary": metrics_summary,
        "leaderboard_row": leaderboard_row,
        "manifest": manifest,
        "output_paths": {
            "metrics_summary_json": _relative_or_absolute(metrics_summary_path, output_dir),
            "metrics_per_query_jsonl": _relative_or_absolute(per_query_path, output_dir),
            "leaderboard_row_json": _relative_or_absolute(leaderboard_path, output_dir),
            "evaluation_manifest_json": _relative_or_absolute(manifest_path, output_dir),
        },
    }
