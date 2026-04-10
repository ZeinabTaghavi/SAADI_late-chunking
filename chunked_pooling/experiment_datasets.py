from __future__ import annotations

import ast
import logging
import json
import os
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
LOOGLE_DATASET_NAMES = {"loogle", "bigai-nlco/loogle", "bigainlco/loogle"}
NARRATIVEQA_DATASET_NAMES = {"narrativeqa", "deepmind/narrativeqa"}
QUALITY_DATASET_NAMES = {"quality", "tasksource/quality", "tasksource/quality"}
NOVELHOPQA_DATASET_NAMES = {
    "novelqa",
    "novelhopqa",
    "abhaygupta1266/novelhopqa",
}
LOOGLE_HF_DATASET_IDS = ("bigai-nlco/LooGLE", "bigainlco/LooGLE")
LOOGLE_LEGACY_CONFIG_ALIASES = {
    "longdep_summarization": "summarization",
}
NOVELHOPQA_DATASET_ID = "abhaygupta1266/novelhopqa"
NOVELHOPQA_VALID_SPLITS = ("hop_1", "hop_2", "hop_3", "hop_4")
NOVELHOPQA_CONFIG_ALIASES = {
    "default": "all",
    "full": "all",
    "all_hops": "all",
}
QUALITY_DATASET_ID = "tasksource/QuALITY"
QUALITY_VALID_SPLITS = {"train", "validation"}
QUALITY_SPLIT_ALIASES = {
    "default": "validation",
    "dev": "validation",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "validation",
    "train": "train",
}


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


def _resolve_narrativeqa_config(
    requested: Optional[str], dataset_name: str = "deepmind/narrativeqa"
) -> Optional[str]:
    try:
        configs = datasets.get_dataset_config_names(dataset_name) or []
    except Exception:
        configs = []
    if not configs:
        return None
    if requested in configs:
        return requested
    return configs[0]


def _load_narrativeqa_dataset(*, cfg: Optional[str]):
    try:
        return (
            datasets.load_dataset("deepmind/narrativeqa", name=cfg)
            if cfg
            else datasets.load_dataset("deepmind/narrativeqa")
        )
    except ValueError as exc:
        msg = str(exc)
        if "Feature type 'List' not found" in msg:
            raise RuntimeError(
                "Loading 'deepmind/narrativeqa' failed because your installed "
                "`datasets` package is too old for this dataset metadata. Upgrade "
                "datasets (for example `python -m pip install -U \"datasets>=4\"`) "
                "and retry."
            ) from exc
        raise


def load_narrativeqa_bundle(loader_config: Dict[str, object]) -> DatasetBundle:
    import random

    split = str(loader_config.get("split") or "test")
    config_name = loader_config.get("config_name") or "default"
    qa_n = loader_config.get("qa_n", "all")
    qa_selection_method = str(loader_config.get("qa_selection_method") or "first")

    cfg = _resolve_narrativeqa_config(config_name)
    logger.info(
        "Loading NarrativeQA split=%s config=%s n=%s selection=%s",
        split,
        cfg,
        qa_n,
        qa_selection_method,
    )
    dataset_dict = _load_narrativeqa_dataset(cfg=cfg)

    documents: OrderedDict[str, Dict[str, object]] = OrderedDict()
    qa_candidates: List[Dict[str, object]] = []

    for row_index, row in enumerate(dataset_dict[split]):
        doc = row.get("document") or {}
        doc_id = doc.get("id") or row.get("document_id") or row.get("doc_id")
        if not doc_id:
            continue
        doc_id = str(doc_id)
        doc_text = _coerce_to_text(doc.get("text") if isinstance(doc, dict) else doc).strip()
        if doc_text and doc_id not in documents:
            documents[doc_id] = {
                **row,
                "doc_id": doc_id,
                "text": doc_text,
            }

        question = row.get("question")
        if isinstance(question, dict):
            question = question.get("text")
        question_text = _coerce_to_text(question).strip()
        if not question_text:
            continue

        answer_text = _coerce_to_text(row.get("answers")).strip()
        if not answer_text:
            continue

        qa_candidates.append(
            {
                "doc_id": doc_id,
                "question": question_text,
                "answers": [answer_text],
                "retrieval_spans": [],
                "source_row_index": row_index,
                "raw_row": dict(row),
            }
        )

    total = len(qa_candidates)
    if isinstance(qa_n, str):
        qa_n = total if qa_n.lower() == "all" else int(qa_n)
    qa_n = min(int(qa_n), total)
    indices = list(range(total))
    if qa_n < total:
        indices = (
            random.sample(indices, qa_n)
            if qa_selection_method == "random"
            else indices[:qa_n]
        )

    qa_entries: List[Dict[str, object]] = []
    for qa_index in indices:
        row = qa_candidates[qa_index]
        qa_entries.append(
            {
                "query_id": f"narrativeqa_{qa_index}",
                "source_qa_index": qa_index,
                "doc_id": row["doc_id"],
                "document_id": row["doc_id"],
                "question": row["question"],
                "reference_answers": list(row["answers"]),
                "answers": list(row["answers"]),
                "retrieval_spans": [],
                "evidence_spans": [],
                "source_row_index": row["source_row_index"],
                "raw_row": row["raw_row"],
                "relevant_doc_ids": [row["doc_id"]],
            }
        )

    return DatasetBundle(
        documents=documents,
        qa_entries=qa_entries,
        metadata={
            "loader_type": "narrativeqa",
            "split": split,
            "config_name": cfg,
            "qa_n": loader_config.get("qa_n", "all"),
            "qa_selection_method": qa_selection_method,
        },
    )


def _normalize_quality_split(split: Optional[str]) -> str:
    raw = str(split or "validation").strip().lower()
    normalized = QUALITY_SPLIT_ALIASES.get(raw, raw)
    if normalized not in QUALITY_VALID_SPLITS:
        raise ValueError(
            "Unsupported QuALITY split. Use one of: train, validation "
            f"(got: {split!r})."
        )
    return normalized


def _load_quality_split(split_name: str):
    try:
        return datasets.load_dataset(QUALITY_DATASET_ID, split=split_name)
    except TypeError:
        return datasets.load_dataset(QUALITY_DATASET_ID)[split_name]


def _quality_row_doc_id(row: Dict[str, Any], *, fallback_index: int) -> str:
    for key in ("article_id", "document_id", "doc_id"):
        value = row.get(key)
        if isinstance(value, (int, str)) and str(value).strip():
            return f"article:{str(value).strip()}"
    title = row.get("title")
    if isinstance(title, str) and title.strip():
        safe_title = "_".join(title.strip().split())
        return f"article:{safe_title}"
    return f"article:{fallback_index}"


def _quality_options_list(row: Dict[str, Any]) -> List[str]:
    raw = row.get("options")
    if isinstance(raw, list):
        return [text for text in (_coerce_to_text(item).strip() for item in raw) if text]
    return []


def _quality_gold_option_index(row: Dict[str, Any], options: Sequence[str]) -> Optional[int]:
    for key in ("gold_label", "writer_label"):
        value = row.get(key)
        try:
            idx = int(value)
        except Exception:
            continue
        if 1 <= idx <= len(options):
            return idx - 1
        if 0 <= idx < len(options):
            return idx
    return None


def _quality_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in ("title", "source", "author", "topic", "year"):
        value = _coerce_to_text(row.get(key)).strip()
        if value:
            metadata[key] = value
    difficult = row.get("difficult")
    try:
        if difficult is not None:
            metadata["difficult"] = int(difficult)
    except Exception:
        pass
    return metadata


def load_quality_bundle(loader_config: Dict[str, object]) -> DatasetBundle:
    import random

    split_name = _normalize_quality_split(str(loader_config.get("split") or "validation"))
    config_name = loader_config.get("config_name") or "default"
    qa_n = loader_config.get("qa_n", "all")
    qa_selection_method = str(loader_config.get("qa_selection_method") or "first")

    _ = config_name
    logger.info(
        "Loading QuALITY split=%s n=%s selection=%s",
        split_name,
        qa_n,
        qa_selection_method,
    )
    rows = _load_quality_split(split_name)

    documents: OrderedDict[str, Dict[str, object]] = OrderedDict()
    qa_candidates: List[Dict[str, object]] = []

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        doc_id = _quality_row_doc_id(row, fallback_index=row_index)
        article_text = _coerce_to_text(row.get("article")).strip()
        if article_text and doc_id not in documents:
            documents[doc_id] = {
                **row,
                "doc_id": doc_id,
                "text": article_text,
            }

        if doc_id not in documents:
            continue

        question = _coerce_to_text(row.get("question")).strip()
        if not question:
            continue

        options = _quality_options_list(row)
        gold_idx = _quality_gold_option_index(row, options)
        if gold_idx is None or gold_idx >= len(options):
            continue

        answer_text = options[gold_idx]
        if not answer_text:
            continue

        candidate: Dict[str, object] = {
            "query_id": str(row.get("question_unique_id") or f"{doc_id}:q{row_index}"),
            "doc_id": doc_id,
            "question": question,
            "answers": [answer_text],
            "retrieval_spans": [],
            "choices": list(options),
            "gold_option_index": gold_idx,
            "gold_option_label": gold_idx + 1,
            "source_row_index": row_index,
            "raw_row": dict(row),
        }
        metadata = _quality_metadata(row)
        if metadata:
            candidate["metadata"] = metadata
        qa_candidates.append(candidate)

    total = len(qa_candidates)
    if isinstance(qa_n, str):
        qa_n = total if qa_n.lower() == "all" else int(qa_n)
    qa_n = min(int(qa_n), total)
    indices = list(range(total))
    if qa_n < total:
        indices = (
            random.sample(indices, qa_n)
            if qa_selection_method == "random"
            else indices[:qa_n]
        )

    qa_entries: List[Dict[str, object]] = []
    for qa_index in indices:
        row = qa_candidates[qa_index]
        qa_entries.append(
            {
                "query_id": row["query_id"],
                "source_qa_index": qa_index,
                "doc_id": row["doc_id"],
                "document_id": row["doc_id"],
                "question": row["question"],
                "reference_answers": list(row["answers"]),
                "answers": list(row["answers"]),
                "retrieval_spans": [],
                "evidence_spans": [],
                "choices": list(row["choices"]),
                "gold_option_index": row["gold_option_index"],
                "gold_option_label": row["gold_option_label"],
                "source_row_index": row["source_row_index"],
                "raw_row": row["raw_row"],
                "relevant_doc_ids": [row["doc_id"]],
                **({"metadata": row["metadata"]} if "metadata" in row else {}),
            }
        )

    return DatasetBundle(
        documents=documents,
        qa_entries=qa_entries,
        metadata={
            "loader_type": "quality",
            "split": split_name,
            "config_name": config_name,
            "qa_n": loader_config.get("qa_n", "all"),
            "qa_selection_method": qa_selection_method,
        },
    )


def _normalize_novelhopqa_config(config_name: Optional[str]) -> str:
    raw = str(config_name or "all").strip().lower()
    normalized = NOVELHOPQA_CONFIG_ALIASES.get(raw, raw)
    if normalized == "all" or normalized in NOVELHOPQA_VALID_SPLITS:
        return normalized
    raise ValueError(
        "Unsupported NovelHopQA config_name. Use one of: all/default, "
        "hop_1, hop_2, hop_3, hop_4."
    )


def _novelhopqa_selected_splits(mode: str) -> List[str]:
    if mode == "all":
        return list(NOVELHOPQA_VALID_SPLITS)
    return [mode]


def _load_novelhopqa_split(split_name: str):
    major = _datasets_version_major()
    kwargs: Dict[str, Any] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        return datasets.load_dataset(
            NOVELHOPQA_DATASET_ID, "default", split=split_name, **kwargs
        )
    except TypeError:
        try:
            return datasets.load_dataset(
                NOVELHOPQA_DATASET_ID, split=split_name, **kwargs
            )
        except TypeError:
            return datasets.load_dataset(NOVELHOPQA_DATASET_ID, split=split_name)
    except Exception:
        return datasets.load_dataset(NOVELHOPQA_DATASET_ID, "default", split=split_name)


def _safe_component(value: str, *, default: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return out or default


def _normalize_book_key(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _novelhopqa_query_id(row: Dict[str, Any], *, split_name: str, index: int) -> str:
    base = row.get("qid") or row.get("question_id") or row.get("id") or index
    return f"{split_name}:{_safe_component(str(base), default=str(index))}"


def _find_top_file_ci(root: Path, name: str) -> Optional[Path]:
    try:
        for child in root.iterdir():
            if child.is_file() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _find_child_dir_ci(root: Path, name: str) -> Optional[Path]:
    try:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _looks_like_books_root(root: Path) -> bool:
    if not root.exists():
        return False
    if root.is_file():
        return root.suffix.lower() == ".txt"
    if _find_top_file_ci(root, "bookmeta.json") is not None:
        return True
    books_dir = _find_child_dir_ci(root, "Books")
    if books_dir is not None:
        return True
    try:
        return any(
            child.is_file() and child.suffix.lower() == ".txt" for child in root.iterdir()
        )
    except Exception:
        return False


def _coerce_books_root(raw_root: Path) -> Path:
    root = raw_root.expanduser().resolve()
    if _looks_like_books_root(root):
        return root
    try:
        for child in root.iterdir():
            if child.is_dir() and _looks_like_books_root(child):
                return child.resolve()
    except Exception:
        pass
    return root


def _resolve_novelhopqa_books_root(raw_root: Optional[str]) -> Path:
    configured_raw = str(raw_root).strip() if raw_root is not None else ""
    env_books_root = str(os.environ.get("NOVELHOPQA_BOOKS_ROOT") or "").strip()
    env_novelqa_root = str(os.environ.get("NOVELQA_DATASET_DIR") or "").strip()
    default_relative = (
        Path(__file__).resolve().parents[1]
        / "../passing_meta_tag/novelhopqa/book-corpus-root"
    ).resolve()
    candidates = [
        ("NOVELHOPQA_BOOKS_ROOT", env_books_root),
        ("NOVELQA_DATASET_DIR", env_novelqa_root),
        ("dataset.books_root", configured_raw),
        ("default_relative", str(default_relative)),
    ]
    unusable_attempts: List[str] = []
    saw_candidate = False
    for label, configured in candidates:
        if not configured:
            continue
        saw_candidate = True
        root = _coerce_books_root(Path(configured))
        if _looks_like_books_root(root):
            if unusable_attempts:
                logger.warning(
                    "Ignoring unusable NovelHopQA corpus root candidate(s): %s. Using %s=%s",
                    ", ".join(unusable_attempts),
                    label,
                    root,
                )
            return root
        unusable_attempts.append(f"{label}={root}")

    if not saw_candidate:
        raise RuntimeError(
            "NovelHopQA whole-book loading requires a book corpus root. Set "
            "dataset.books_root or NOVELHOPQA_BOOKS_ROOT to a directory containing "
            "bookmeta.json + book text files (or a title-mapped text corpus)."
        )
    raise RuntimeError(
        "NovelHopQA whole-book loading could not find a usable corpus root at "
        f"{'; '.join(unusable_attempts)}. Expected bookmeta.json, a Books/ directory, or .txt files."
    )


def _iter_bookmeta_entries(payload: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                out.append(item)
        return out
    if isinstance(payload, dict):
        books = payload.get("books")
        if isinstance(books, list):
            for item in books:
                if isinstance(item, dict):
                    out.append(item)
            return out
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("BID", key)
            out.append(row)
    return out


def _candidate_text_paths(root: Path, file_name: str) -> List[Path]:
    p = Path(str(file_name))
    if p.is_absolute():
        return [p]
    if len(p.parts) > 1:
        return [(root / p).resolve()]
    return [
        (root / "Books" / "PublicDomain" / p).resolve(),
        (root / "Books" / "publicdomain" / p).resolve(),
        (root / "Books" / "CopyrightProtected" / p).resolve(),
        (root / "Books" / "copyrightprotected" / p).resolve(),
        (root / "Books" / p).resolve(),
        (root / p).resolve(),
        (root / "Demonstration" / p).resolve(),
    ]


def _book_doc_id(title: str, *, fallback: str) -> str:
    return f"book:{_safe_component(title, default=fallback)}"


def _book_title(row: Dict[str, Any]) -> Optional[str]:
    text = _coerce_to_text(row.get("book") or row.get("title") or row.get("book_title")).strip()
    return text or None


def _novelhopqa_env_flag(name: str) -> bool:
    raw = str(os.environ.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _iter_title_variants(value: Optional[str]) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    variants: List[str] = [raw]
    cleanup_suffixes = (
        " complete",
        " unabridged",
        " illustrated",
        " with illustrations",
    )
    for sep in (":", ";", ",", " - ", " — ", " – "):
        if sep in raw:
            head = raw.split(sep, 1)[0].strip()
            if head:
                variants.append(head)
    for candidate in list(variants):
        lowered = candidate.casefold()
        for prefix in ("the ", "a ", "an "):
            if lowered.startswith(prefix):
                trimmed = candidate[len(prefix) :].strip()
                if trimmed:
                    variants.append(trimmed)
        for suffix in cleanup_suffixes:
            if lowered.endswith(suffix):
                trimmed = candidate[: -len(suffix)].strip(" ,;-:")
                if trimmed:
                    variants.append(trimmed)
    out: List[str] = []
    seen: set[str] = set()
    for candidate in variants:
        key = _normalize_book_key(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _title_like_lines(text: str, *, limit: int = 5) -> List[str]:
    raw_candidates: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.replace("\ufeff", "").replace("\xa0", " ").strip()
        if not line:
            continue
        lowered = line.casefold()
        if "project gutenberg ebook of" in lowered:
            match = re.search(
                r"project gutenberg ebook of\s+(.+?)(?:,\s+by\b|$)",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                title = match.group(1).strip(" .,:;!-")
                if title:
                    raw_candidates.append(title)
                    if len(raw_candidates) >= limit:
                        break
            continue
        if (
            "project gutenberg" in lowered
            or "www.gutenberg.org" in lowered
            or lowered.startswith("***")
            or lowered.startswith("by ")
            or lowered.startswith("translated by ")
            or lowered.startswith("produced by ")
            or lowered.startswith("release date")
            or lowered.startswith("language:")
            or lowered.startswith("contents")
            or lowered.startswith("chapter ")
            or lowered.startswith("book ")
        ):
            continue
        if len(line) > 160:
            continue
        if sum(ch.isalpha() for ch in line) < 3:
            continue
        raw_candidates.append(line)
        if len(raw_candidates) >= limit:
            break

    out: List[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        normalized = candidate.strip(" .,:;!-")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)

    stitched_parts: List[str] = []
    total_words = 0
    for candidate in raw_candidates[:3]:
        lowered = candidate.casefold()
        if stitched_parts and (
            lowered.startswith("by ")
            or lowered.startswith("translated by ")
            or lowered.startswith("produced by ")
        ):
            break
        if len(candidate) > 80:
            break
        stitched_parts.append(candidate.strip(" .,:;!-"))
        total_words += len(candidate.split())
        if len(stitched_parts) < 2 or total_words > 12:
            continue
        stitched = " ".join(part for part in stitched_parts if part).strip(" .,:;!-")
        key = stitched.casefold()
        if stitched and key not in seen:
            seen.add(key)
            out.append(stitched)
    return out


def _register_book_aliases(
    books: Dict[str, tuple[str, str]],
    *,
    doc_id: str,
    text: str,
    candidates: Sequence[str],
) -> None:
    for raw in candidates:
        for variant in _iter_title_variants(raw):
            key = _normalize_book_key(variant)
            if key:
                books.setdefault(key, (doc_id, text))


def _novelhopqa_report_dir() -> Path:
    configured = str(os.environ.get("NOVELHOPQA_REPORT_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "novelhopqa"


def _write_title_report(path: Path, titles: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        {str(title).strip() for title in titles if str(title).strip()}, key=str.casefold
    )
    content = "\n".join(ordered)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _load_books_from_root(root: Path) -> tuple[Dict[str, tuple[str, str]], Dict[str, str]]:
    books: Dict[str, tuple[str, str]] = {}
    titles_by_doc_id: Dict[str, str] = {}
    bookmeta_path = _find_top_file_ci(root, "bookmeta.json")
    if bookmeta_path is not None and bookmeta_path.exists():
        payload = _read_json(bookmeta_path)
        for index, row in enumerate(_iter_bookmeta_entries(payload)):
            title = _coerce_to_text(row.get("title") or row.get("book") or row.get("name")).strip()
            doc_id = str(
                row.get("BID")
                or row.get("bid")
                or _book_doc_id(title, fallback=f"book_{index}")
            ).strip()
            txt_name = (
                row.get("txtfile")
                or row.get("txt_file")
                or row.get("book_file")
                or row.get("text_file")
            )
            candidate_names: List[str] = []
            if isinstance(txt_name, str) and txt_name.strip():
                candidate_names.append(txt_name.strip())
            if doc_id:
                candidate_names.extend(
                    [f"{doc_id}.txt", f"{doc_id.upper()}.txt", f"{doc_id.lower()}.txt"]
                )
            if not title or not candidate_names:
                continue
            seen_candidates: set[Path] = set()
            for raw_name in candidate_names:
                for candidate in _candidate_text_paths(root, raw_name):
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    if not candidate.exists():
                        continue
                    text = _load_text(candidate)
                    if not text:
                        continue
                    aliases = [
                        title,
                        doc_id,
                        Path(raw_name).stem,
                        candidate.stem,
                        *_title_like_lines(text),
                    ]
                    canonical_title = str(title or candidate.stem or doc_id).strip()
                    if canonical_title:
                        titles_by_doc_id.setdefault(doc_id, canonical_title)
                    _register_book_aliases(
                        books, doc_id=doc_id, text=text, candidates=aliases
                    )
                    break
                else:
                    continue
                break
        if books:
            return books, titles_by_doc_id

    if root.is_file() and root.suffix.lower() == ".txt":
        title = root.stem
        text = _load_text(root)
        if not text:
            return {}, {}
        doc_id = _book_doc_id(title, fallback="book")
        titles_by_doc_id[doc_id] = title
        _register_book_aliases(
            books,
            doc_id=doc_id,
            text=text,
            candidates=[title, *_title_like_lines(text)],
        )
        return books, titles_by_doc_id

    try:
        txt_files = sorted(p for p in root.rglob("*.txt") if p.is_file())
    except Exception:
        txt_files = []
    for path in txt_files:
        text = _load_text(path)
        if not text:
            continue
        stem = path.stem
        doc_id = _book_doc_id(stem, fallback=stem)
        title_candidates = _title_like_lines(text)
        canonical_title = str(title_candidates[0] if title_candidates else stem).strip()
        if canonical_title:
            titles_by_doc_id.setdefault(doc_id, canonical_title)
        _register_book_aliases(
            books,
            doc_id=doc_id,
            text=text,
            candidates=[stem, *title_candidates],
        )
    return books, titles_by_doc_id


def _load_novelhopqa_all(
    *,
    mode: str,
    split: str,
    books_root: Optional[str] = None,
) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
    _ = split
    root = _resolve_novelhopqa_books_root(books_root)
    books_by_key, titles_by_doc_id = _load_books_from_root(root)
    if not books_by_key:
        raise RuntimeError(
            "NovelHopQA whole-book loading found zero books under "
            f"{root}. Expected title-mapped .txt files or bookmeta.json entries."
        )

    docs: Dict[str, str] = {}
    qa_entries: List[Dict[str, Any]] = []
    missing_books: set[str] = set()
    used_doc_ids: set[str] = set()
    for split_name in _novelhopqa_selected_splits(mode):
        rows = _load_novelhopqa_split(split_name)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            book_title = _book_title(row)
            if not book_title:
                continue
            book_record = books_by_key.get(_normalize_book_key(book_title))
            if book_record is None:
                missing_books.add(book_title)
                continue
            doc_id, document_text = book_record
            used_doc_ids.add(doc_id)
            if document_text and doc_id not in docs:
                docs[doc_id] = document_text

            context = _coerce_to_text(
                row.get("context") or row.get("passage") or row.get("document")
            ).strip()
            question = _coerce_to_text(row.get("question") or row.get("query")).strip()
            answer = _coerce_to_text(row.get("answer") or row.get("gold_answer")).strip()
            if not context:
                continue
            if not question or not answer:
                continue
            qa_entries.append(
                {
                    "query_id": _novelhopqa_query_id(
                        row, split_name=split_name, index=index
                    ),
                    "question": question,
                    "document_id": doc_id,
                    "book_title": book_title,
                    "gold_context_window": context,
                    "retrieval_span_mode": "window",
                    "answers": [answer],
                    "retrieval_spans": [context],
                    "source_split": split_name,
                    "raw_row": dict(row),
                }
            )

    report_dir = _novelhopqa_report_dir()
    _write_title_report(report_dir / "novelhop_query_remained.txt", missing_books)
    unused_books = {
        title for doc_id, title in titles_by_doc_id.items() if doc_id not in used_doc_ids
    }
    _write_title_report(report_dir / "novelhop_book_remained.txt", unused_books)

    if missing_books:
        sample = ", ".join(sorted(missing_books)[:5])
        if not _novelhopqa_env_flag("NOVELHOPQA_SUBSET_MODE"):
            raise RuntimeError(
                "NovelHopQA whole-book loading could not resolve book texts for "
                f"{len(missing_books)} title(s) under {root}. Example title(s): {sample}. "
                "Provide dataset.books_root or NOVELHOPQA_BOOKS_ROOT pointing to a corpus with matching titles."
            )
        logger.warning(
            "NovelHopQA subset mode enabled; skipped %d unresolved title(s) under %s. "
            "Example title(s): %s. Reports written to %s",
            len(missing_books),
            root,
            sample,
            report_dir,
        )

    return docs, qa_entries


def load_novelhopqa_bundle(loader_config: Dict[str, object]) -> DatasetBundle:
    import random

    split = str(loader_config.get("split") or "test")
    docs_config_name = loader_config.get("docs_config_name") or loader_config.get("config_name") or "all"
    qa_config_name = loader_config.get("qa_config_name") or loader_config.get("config_name") or "all"
    qa_n = loader_config.get("qa_n", "all")
    qa_selection_method = str(loader_config.get("qa_selection_method") or "first")
    books_root = loader_config.get("books_root")

    docs_mode = _normalize_novelhopqa_config(docs_config_name)
    qa_mode = _normalize_novelhopqa_config(qa_config_name)

    logger.info(
        "Loading NovelHopQA split=%s docs_config=%s qa_config=%s n=%s selection=%s",
        split,
        docs_mode,
        qa_mode,
        qa_n,
        qa_selection_method,
    )

    documents_raw, _ = _load_novelhopqa_all(
        mode=docs_mode,
        split=split,
        books_root=str(books_root) if books_root is not None else None,
    )
    _, qa_candidates_raw = _load_novelhopqa_all(
        mode=qa_mode,
        split=split,
        books_root=str(books_root) if books_root is not None else None,
    )

    documents: OrderedDict[str, Dict[str, object]] = OrderedDict(
        (
            doc_id,
            {
                "doc_id": doc_id,
                "text": text,
            },
        )
        for doc_id, text in documents_raw.items()
    )

    valid_doc_ids = set(documents.keys())
    qa_candidates = [
        row
        for row in qa_candidates_raw
        if str(row.get("document_id") or "") in valid_doc_ids
    ]

    total = len(qa_candidates)
    if isinstance(qa_n, str):
        qa_n = total if qa_n.lower() == "all" else int(qa_n)
    qa_n = min(int(qa_n), total)
    indices = list(range(total))
    if qa_n < total:
        indices = (
            random.sample(indices, qa_n)
            if qa_selection_method == "random"
            else indices[:qa_n]
        )

    qa_entries: List[Dict[str, object]] = []
    for qa_index in indices:
        row = qa_candidates[qa_index]
        doc_id = str(row["document_id"])
        qa_entries.append(
            {
                "query_id": str(row["query_id"]),
                "source_qa_index": qa_index,
                "doc_id": doc_id,
                "document_id": doc_id,
                "question": str(row["question"]),
                "reference_answers": list(row["answers"]),
                "answers": list(row["answers"]),
                "retrieval_spans": list(row["retrieval_spans"]),
                "evidence_spans": list(row["retrieval_spans"]),
                "book_title": row.get("book_title"),
                "gold_context_window": row.get("gold_context_window"),
                "retrieval_span_mode": row.get("retrieval_span_mode"),
                "source_split": row.get("source_split"),
                "raw_row": row.get("raw_row"),
                "relevant_doc_ids": [doc_id],
            }
        )

    return DatasetBundle(
        documents=documents,
        qa_entries=qa_entries,
        metadata={
            "loader_type": "novelhopqa",
            "split": split,
            "docs_config_name": docs_mode,
            "qa_config_name": qa_mode,
            "qa_n": loader_config.get("qa_n", "all"),
            "qa_selection_method": qa_selection_method,
            "books_root": str(books_root) if books_root is not None else None,
        },
    )


def _normalize_loogle_requested_config(requested: Optional[str]) -> str:
    raw = str(requested or "").strip()
    if not raw:
        return "shortdep_qa"
    return LOOGLE_LEGACY_CONFIG_ALIASES.get(raw, raw)


def _loogle_config_candidates(requested: Optional[str]) -> List[str]:
    normalized = _normalize_loogle_requested_config(requested)
    out = [normalized]
    for old_name, new_name in LOOGLE_LEGACY_CONFIG_ALIASES.items():
        if normalized == new_name:
            out.append(old_name)
        elif normalized == old_name:
            out.append(new_name)
    return out


def _get_loogle_config_names(dataset_name: str) -> List[str]:
    major = _datasets_version_major()
    kwargs: Dict[str, object] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            return list(datasets.get_dataset_config_names(dataset_name, **kwargs) or [])
        except TypeError:
            return list(datasets.get_dataset_config_names(dataset_name) or [])
    except Exception:
        return []


def _resolve_loogle_dataset_and_config(
    requested_config_name: Optional[str],
) -> tuple[str, Optional[str]]:
    desired_candidates = _loogle_config_candidates(requested_config_name)
    for dataset_name in LOOGLE_HF_DATASET_IDS:
        configs = _get_loogle_config_names(dataset_name)
        if not configs:
            continue
        for cfg in desired_candidates:
            if cfg in configs:
                return dataset_name, cfg
        if "shortdep_qa" in configs:
            return dataset_name, "shortdep_qa"
        return dataset_name, configs[0]
    fallback_config = desired_candidates[0] if desired_candidates else None
    return LOOGLE_HF_DATASET_IDS[0], fallback_config


def _load_loogle_dataset(*, cfg: Optional[str]):
    major = _datasets_version_major()
    preferred_name, preferred_cfg = _resolve_loogle_dataset_and_config(cfg)
    dataset_order = [preferred_name] + [
        name for name in LOOGLE_HF_DATASET_IDS if name != preferred_name
    ]
    cfg_candidates: List[Optional[str]] = [preferred_cfg]
    for candidate in _loogle_config_candidates(cfg):
        if candidate not in cfg_candidates:
            cfg_candidates.append(candidate)

    kwargs_candidates: List[Dict[str, object]] = []
    if major is not None and major >= 4:
        kwargs_candidates.append({"revision": "refs/convert/parquet"})
    kwargs_candidates.append({})
    if major is None or major < 4:
        kwargs_candidates.append({"trust_remote_code": True})

    def _attempt_load(*, force_offline: bool):
        prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        prev_datasets_offline = os.environ.get("HF_DATASETS_OFFLINE")
        offline_download_config = None
        if force_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            try:
                from datasets import DownloadConfig

                offline_download_config = DownloadConfig(local_files_only=True)
            except Exception:
                offline_download_config = None
        last_exc: Optional[Exception] = None
        try:
            for dataset_name in dataset_order:
                for cfg_name in cfg_candidates:
                    for kwargs in kwargs_candidates:
                        call_kwargs = dict(kwargs)
                        if (
                            force_offline
                            and offline_download_config is not None
                            and "download_config" not in call_kwargs
                        ):
                            call_kwargs["download_config"] = offline_download_config
                        try:
                            return (
                                datasets.load_dataset(
                                    dataset_name,
                                    name=cfg_name,
                                    **call_kwargs,
                                )
                                if cfg_name
                                else datasets.load_dataset(dataset_name, **call_kwargs)
                            )
                        except TypeError:
                            try:
                                return (
                                    datasets.load_dataset(dataset_name, name=cfg_name)
                                    if cfg_name
                                    else datasets.load_dataset(dataset_name)
                                )
                            except Exception as exc:
                                last_exc = exc
                        except Exception as exc:
                            last_exc = exc
                            continue
            return last_exc
        finally:
            if force_offline:
                if prev_hub_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_hub_offline
                if prev_datasets_offline is None:
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                else:
                    os.environ["HF_DATASETS_OFFLINE"] = prev_datasets_offline

    first_try = _attempt_load(force_offline=False)
    if not isinstance(first_try, Exception):
        return first_try

    second_try = _attempt_load(force_offline=True)
    if not isinstance(second_try, Exception):
        return second_try

    requested = cfg if cfg is not None else "default"
    raise RuntimeError(
        f"Failed to load LooGLE dataset (requested config={requested!r}) from any "
        f"known dataset id: {LOOGLE_HF_DATASET_IDS}."
    ) from second_try


def _extract_loogle_document_text(row: Dict[str, Any]) -> Optional[str]:
    for key in ("context", "input", "document", "text", "article", "passage"):
        text = _coerce_to_text(row.get(key)).strip()
        if text:
            return text
    return None


def _extract_loogle_doc_id(row: Dict[str, Any], *, fallback_index: int) -> str:
    for key in ("doc_id", "document_id", "docid", "title"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return f"doc_{fallback_index}"


def _to_text_list(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_to_text_list(item))
        return out
    text = _coerce_to_text(value).strip()
    if text:
        out.append(text)
    return out


def _parse_loogle_qa_pairs(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return []
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
    return []


def _iter_rows_for_split(dataset_dict: Any, *, split: str) -> Iterable[Dict[str, Any]]:
    rows = dataset_dict[split]
    for row in rows:
        if isinstance(row, dict):
            yield row


def load_loogle_bundle(loader_config: Dict[str, object]) -> DatasetBundle:
    import random

    split = str(loader_config.get("split") or "test")
    config_name = loader_config.get("config_name") or "shortdep_qa"
    qa_n = loader_config.get("qa_n", "all")
    qa_selection_method = str(loader_config.get("qa_selection_method") or "first")

    logger.info(
        "Loading LooGLE split=%s config=%s n=%s selection=%s",
        split,
        config_name,
        qa_n,
        qa_selection_method,
    )
    dataset_dict = _load_loogle_dataset(cfg=config_name)

    documents: OrderedDict[str, Dict[str, object]] = OrderedDict()
    qa_candidates: List[Dict[str, object]] = []

    for row_index, row in enumerate(_iter_rows_for_split(dataset_dict, split=split)):
        doc_id = _extract_loogle_doc_id(row, fallback_index=row_index)
        text = _extract_loogle_document_text(row) or ""
        if text and doc_id not in documents:
            documents[doc_id] = {
                **row,
                "doc_id": doc_id,
                "text": text,
            }

        qa_pairs = _parse_loogle_qa_pairs(row.get("qa_pairs"))
        if qa_pairs:
            for pair_index, pair in enumerate(qa_pairs):
                question = _coerce_to_text(pair.get("Q") or pair.get("question")).strip()
                if not question:
                    continue
                answer_texts = _to_text_list(pair.get("A") or pair.get("answer"))
                spans = _to_text_list(pair.get("S") or pair.get("evidence"))
                if answer_texts or spans:
                    qa_candidates.append(
                        {
                            "doc_id": doc_id,
                            "question": question,
                            "answers": answer_texts,
                            "retrieval_spans": spans,
                            "source_row_index": row_index,
                            "source_pair_index": pair_index,
                            "raw_qa_pair": pair,
                        }
                    )
            continue

        question = _coerce_to_text(
            row.get("question") or row.get("Q") or row.get("query")
        ).strip()
        if not question:
            continue
        answer_texts = _to_text_list(row.get("answer") or row.get("A") or row.get("answers"))
        spans = _to_text_list(row.get("evidence") or row.get("S") or row.get("span"))
        if answer_texts or spans:
            qa_candidates.append(
                {
                    "doc_id": doc_id,
                    "question": question,
                    "answers": answer_texts,
                    "retrieval_spans": spans,
                    "source_row_index": row_index,
                    "source_pair_index": None,
                    "raw_qa_pair": None,
                }
            )

    total = len(qa_candidates)
    if isinstance(qa_n, str):
        qa_n = total if qa_n.lower() == "all" else int(qa_n)
    qa_n = min(int(qa_n), total)
    indices = list(range(total))
    if qa_n < total:
        indices = (
            random.sample(indices, qa_n)
            if qa_selection_method == "random"
            else indices[:qa_n]
        )

    qa_entries: List[Dict[str, object]] = []
    for qa_index in indices:
        row = qa_candidates[qa_index]
        qa_entries.append(
            {
                "query_id": f"loogle_{qa_index}",
                "source_qa_index": qa_index,
                "doc_id": row["doc_id"],
                "document_id": row["doc_id"],
                "question": row["question"],
                "reference_answers": list(row["answers"]),
                "answers": list(row["answers"]),
                "retrieval_spans": list(row["retrieval_spans"]),
                "evidence_spans": list(row["retrieval_spans"]),
                "source_row_index": row["source_row_index"],
                "source_pair_index": row["source_pair_index"],
                "raw_qa_pair": row["raw_qa_pair"],
                "relevant_doc_ids": [row["doc_id"]],
            }
        )

    return DatasetBundle(
        documents=documents,
        qa_entries=qa_entries,
        metadata={
            "loader_type": "loogle",
            "split": split,
            "config_name": config_name,
            "qa_n": loader_config.get("qa_n", "all"),
            "qa_selection_method": qa_selection_method,
        },
    )


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
    if dataset_name in LOOGLE_DATASET_NAMES:
        return load_loogle_bundle(loader_config)
    if dataset_name in NARRATIVEQA_DATASET_NAMES:
        return load_narrativeqa_bundle(loader_config)
    if dataset_name in QUALITY_DATASET_NAMES:
        return load_quality_bundle(loader_config)
    if dataset_name in NOVELHOPQA_DATASET_NAMES:
        return load_novelhopqa_bundle(loader_config)

    if dataset_name not in DATASET_PRESETS:
        supported = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(
            f"Unsupported dataset '{loader_config['dataset_name']}'. "
            f"Supported presets: {supported}, loogle, narrativeqa, novelqa, qasper, quality"
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
