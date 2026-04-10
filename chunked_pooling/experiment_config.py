from __future__ import annotations

import copy
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


RETRIEVER_ALIASES = {
    "jina": {
        "name": "jina",
        "type": "dense",
        "model_name": "jinaai/jina-embeddings-v2-small-en",
        "normalize": True,
        "distance_metric": "cosine",
    },
    "jina-base": {
        "name": "jina-base",
        "type": "dense",
        "model_name": "jinaai/jina-embeddings-v2-base-en",
        "normalize": True,
        "distance_metric": "cosine",
    },
    "jina-v3": {
        "name": "jina-v3",
        "type": "dense",
        "model_name": "jinaai/jina-embeddings-v3",
        "normalize": True,
        "distance_metric": "cosine",
    },
    "contriever": {
        "name": "contriever",
        "type": "dense",
        "model_name": "facebook/contriever",
        "normalize": True,
        "distance_metric": "cosine",
    },
    "qwen": {
        "name": "qwen",
        "type": "dense",
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "tokenizer_name": "Qwen/Qwen3-Embedding-8B",
        "normalize": True,
        "distance_metric": "cosine",
        "pooling": "last_token",
        "padding_side": "left",
        "max_length": 32768,
        "query_prompt": (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query:"
        ),
    },
    "bm25": {
        "name": "bm25",
        "type": "bm25",
        "model_name": "bm25",
        "normalize": False,
        "distance_metric": "bm25",
    },
}


def load_yaml_file(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("The default experiment YAML must parse to a mapping.")
    return data


def _get_path(config: Dict[str, object], path: Sequence[str]):
    current = config
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _pick(
    config: Dict[str, object],
    used_paths: set,
    *paths: Sequence[str],
    default=None,
):
    for path in paths:
        value = _get_path(config, path)
        if value is not None:
            used_paths.add(tuple(path))
            return value
    return default


def _flatten_paths(node, prefix=()) -> List[Tuple[str, ...]]:
    if isinstance(node, dict):
        items = []
        for key, value in node.items():
            next_prefix = prefix + (str(key),)
            items.append(next_prefix)
            items.extend(_flatten_paths(value, next_prefix))
        return items
    return []


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_int(value, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    return int(value)


def parse_retriever_spec(spec) -> Dict[str, object]:
    if isinstance(spec, dict):
        base = copy.deepcopy(spec)
    else:
        raw_spec = str(spec).strip()
        lower_spec = raw_spec.lower()
        if lower_spec in RETRIEVER_ALIASES:
            base = copy.deepcopy(RETRIEVER_ALIASES[lower_spec])
        elif raw_spec.startswith("{"):
            base = json.loads(raw_spec)
        else:
            base = {}
            parts = [part.strip() for part in raw_spec.split(",") if part.strip()]
            for part in parts:
                if "=" not in part:
                    if part in {"dense", "bm25"}:
                        base["type"] = part
                    else:
                        base["name"] = part
                    continue
                key, value = part.split("=", 1)
                base[key.strip()] = value.strip()

    retriever_type = str(base.get("type") or "dense").lower()
    name = str(base.get("name") or base.get("model_name") or retriever_type)

    if name.lower() in RETRIEVER_ALIASES and "model_name" not in base:
        alias_defaults = copy.deepcopy(RETRIEVER_ALIASES[name.lower()])
        alias_defaults.update(base)
        base = alias_defaults
        retriever_type = str(base.get("type") or retriever_type)
        name = str(base.get("name") or name)

    if retriever_type == "bm25":
        base.setdefault("model_name", "bm25")
        base.setdefault("distance_metric", "bm25")
        base.setdefault("normalize", False)
    else:
        if "model_name" not in base:
            raise ValueError(f"Retriever spec '{spec}' is missing model_name.")
        base.setdefault("distance_metric", "cosine")
        base["normalize"] = _to_bool(base.get("normalize"), default=True)
        base.setdefault("pooling", "mean")

    base["name"] = name
    base["type"] = retriever_type
    if "max_length" in base:
        base["max_length"] = _to_int(base["max_length"])
    if "window_overlap_tokens" in base:
        base["window_overlap_tokens"] = _to_int(base["window_overlap_tokens"], 0)
    return base


def parse_retriever_specs(
    cli_specs: Sequence[str],
    default_experiment: Dict[str, object],
) -> List[Dict[str, object]]:
    specs = list(cli_specs)
    if not specs:
        yaml_specs = default_experiment.get("retrievers") or default_experiment.get(
            "retriever_specs"
        )
        if not yaml_specs:
            retrieval_cfg = default_experiment.get("retrieval") or {}
            yaml_specs = retrieval_cfg.get("retriever")
        if isinstance(yaml_specs, list):
            specs = yaml_specs
        elif yaml_specs:
            specs = [yaml_specs]

    if not specs:
        specs = ["jina"]

    return [parse_retriever_spec(spec) for spec in specs]


def resolve_run_name(
    dataset_name: str,
    run_name_override: Optional[str],
    chunking_config: Dict[str, object],
    retrievers: Sequence[Dict[str, object]],
) -> str:
    if run_name_override:
        return run_name_override

    retriever_names = "+".join(retriever["name"] for retriever in retrievers)
    strategy = str(chunking_config.get("strategy") or "fixed")
    if strategy == "fixed":
        size_part = f"c{chunking_config.get('chunk_size')}"
    else:
        size_part = f"n{chunking_config.get('n_sentences')}"
    overlap_part = f"o{chunking_config.get('overlap') or 0}"
    return f"{dataset_name}__{strategy}-{size_part}-{overlap_part}__{retriever_names}"


def resolve_run_config(
    dataset_name: str,
    default_experiment: Dict[str, object],
    retrievers: Sequence[Dict[str, object]],
    run_name_override: Optional[str] = None,
    output_root_override: Optional[str] = None,
    resume: bool = True,
    overrides: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, object], List[str]]:
    used_paths = set()
    notes: List[str] = []
    overrides = overrides or {}

    dataset_loader_config = {
        "type": _pick(
            default_experiment,
            used_paths,
            ("dataset_loader", "type"),
            ("loader", "type"),
            default="task_registry",
        ),
        "dataset_name": dataset_name,
        "split": _pick(
            default_experiment,
            used_paths,
            ("dataset", "split"),
            ("dataset_loader", "split"),
            ("eval_split",),
            ("split",),
            default="test",
        ),
        "config_name": _pick(
            default_experiment,
            used_paths,
            ("dataset", "config_name"),
            ("dataset_loader", "config_name"),
            default=None,
        ),
        "docs_config_name": _pick(
            default_experiment,
            used_paths,
            ("dataset", "docs_config_name"),
            ("dataset_loader", "docs_config_name"),
            default=None,
        ),
        "qa_config_name": _pick(
            default_experiment,
            used_paths,
            ("dataset", "qa_config_name"),
            ("dataset_loader", "qa_config_name"),
            default=None,
        ),
        "qa_n": _pick(
            default_experiment,
            used_paths,
            ("dataset", "qa_n"),
            ("dataset_loader", "qa_n"),
            default=None,
        ),
        "qa_selection_method": _pick(
            default_experiment,
            used_paths,
            ("dataset", "qa_selection_method"),
            ("dataset_loader", "qa_selection_method"),
            default=None,
        ),
        "max_docs": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("sample", "max_documents"),
                ("dataset", "max_docs"),
                ("dataset_loader", "max_docs"),
                ("max_docs",),
            )
        ),
        "max_questions": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("sample", "max_qa_entries"),
                ("dataset", "max_questions"),
                ("dataset_loader", "max_questions"),
                ("max_questions",),
            )
        ),
        "selected_doc_ids": _pick(
            default_experiment,
            used_paths,
            ("dataset", "selected_doc_ids"),
            ("dataset_loader", "selected_doc_ids"),
            default=None,
        ),
        "selected_query_ids": _pick(
            default_experiment,
            used_paths,
            ("dataset", "selected_query_ids"),
            ("dataset_loader", "selected_query_ids"),
            default=None,
        ),
        "prepend_title": _to_bool(
            _pick(
                default_experiment,
                used_paths,
                ("dataset_loader", "prepend_title"),
                ("dataset", "prepend_title"),
                default=True,
            ),
            default=True,
        ),
    }

    local_loader = _pick(
        default_experiment,
        used_paths,
        ("dataset_loader", "local_json"),
        default=None,
    )
    if isinstance(local_loader, dict):
        dataset_loader_config["local_json"] = local_loader

    chunking_config = {
        "strategy": _pick(
            default_experiment,
            used_paths,
            ("chunking", "strategy"),
            ("ingest", "late_chunking_strategy"),
            ("strategy",),
            default="fixed",
        ),
        "chunk_size": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("chunking", "chunk_size"),
                ("ingest", "chunk_size"),
                ("chunk_size",),
                default=256,
            ),
            default=256,
        ),
        "overlap": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("chunking", "overlap"),
                ("ingest", "chunk_overlap"),
                ("overlap",),
                default=0,
            ),
            default=0,
        ),
        "n_sentences": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("chunking", "n_sentences"),
                ("n_sentences",),
                default=1,
            ),
            default=1,
        ),
        "sentence_overlap": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("chunking", "sentence_overlap"),
                default=0,
            ),
            default=0,
        ),
        "semantic_model_name": _pick(
            default_experiment,
            used_paths,
            ("chunking", "semantic_model_name"),
            ("chunking_model",),
            default=None,
        ),
        "tokenizer_name": _pick(
            default_experiment,
            used_paths,
            ("chunking", "tokenizer_name"),
            ("tokenizer_name",),
            ("model_name",),
            default=None,
        ),
    }

    late_chunking_config = {
        "enabled": True,
        "max_tokens_per_forward": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("late_chunking", "max_tokens_per_forward"),
                ("truncate_max_length",),
                default=None,
            )
        ),
        "window_overlap_tokens": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("late_chunking", "window_overlap_tokens"),
                ("long_late_chunking_overlap_size",),
                default=256,
            ),
            default=256,
        ),
        "window_size_tokens": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("late_chunking", "window_size_tokens"),
                ("long_late_chunking_embed_size",),
                default=None,
            )
        ),
    }

    retrieval_scope = _pick(
        default_experiment,
        used_paths,
        ("retrieval", "scope"),
        ("retrieval_scope",),
        default="per_document",
    )
    retrieval_config = {
        "scope": retrieval_scope,
        "retrieve_k": _to_int(
            _pick(
                default_experiment,
                used_paths,
                ("retrieval", "retrieve_k"),
                ("retrieve_k",),
                ("top_k",),
                default=10,
            ),
            default=10,
        ),
    }

    output_root = output_root_override or _pick(
        default_experiment,
        used_paths,
        ("output", "root"),
        default="late_chunk_runs",
    )

    profiling_config = {
        "capture_resource_usage": _to_bool(
            _pick(
                default_experiment,
                used_paths,
                ("profiling", "capture_resource_usage"),
                default=True,
            ),
            default=True,
        )
    }

    override_mappings = [
        (dataset_loader_config, "max_docs", "max_docs", "dataset_loader.max_docs"),
        (
            dataset_loader_config,
            "max_questions",
            "max_questions",
            "dataset_loader.max_questions",
        ),
        (
            chunking_config,
            "strategy",
            "chunking_strategy",
            "chunking.strategy",
        ),
        (chunking_config, "chunk_size", "chunk_size", "chunking.chunk_size"),
        (chunking_config, "overlap", "chunk_overlap", "chunking.overlap"),
        (chunking_config, "n_sentences", "n_sentences", "chunking.n_sentences"),
        (
            chunking_config,
            "sentence_overlap",
            "sentence_overlap",
            "chunking.sentence_overlap",
        ),
        (
            chunking_config,
            "tokenizer_name",
            "chunk_tokenizer_name",
            "chunking.tokenizer_name",
        ),
        (retrieval_config, "retrieve_k", "retrieve_k", "retrieval.retrieve_k"),
        (
            retrieval_config,
            "scope",
            "retrieval_scope",
            "retrieval.scope",
        ),
        (
            late_chunking_config,
            "max_tokens_per_forward",
            "late_max_tokens_per_forward",
            "late_chunking.max_tokens_per_forward",
        ),
        (
            late_chunking_config,
            "window_overlap_tokens",
            "late_window_overlap_tokens",
            "late_chunking.window_overlap_tokens",
        ),
    ]
    for container, key, override_key, label in override_mappings:
        value = overrides.get(override_key)
        if value is None:
            continue
        container[key] = value
        notes.append(f"{label} overridden from the command line: {value}")

    run_name = resolve_run_name(
        dataset_name=dataset_name,
        run_name_override=run_name_override
        or _pick(default_experiment, used_paths, ("output", "run_name"), default=None),
        chunking_config=chunking_config,
        retrievers=retrievers,
    )

    if dataset_loader_config["type"] == "task_registry" and not dataset_loader_config.get(
        "split"
    ):
        dataset_loader_config["split"] = "test"

    if not dataset_loader_config["docs_config_name"]:
        dataset_loader_config["docs_config_name"] = dataset_loader_config["config_name"]
    if not dataset_loader_config["qa_config_name"]:
        dataset_loader_config["qa_config_name"] = dataset_loader_config["config_name"]

    if dataset_loader_config["qa_selection_method"] is None:
        dataset_loader_config["qa_selection_method"] = "first"
    if dataset_loader_config["qa_n"] is None:
        dataset_loader_config["qa_n"] = "all"

    if chunking_config["tokenizer_name"] is None:
        first_dense = next(
            (retriever for retriever in retrievers if retriever["type"] == "dense"),
            None,
        )
        if first_dense is not None:
            chunking_config["tokenizer_name"] = first_dense["model_name"]
            notes.append(
                "chunking.tokenizer_name was not set; the first dense retriever model "
                f"('{first_dense['model_name']}') was used as the canonical chunking tokenizer."
            )
        else:
            chunking_config["tokenizer_name"] = "bert-base-uncased"
            notes.append(
                "chunking.tokenizer_name was not set and no dense retriever was configured; "
                "falling back to 'bert-base-uncased' for canonical chunk boundaries."
            )

    ingest_strategy = _get_path(default_experiment, ("ingest", "strategy"))
    if ingest_strategy is not None and str(ingest_strategy) not in {
        "fixed",
        "sentences",
        "semantic",
    }:
        chunking_config["strategy"] = "fixed"
        notes.append(
            f"ingest.strategy='{ingest_strategy}' does not map directly to late chunking; "
            "using fixed token chunking and reusing ingest.chunk_size/chunk_overlap."
        )

    if late_chunking_config["window_size_tokens"] is not None:
        late_chunking_config["max_tokens_per_forward"] = late_chunking_config[
            "window_size_tokens"
        ]

    if chunking_config["strategy"] == "semantic" and chunking_config["overlap"]:
        notes.append(
            "chunking.overlap is ignored for semantic chunking because semantic boundaries "
            "are determined by the semantic splitter."
        )
        chunking_config["overlap"] = 0

    config = {
        "dataset_name": dataset_name,
        "run_name": run_name,
        "resume": resume,
        "output_root": output_root,
        "dataset_loader": dataset_loader_config,
        "chunking": chunking_config,
        "late_chunking": late_chunking_config,
        "retrievers": [copy.deepcopy(retriever) for retriever in retrievers],
        "retrieval": retrieval_config,
        "profiling": profiling_config,
    }

    all_paths = set(_flatten_paths(default_experiment))
    ignored_paths = sorted(
        ".".join(path)
        for path in all_paths
        if path not in used_paths and not any(path[: len(used)] == used for used in used_paths)
    )
    if ignored_paths:
        notes.append(
            "Ignored YAML fields without a direct mapping: " + ", ".join(ignored_paths)
        )

    return config, notes
