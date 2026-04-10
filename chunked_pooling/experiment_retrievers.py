from __future__ import annotations

import importlib.util
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from packaging.version import Version
from transformers import AutoTokenizer

from chunked_pooling import chunked_pooling
from chunked_pooling.experiment_chunking import resolve_model_max_length
from chunked_pooling.wrappers import load_model


def _sanitize_positive_length(value) -> Optional[int]:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value > 100_000:
        return None
    return value


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_torch_dtype(value):
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value

    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized == "auto":
        return "auto"

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch_dtype value: {value}")
    return mapping[normalized]


def _accelerate_is_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _unwrap_model(model):
    return getattr(model, "_model", model)


def _resolve_input_device(model) -> torch.device:
    base_model = _unwrap_model(model)
    hf_device_map = getattr(base_model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str) and mapped_device.startswith("cuda"):
                return torch.device(mapped_device)

    for candidate in (model, base_model):
        try:
            return next(candidate.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            continue

    device = getattr(base_model, "device", None) or getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cpu")


def _validate_runtime_requirements(config: Dict[str, object]) -> None:
    min_transformers_version = config.get("min_transformers_version")
    if min_transformers_version is None:
        return

    installed_version = Version(transformers.__version__)
    required_version = Version(str(min_transformers_version))
    if installed_version >= required_version:
        return

    retriever_name = str(config.get("name") or config.get("model_name") or "retriever")
    model_name = str(config.get("model_name") or retriever_name)
    raise RuntimeError(
        f"Retriever '{retriever_name}' requires transformers>={required_version}, "
        f"but the current environment has transformers=={installed_version}. "
        f"This is required to load '{model_name}' correctly; older versions fail on "
        "the Qwen3 architecture with errors like KeyError: 'qwen3'. Upgrade the "
        "environment or remove this retriever from --retriever / RETRIEVERS."
    )


def _build_model_load_kwargs(config: Dict[str, object]) -> Dict[str, object]:
    load_kwargs: Dict[str, object] = {}

    device_map = config.get("device_map")
    if device_map is None and _to_bool(
        config.get("shard_across_available_gpus"), default=False
    ):
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device_map = "auto"

    if device_map is not None:
        if not _accelerate_is_available():
            raise RuntimeError(
                "Multi-GPU model sharding requires the 'accelerate' package. "
                "Install accelerate or disable model sharding for this retriever."
            )
        load_kwargs["device_map"] = device_map
        load_kwargs["low_cpu_mem_usage"] = _to_bool(
            config.get("low_cpu_mem_usage"), default=True
        )
    elif "low_cpu_mem_usage" in config:
        load_kwargs["low_cpu_mem_usage"] = _to_bool(
            config.get("low_cpu_mem_usage"), default=False
        )

    torch_dtype = _parse_torch_dtype(config.get("torch_dtype"))
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    attn_implementation = config.get("attn_implementation")
    if attn_implementation:
        load_kwargs["attn_implementation"] = str(attn_implementation)

    return load_kwargs


def _resolve_model_context_window(model, tokenizer) -> int:
    candidates: List[int] = []

    tokenizer_max_length = _sanitize_positive_length(
        getattr(tokenizer, "model_max_length", None)
    )
    if tokenizer_max_length is not None:
        candidates.append(tokenizer_max_length)

    model_config = getattr(model, "config", None)
    if model_config is not None:
        for attribute_name in (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "seq_length",
        ):
            value = _sanitize_positive_length(getattr(model_config, attribute_name, None))
            if value is not None:
                candidates.append(value)

    if not candidates:
        return resolve_model_max_length(tokenizer)
    return min(candidates)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = _to_numpy(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _mean_pool(last_hidden_state, attention_mask):
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * expanded_mask, dim=1)
    counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed / counts


def _last_token_pool(last_hidden_state, attention_mask):
    left_padding = bool(
        (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    )
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
    ]


@dataclass
class DenseRetriever:
    name: str
    model_name: str
    tokenizer_name: str
    normalize: bool
    distance_metric: str
    model: object
    tokenizer: object
    has_instructions: bool
    query_instruction: str
    document_instruction: str
    use_builtin_query_encoder: bool
    device: torch.device
    max_length: int
    model_context_window: int
    pooling: str

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "DenseRetriever":
        _validate_runtime_requirements(config)
        model_load_kwargs = _build_model_load_kwargs(config)
        model, has_instructions = load_model(
            str(config["model_name"]), **model_load_kwargs
        )
        tokenizer_name = str(config.get("tokenizer_name") or config["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        padding_side = str(config.get("padding_side") or getattr(tokenizer, "padding_side", "right"))
        tokenizer.padding_side = padding_side

        if torch.cuda.is_available() and "device_map" not in model_load_kwargs:
            model = model.cuda()
        model.eval()

        query_override = config.get("query_prompt") is not None
        document_override = config.get("document_prompt") is not None
        query_instruction = str(config.get("query_prompt") or "")
        document_instruction = str(config.get("document_prompt") or "")
        if has_instructions and hasattr(model, "get_instructions"):
            instructions = model.get_instructions()
            if not query_override:
                query_instruction = instructions[0]
            if not document_override:
                document_instruction = instructions[1]

        model_context_window = _resolve_model_context_window(model, tokenizer)
        configured_max_length = _sanitize_positive_length(config.get("max_length"))
        max_length = (
            min(configured_max_length, model_context_window)
            if configured_max_length is not None
            else model_context_window
        )

        return cls(
            name=str(config["name"]),
            model_name=str(config["model_name"]),
            tokenizer_name=tokenizer_name,
            normalize=bool(config.get("normalize", True)),
            distance_metric=str(config.get("distance_metric") or "cosine"),
            model=model,
            tokenizer=tokenizer,
            has_instructions=has_instructions,
            query_instruction=query_instruction,
            document_instruction=document_instruction,
            use_builtin_query_encoder=hasattr(model, "encode_queries") and not query_override,
            device=_resolve_input_device(model),
            max_length=max_length,
            model_context_window=model_context_window,
            pooling=str(config.get("pooling") or "mean"),
        )

    def _tokenize_with_prompt(self, texts: Sequence[str], prompt: str):
        prompt = prompt or ""
        return self.tokenizer(
            [prompt + text for text in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def _generic_encode(self, texts: Sequence[str], prompt: str) -> np.ndarray:
        inputs = self._tokenize_with_prompt(texts, prompt)
        if self.device.type == "cuda":
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs[0]
        if self.pooling == "last_token":
            embeddings = _last_token_pool(last_hidden_state, inputs["attention_mask"])
        else:
            embeddings = _mean_pool(last_hidden_state, inputs["attention_mask"])
        embeddings = embeddings.detach().cpu().numpy()
        return normalize_rows(embeddings) if self.normalize else embeddings

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.use_builtin_query_encoder and hasattr(self.model, "encode_queries"):
            embeddings = self.model.encode_queries(list(texts))
            embeddings = _to_numpy(embeddings)
            return normalize_rows(embeddings) if self.normalize else embeddings

        if hasattr(self.model, "encode") and not self.query_instruction:
            embeddings = self.model.encode(list(texts))
            embeddings = _to_numpy(embeddings)
            return normalize_rows(embeddings) if self.normalize else embeddings

        return self._generic_encode(texts, self.query_instruction)

    def document_instruction_token_count(self) -> int:
        if not self.document_instruction:
            return 0
        return len(
            self.tokenizer(
                self.document_instruction,
                add_special_tokens=False,
            )["input_ids"]
        )

    def effective_max_tokens_per_forward(
        self, requested_max_tokens_per_forward: Optional[int]
    ) -> int:
        requested = _sanitize_positive_length(requested_max_tokens_per_forward)
        if requested is None:
            return self.model_context_window
        return min(requested, self.model_context_window)

    def _forward_document_embeddings(
        self,
        text: str,
        max_tokens_per_forward: Optional[int],
        window_overlap_tokens: int,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        effective_max_tokens_per_forward = self.effective_max_tokens_per_forward(
            max_tokens_per_forward
        )
        prompt = self.document_instruction or ""
        model_inputs = self.tokenizer(
            prompt + text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        total_input_tokens = int(model_inputs["input_ids"].shape[1])

        if self.device.type == "cuda":
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

        if (
            total_input_tokens <= effective_max_tokens_per_forward
        ):
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            return outputs[0], {
                "segmentation_or_windowing_strategy": "single_forward",
                "effective_max_tokens_per_forward": effective_max_tokens_per_forward,
                "encoder_windows": [
                    {
                        "window_index": 0,
                        "input_token_start": 0,
                        "input_token_end": total_input_tokens,
                        "dropped_left_tokens": 0,
                        "kept_token_start": 0,
                        "kept_token_end": total_input_tokens,
                    }
                ],
                "full_model_input_token_count": total_input_tokens,
            }

        if (
            window_overlap_tokens < 0
            or window_overlap_tokens >= effective_max_tokens_per_forward
        ):
            raise ValueError(
                "window_overlap_tokens must be >= 0 and smaller than the effective "
                "max_tokens_per_forward for this retriever."
            )

        outputs = []
        windows = []
        step = effective_max_tokens_per_forward - window_overlap_tokens
        window_index = 0
        for start in range(0, total_input_tokens, step):
            end = min(start + effective_max_tokens_per_forward, total_input_tokens)
            window_inputs = {
                key: value[:, start:end] for key, value in model_inputs.items()
            }
            with torch.no_grad():
                window_output = self.model(**window_inputs)[0]

            dropped_left_tokens = 0 if start == 0 else window_overlap_tokens
            kept_output = (
                window_output
                if dropped_left_tokens == 0
                else window_output[:, dropped_left_tokens:]
            )
            kept_token_start = start + dropped_left_tokens
            kept_token_end = start + int(window_output.shape[1])
            windows.append(
                {
                    "window_index": window_index,
                    "input_token_start": start,
                    "input_token_end": end,
                    "dropped_left_tokens": dropped_left_tokens,
                    "kept_token_start": kept_token_start,
                    "kept_token_end": kept_token_end,
                }
            )
            outputs.append(kept_output)
            window_index += 1
            if end >= total_input_tokens:
                break

        return torch.cat(outputs, dim=1), {
            "segmentation_or_windowing_strategy": "sliding_windows",
            "effective_max_tokens_per_forward": effective_max_tokens_per_forward,
            "encoder_windows": windows,
            "full_model_input_token_count": total_input_tokens,
        }

    def encode_late_chunks(
        self,
        text: str,
        model_token_spans: Sequence[Tuple[int, int]],
        max_tokens_per_forward: Optional[int],
        window_overlap_tokens: int,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        token_embeddings, window_metadata = self._forward_document_embeddings(
            text=text,
            max_tokens_per_forward=max_tokens_per_forward,
            window_overlap_tokens=window_overlap_tokens,
        )
        output_embeddings = chunked_pooling(
            [token_embeddings],
            [list(model_token_spans)],
            max_length=None,
        )[0]
        matrix = np.vstack([_to_numpy(embedding) for embedding in output_embeddings])
        if self.normalize:
            matrix = normalize_rows(matrix)
        return matrix, window_metadata


class BM25Index:
    def __init__(self, chunk_records: Sequence[Dict[str, object]]):
        self.chunk_records = list(chunk_records)
        self.chunk_ids = [str(chunk["chunk_id"]) for chunk in self.chunk_records]
        self.doc_to_indices: Dict[str, List[int]] = {}
        for index, chunk in enumerate(self.chunk_records):
            self.doc_to_indices.setdefault(str(chunk["doc_id"]), []).append(index)

        self.documents = [self._tokenize(str(chunk["raw_text"])) for chunk in self.chunk_records]
        self.doc_lengths = np.array([len(tokens) for tokens in self.documents], dtype=np.float64)
        self.average_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.documents]

        document_frequency = Counter()
        for tokens in self.documents:
            for term in set(tokens):
                document_frequency[term] += 1

        self.idf = {}
        total_docs = len(self.documents)
        for term, df in document_frequency.items():
            self.idf[term] = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))

        self.k1 = 1.5
        self.b = 0.75

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(
        self,
        query_text: str,
        top_k: int,
        candidate_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[List[int], List[float]]:
        query_terms = self._tokenize(query_text)
        if candidate_indices is None:
            candidate_indices = list(range(len(self.documents)))

        scores = []
        for index in candidate_indices:
            tf = self.term_frequencies[index]
            doc_length = self.doc_lengths[index]
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                numerator = tf[term] * (self.k1 + 1.0)
                denominator = tf[term] + self.k1 * (
                    1.0 - self.b + self.b * (doc_length / max(self.average_doc_length, 1.0))
                )
                score += idf * (numerator / max(denominator, 1e-9))
            scores.append((index, float(score)))

        scores.sort(key=lambda item: item[1], reverse=True)
        top_hits = scores[: min(top_k, len(scores))]
        return [index for index, _ in top_hits], [score for _, score in top_hits]
