#!/usr/bin/env python3
"""Measured one- and two-chunk parity smoke test for the late-chunking path."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from chunked_pooling.experiment_chunking import build_encoder_chunk_mappings
from chunked_pooling.experiment_retrievers import DenseRetriever


MODEL_NAME = "facebook/contriever"
DOCUMENT = (
    "Late chunking lets a passage retain signals from the surrounding document. "
    "The smoke test compares contextual token pooling with independent chunk encoding. "
    "Evidence"
)


def _chunk_records(
    text: str,
    token_offsets: Sequence[Tuple[int, int]],
    token_spans: Sequence[Tuple[int, int]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for index, (token_start, token_end) in enumerate(token_spans):
        char_start = int(token_offsets[token_start][0])
        char_end = int(token_offsets[token_end - 1][1])
        records.append(
            {
                "doc_id": "smoke-document",
                "chunk_id": f"smoke-chunk-{index}",
                "chunk_index": index,
                "raw_text": text[char_start:char_end],
                "char_start": char_start,
                "char_end": char_end,
                "token_start": token_start,
                "token_end": token_end,
            }
        )
    return records


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left, right)
        / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def _measure_case(
    retriever: DenseRetriever,
    text: str,
    token_offsets: Sequence[Tuple[int, int]],
    token_spans: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    records = _chunk_records(text, token_offsets, token_spans)
    model_spans, mapping_metadata, _ = build_encoder_chunk_mappings(
        chunk_records=records,
        text=text,
        tokenizer=retriever.tokenizer,
        instruction_token_count=retriever.document_instruction_token_count(),
        instruction_text=retriever.document_instruction,
    )
    late_vectors, window_metadata = retriever.encode_late_chunks(
        text=text,
        model_token_spans=model_spans,
        max_tokens_per_forward=retriever.max_length,
        window_overlap_tokens=0,
    )
    naive_vectors = retriever._generic_encode(
        [str(record["raw_text"]) for record in records],
        retriever.document_instruction,
    )

    return {
        "token_spans": [list(span) for span in token_spans],
        "model_token_spans": [list(span) for span in model_spans],
        "exact_char_matches": [
            bool(metadata["exact_char_match"]) for metadata in mapping_metadata
        ],
        "windowing": window_metadata["segmentation_or_windowing_strategy"],
        "cosine_similarities": [
            _cosine(late, naive)
            for late, naive in zip(late_vectors, naive_vectors)
        ],
        "late_vector_norms": [
            float(np.linalg.norm(vector)) for vector in late_vectors
        ],
        "naive_vector_norms": [
            float(np.linalg.norm(vector)) for vector in naive_vectors
        ],
        "late_first_five": [
            [float(value) for value in vector[:5]] for vector in late_vectors
        ],
        "naive_first_five": [
            [float(value) for value in vector[:5]] for vector in naive_vectors
        ],
    }


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    retriever = DenseRetriever.from_config(
        {
            "name": "contriever",
            "type": "dense",
            "model_name": MODEL_NAME,
            "normalize": True,
            "distance_metric": "cosine",
            "pooling": "mean",
        }
    )
    tokenization = retriever.tokenizer(
        DOCUMENT,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_offsets = tokenization["offset_mapping"]
    token_count = len(token_offsets)
    midpoint = token_count // 2
    if midpoint == 0 or midpoint * 2 != token_count:
        raise RuntimeError(
            f"Smoke-test document must have a positive even token count, got {token_count}."
        )

    result = {
        "model": MODEL_NAME,
        "pooling": retriever.pooling,
        "normalize": retriever.normalize,
        "model_context_window": retriever.model_context_window,
        "document_token_count_without_special_tokens": token_count,
        "one_full_span_chunk": _measure_case(
            retriever,
            DOCUMENT,
            token_offsets,
            [(0, token_count)],
        ),
        "two_equal_chunks": _measure_case(
            retriever,
            DOCUMENT,
            token_offsets,
            [(0, midpoint), (midpoint, token_count)],
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
