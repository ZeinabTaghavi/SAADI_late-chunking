from __future__ import annotations

import bisect
import hashlib
from typing import Dict, Iterable, List, Sequence, Tuple

from chunked_pooling.chunking import Chunker


def resolve_model_max_length(tokenizer, fallback: int = 8192) -> int:
    model_max_length = getattr(tokenizer, "model_max_length", fallback)
    if model_max_length is None or model_max_length > 100_000:
        return fallback
    return min(fallback, int(model_max_length))


def make_chunking_signature(chunking_config: Dict[str, object]) -> str:
    payload = "|".join(
        [
            str(chunking_config.get("strategy")),
            str(chunking_config.get("chunk_size")),
            str(chunking_config.get("overlap")),
            str(chunking_config.get("n_sentences")),
            str(chunking_config.get("sentence_overlap")),
            str(chunking_config.get("semantic_model_name")),
            str(chunking_config.get("tokenizer_name")),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def make_chunk_id(
    doc_id: str,
    chunk_index: int,
    char_start: int,
    char_end: int,
    chunking_signature: str,
) -> str:
    digest = hashlib.sha1(
        f"{doc_id}|{chunk_index}|{char_start}|{char_end}|{chunking_signature}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{doc_id}__chunk_{chunk_index:05d}__{digest}"


def _fixed_token_spans(
    token_count: int,
    chunk_size: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    if chunk_size is None or chunk_size < 1:
        raise ValueError("chunk_size must be >= 1 for fixed chunking.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")
    if token_count == 0:
        return []

    spans = []
    step = chunk_size - overlap
    start = 0
    while start < token_count:
        end = min(start + chunk_size, token_count)
        spans.append((start, end))
        if end >= token_count:
            break
        start += step
    return spans


def _sentence_boundaries(text: str, tokenizer) -> List[Tuple[int, int]]:
    tokens = tokenizer.encode_plus(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_list = tokens.tokens(0)
    spans = []
    sentence_start = 0

    for index in range(len(token_list)):
        is_terminal = token_list[index] in (".", "!", "?")
        if not is_terminal:
            continue
        at_end = index + 1 == len(token_list)
        next_starts_new_text = at_end or (
            tokens.token_to_chars(index).end != tokens.token_to_chars(index + 1).start
        )
        if next_starts_new_text:
            spans.append((sentence_start, index + 1))
            sentence_start = index + 1

    if sentence_start < len(token_list):
        spans.append((sentence_start, len(token_list)))

    return spans


def _group_sentence_spans(
    sentence_spans: Sequence[Tuple[int, int]],
    n_sentences: int,
    sentence_overlap: int,
) -> List[Tuple[int, int]]:
    if n_sentences is None or n_sentences < 1:
        raise ValueError("n_sentences must be >= 1 for sentence chunking.")
    if sentence_overlap < 0 or sentence_overlap >= n_sentences:
        raise ValueError("sentence_overlap must be >= 0 and < n_sentences.")
    if not sentence_spans:
        return []

    grouped = []
    step = n_sentences - sentence_overlap
    start_sentence = 0
    while start_sentence < len(sentence_spans):
        end_sentence = min(start_sentence + n_sentences, len(sentence_spans))
        start_token = sentence_spans[start_sentence][0]
        end_token = sentence_spans[end_sentence - 1][1]
        grouped.append((start_token, end_token))
        if end_sentence >= len(sentence_spans):
            break
        start_sentence += step
    return grouped


def compute_canonical_chunk_spans(
    text: str,
    tokenizer,
    chunking_config: Dict[str, object],
) -> List[Tuple[int, int]]:
    strategy = str(chunking_config.get("strategy") or "fixed")
    overlap = int(chunking_config.get("overlap") or 0)
    sentence_overlap = int(chunking_config.get("sentence_overlap") or 0)

    if strategy == "fixed":
        tokenization = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        return _fixed_token_spans(
            token_count=len(tokenization["offset_mapping"]),
            chunk_size=int(chunking_config.get("chunk_size") or 256),
            overlap=overlap,
        )

    if strategy == "sentences":
        sentence_spans = _sentence_boundaries(text, tokenizer)
        return _group_sentence_spans(
            sentence_spans=sentence_spans,
            n_sentences=int(chunking_config.get("n_sentences") or 1),
            sentence_overlap=sentence_overlap,
        )

    if strategy == "semantic":
        if overlap:
            raise ValueError("Token overlap is not supported for semantic chunking.")
        chunker = Chunker(chunking_strategy="semantic")
        return chunker.chunk(
            text,
            tokenizer=tokenizer,
            chunking_strategy="semantic",
            embedding_model_name=chunking_config.get("semantic_model_name"),
        )

    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def build_chunk_records(
    doc_id: str,
    text: str,
    tokenizer,
    chunking_config: Dict[str, object],
    chunking_signature: str,
) -> Tuple[List[Dict[str, object]], int]:
    tokenization = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_offsets = tokenization["offset_mapping"]
    spans = compute_canonical_chunk_spans(text, tokenizer, chunking_config)

    records = []
    for chunk_index, (token_start, token_end) in enumerate(spans):
        if token_start >= token_end:
            continue
        char_start = int(token_offsets[token_start][0])
        char_end = int(token_offsets[token_end - 1][1])
        chunk_id = make_chunk_id(
            doc_id=doc_id,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            chunking_signature=chunking_signature,
        )
        records.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "raw_text": text[char_start:char_end],
                "char_start": char_start,
                "char_end": char_end,
                "token_start": token_start,
                "token_end": token_end,
            }
        )

    return records, len(token_offsets)


def char_span_to_token_span(
    token_offsets: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Tuple[int, int]:
    start_candidates = [offset[0] for offset in token_offsets]
    end_candidates = [offset[1] for offset in token_offsets]

    token_start = bisect.bisect_left(start_candidates, char_start)
    if token_start >= len(token_offsets):
        token_start = len(token_offsets) - 1
    while token_start > 0 and token_offsets[token_start][0] > char_start:
        token_start -= 1
    while token_start < len(token_offsets) and token_offsets[token_start][1] <= char_start:
        token_start += 1

    token_end = bisect.bisect_left(start_candidates, char_end)
    while token_end < len(token_offsets) and token_offsets[token_end][0] < char_end:
        token_end += 1

    token_start = max(0, min(token_start, len(token_offsets)))
    token_end = max(token_start, min(token_end, len(token_offsets)))
    return token_start, token_end


def extend_special_tokens(
    annotations: Sequence[Tuple[int, int]],
    n_instruction_tokens: int = 0,
    include_prefix: bool = True,
    include_sep: bool = True,
) -> List[Tuple[int, int]]:
    new_annotations = []
    for index, (start, end) in enumerate(annotations):
        add_left_offset = 1 if (not include_prefix) or int(index > 0) else 0
        left_offset = 1 + n_instruction_tokens
        left = start + add_left_offset * left_offset

        add_sep = 1 if include_sep and ((index + 1) == len(annotations)) else 0
        right_offset = left_offset + add_sep
        right = end + right_offset
        new_annotations.append((left, right))
    return new_annotations


def build_encoder_chunk_mappings(
    chunk_records: Sequence[Dict[str, object]],
    text: str,
    tokenizer,
    instruction_token_count: int = 0,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, object]], int]:
    tokenization = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_offsets = tokenization["offset_mapping"]

    encoder_token_spans = []
    per_chunk_metadata = []

    for chunk in chunk_records:
        token_start, token_end = char_span_to_token_span(
            token_offsets=token_offsets,
            char_start=int(chunk["char_start"]),
            char_end=int(chunk["char_end"]),
        )
        if token_end <= token_start:
            continue
        encoder_token_spans.append((token_start, token_end))

        pooled_char_start = int(token_offsets[token_start][0])
        pooled_char_end = int(token_offsets[token_end - 1][1])
        per_chunk_metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "canonical_char_start": chunk["char_start"],
                "canonical_char_end": chunk["char_end"],
                "canonical_token_start": chunk["token_start"],
                "canonical_token_end": chunk["token_end"],
                "encoder_doc_token_start": token_start,
                "encoder_doc_token_end": token_end,
                "pooled_char_start": pooled_char_start,
                "pooled_char_end": pooled_char_end,
                "exact_char_match": (
                    pooled_char_start == int(chunk["char_start"])
                    and pooled_char_end == int(chunk["char_end"])
                ),
            }
        )

    model_token_spans = extend_special_tokens(
        encoder_token_spans,
        n_instruction_tokens=instruction_token_count,
    )
    for metadata, (model_start, model_end) in zip(per_chunk_metadata, model_token_spans):
        metadata["encoder_model_token_start"] = model_start
        metadata["encoder_model_token_end"] = model_end

    return model_token_spans, per_chunk_metadata, len(token_offsets)


def chunk_records_to_index_maps(
    chunk_records: Sequence[Dict[str, object]],
) -> Tuple[List[str], Dict[str, int], Dict[str, List[int]]]:
    chunk_ids = []
    chunk_id_to_index = {}
    doc_to_indices: Dict[str, List[int]] = {}
    for index, chunk in enumerate(chunk_records):
        chunk_id = str(chunk["chunk_id"])
        doc_id = str(chunk["doc_id"])
        chunk_ids.append(chunk_id)
        chunk_id_to_index[chunk_id] = index
        doc_to_indices.setdefault(doc_id, []).append(index)
    return chunk_ids, chunk_id_to_index, doc_to_indices


def iter_flatten_chunks(
    chunks_by_doc: Dict[str, Sequence[Dict[str, object]]],
    ordered_doc_ids: Iterable[str],
) -> List[Dict[str, object]]:
    flattened = []
    for doc_id in ordered_doc_ids:
        flattened.extend(chunks_by_doc.get(doc_id, []))
    return flattened
