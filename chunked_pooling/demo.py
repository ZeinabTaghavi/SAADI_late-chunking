from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from chunked_pooling import chunked_pooling

CHUNKING_STRATEGIES = ("semantic", "fixed", "sentences")
DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
DEFAULT_DOCUMENT = (
    "Berlin is the capital and largest city of Germany, both by area and by "
    "population. Its more than 3.85 million inhabitants make it the European "
    "Union's most populous city, as measured by population within city limits. "
    "The city is also one of the states of Germany, and is the third smallest "
    "state in the country in terms of area."
)
DEFAULT_QUESTION = "What is the population of Berlin?"
DEFAULT_EXPECTED_ANSWER = "3.85 million inhabitants"
DEFAULT_TOP_K = 3


@dataclass
class RankedChunk:
    chunk_index: int
    text: str
    score: float
    contains_expected_answer: bool


def resolve_max_length(tokenizer, fallback: int = 8192) -> int:
    model_max_length = getattr(tokenizer, "model_max_length", fallback)
    if model_max_length is None or model_max_length > 100_000:
        return fallback
    return min(fallback, int(model_max_length))


def to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = to_numpy(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


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


def offsets_to_text_chunks(
    text: str,
    token_offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int]],
) -> List[str]:
    chunks = []
    for start, end in spans:
        if start >= end:
            continue
        chunk_start = token_offsets[start][0]
        chunk_end = token_offsets[end - 1][1]
        chunks.append(text[chunk_start:chunk_end].strip())
    return chunks


def chunk_text(
    document: str,
    tokenizer,
    chunking_strategy: str,
    chunk_size: int,
    n_sentences: int,
    embedding_model_name: str,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    from chunked_pooling.chunking import Chunker

    chunker = Chunker(chunking_strategy=chunking_strategy)
    chunk_kwargs = {}
    if chunking_strategy == "fixed":
        chunk_kwargs["chunk_size"] = chunk_size
    elif chunking_strategy == "sentences":
        chunk_kwargs["n_sentences"] = n_sentences
    elif chunking_strategy == "semantic":
        chunk_kwargs["embedding_model_name"] = embedding_model_name

    spans = chunker.chunk(
        document,
        tokenizer=tokenizer,
        chunking_strategy=chunking_strategy,
        **chunk_kwargs,
    )
    tokenization = tokenizer(
        document,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    chunks = offsets_to_text_chunks(
        text=document,
        token_offsets=tokenization["offset_mapping"],
        spans=spans,
    )
    return chunks, spans


def encode_query(model, question: str) -> np.ndarray:
    if hasattr(model, "encode_queries"):
        embedding = model.encode_queries([question])[0]
    elif hasattr(model, "encode"):
        embedding = model.encode([question])[0]
    else:
        raise AttributeError(
            "The loaded model does not expose encode_queries() or encode()."
        )
    return to_numpy(embedding)


def encode_traditional_chunks(model, chunks: Sequence[str]) -> np.ndarray:
    if hasattr(model, "encode_corpus"):
        embeddings = model.encode_corpus(list(chunks))
    elif hasattr(model, "encode"):
        embeddings = model.encode(list(chunks))
    else:
        raise AttributeError(
            "The loaded model does not expose encode_corpus() or encode()."
        )
    embeddings = to_numpy(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings


def load_demo_model(model_name: str):
    try:
        from chunked_pooling.wrappers import load_model

        return load_model(model_name)
    except ModuleNotFoundError:
        from transformers import AutoModel

        return AutoModel.from_pretrained(model_name, trust_remote_code=True), False


def encode_late_chunks(
    model,
    tokenizer,
    document: str,
    spans: Sequence[Tuple[int, int]],
    model_has_instructions: bool,
    max_length: int,
) -> np.ndarray:
    instruction = ""
    n_instruction_tokens = 0
    if model_has_instructions:
        instruction = model.get_instructions()[1]
        n_instruction_tokens = len(
            tokenizer(instruction, add_special_tokens=False)["input_ids"]
        )

    span_annotations = extend_special_tokens(
        spans, n_instruction_tokens=n_instruction_tokens
    )
    model_inputs = tokenizer(
        instruction + document,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    device = getattr(model, "device", None)
    if device is not None:
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    with torch.no_grad():
        model_outputs = model(**model_inputs)

    embeddings = chunked_pooling(
        model_outputs,
        [span_annotations],
        max_length=max_length,
    )[0]
    return np.vstack([to_numpy(embedding) for embedding in embeddings])


def rank_chunks(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_texts: Sequence[str],
    expected_answer: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> List[RankedChunk]:
    normalized_query = normalize_rows(query_embedding)[0]
    normalized_chunks = normalize_rows(chunk_embeddings)
    scores = normalized_chunks @ normalized_query
    sorted_indices = np.argsort(-scores)

    expected_answer_lower = expected_answer.strip().lower()
    results = []
    for chunk_index in sorted_indices[: min(top_k, len(chunk_texts))]:
        text = chunk_texts[int(chunk_index)]
        contains_expected_answer = bool(expected_answer_lower) and (
            expected_answer_lower in text.lower()
        )
        results.append(
            RankedChunk(
                chunk_index=int(chunk_index),
                text=text,
                score=float(scores[int(chunk_index)]),
                contains_expected_answer=contains_expected_answer,
            )
        )
    return results


def run_demo(
    document: str = DEFAULT_DOCUMENT,
    question: str = DEFAULT_QUESTION,
    expected_answer: str = DEFAULT_EXPECTED_ANSWER,
    model_name: str = DEFAULT_MODEL_NAME,
    chunking_strategy: str = "sentences",
    chunk_size: int = 64,
    n_sentences: int = 1,
    top_k: int = DEFAULT_TOP_K,
):
    from transformers import AutoTokenizer

    model, has_instructions = load_demo_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    max_length = resolve_max_length(tokenizer)

    chunk_texts, spans = chunk_text(
        document=document,
        tokenizer=tokenizer,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        n_sentences=n_sentences,
        embedding_model_name=model_name,
    )
    if not chunk_texts:
        raise ValueError("No chunks were produced from the provided document.")

    query_embedding = encode_query(model, question)
    traditional_embeddings = encode_traditional_chunks(model, chunk_texts)
    late_embeddings = encode_late_chunks(
        model=model,
        tokenizer=tokenizer,
        document=document,
        spans=spans,
        model_has_instructions=has_instructions,
        max_length=max_length,
    )

    return {
        "model_name": model_name,
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "n_sentences": n_sentences,
        "document": document,
        "question": question,
        "expected_answer": expected_answer,
        "chunks": chunk_texts,
        "traditional_results": rank_chunks(
            query_embedding=query_embedding,
            chunk_embeddings=traditional_embeddings,
            chunk_texts=chunk_texts,
            expected_answer=expected_answer,
            top_k=top_k,
        ),
        "late_results": rank_chunks(
            query_embedding=query_embedding,
            chunk_embeddings=late_embeddings,
            chunk_texts=chunk_texts,
            expected_answer=expected_answer,
            top_k=top_k,
        ),
    }


def print_results(results) -> None:
    print(f"Model: {results['model_name']}")
    print(f"Chunking strategy: {results['chunking_strategy']}")
    if results["chunking_strategy"] == "fixed":
        print(f"Chunk size: {results['chunk_size']} tokens")
    if results["chunking_strategy"] == "sentences":
        print(f"Sentences per chunk: {results['n_sentences']}")

    print("\nDocument:")
    print(results["document"])

    print("\nQuestion:")
    print(results["question"])

    print("\nExpected answer:")
    print(results["expected_answer"])

    print("\nChunks:")
    for index, chunk in enumerate(results["chunks"]):
        print(f"[{index}] {chunk}")

    print("\nTraditional chunking ranking:")
    for ranked_chunk in results["traditional_results"]:
        marker = " <- contains expected answer" if ranked_chunk.contains_expected_answer else ""
        print(
            f"[{ranked_chunk.chunk_index}] score={ranked_chunk.score:.4f}{marker}\n"
            f"{ranked_chunk.text}"
        )

    print("\nLate chunking ranking:")
    for ranked_chunk in results["late_results"]:
        marker = " <- contains expected answer" if ranked_chunk.contains_expected_answer else ""
        print(
            f"[{ranked_chunk.chunk_index}] score={ranked_chunk.score:.4f}{marker}\n"
            f"{ranked_chunk.text}"
        )

    late_hit = results["late_results"][0]
    print("\nTop late-chunking result:")
    print(late_hit.text)
    if late_hit.contains_expected_answer:
        print("Late chunking retrieved a chunk that contains the expected answer.")
    else:
        print(
            "Late chunking did not retrieve the expected answer in the top chunk. "
            "Try another model or chunking setup."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple single-document late chunking demo."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--expected-answer", default=DEFAULT_EXPECTED_ANSWER)
    parser.add_argument(
        "--chunking-strategy",
        default="sentences",
        choices=CHUNKING_STRATEGIES,
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--n-sentences", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    results = run_demo(
        document=args.document,
        question=args.question,
        expected_answer=args.expected_answer,
        model_name=args.model_name,
        chunking_strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        n_sentences=args.n_sentences,
        top_k=args.top_k,
    )
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
