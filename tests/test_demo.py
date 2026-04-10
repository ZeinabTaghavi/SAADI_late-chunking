import numpy as np

from chunked_pooling.demo import (
    extend_special_tokens,
    offsets_to_text_chunks,
    rank_chunks,
)


def test_extend_special_tokens_matches_expected_offsets():
    annotations = [(0, 5), (5, 10)]
    extended = extend_special_tokens(annotations, n_instruction_tokens=2)
    assert extended == [(0, 8), (8, 14)]


def test_offsets_to_text_chunks_reconstructs_text_segments():
    text = "Berlin is big. It has many inhabitants."
    offsets = [
        (0, 6),
        (7, 9),
        (10, 13),
        (13, 14),
        (15, 17),
        (18, 21),
        (22, 26),
        (27, 38),
        (38, 39),
    ]
    spans = [(0, 4), (4, 9)]
    chunks = offsets_to_text_chunks(text, offsets, spans)
    assert chunks == ["Berlin is big.", "It has many inhabitants."]


def test_rank_chunks_orders_by_similarity_and_marks_expected_answer():
    query_embedding = np.array([1.0, 0.0])
    chunk_embeddings = np.array([[0.9, 0.1], [0.1, 0.9]])
    chunk_texts = [
        "Berlin has 3.85 million inhabitants.",
        "Munich is in Bavaria.",
    ]
    ranked = rank_chunks(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunk_texts=chunk_texts,
        expected_answer="3.85 million inhabitants",
        top_k=2,
    )
    assert ranked[0].chunk_index == 0
    assert ranked[0].contains_expected_answer is True
    assert ranked[0].score > ranked[1].score
