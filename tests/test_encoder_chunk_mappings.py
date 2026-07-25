from __future__ import annotations

import re

from chunked_pooling.experiment_chunking import build_encoder_chunk_mappings


class SpecialTokenTokenizer:
    def __init__(self, prefix_token_count: int, suffix_token_count: int):
        self.prefix_token_count = prefix_token_count
        self.suffix_token_count = suffix_token_count

    def __call__(
        self,
        text,
        return_offsets_mapping=False,
        add_special_tokens=True,
        **_,
    ):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        token_offsets = [(match.start(), match.end()) for match in matches]
        input_ids = list(range(len(matches)))

        if add_special_tokens:
            token_offsets = (
                [(0, 0)] * self.prefix_token_count
                + token_offsets
                + [(0, 0)] * self.suffix_token_count
            )
            input_ids = (
                [-1] * self.prefix_token_count
                + input_ids
                + [-2] * self.suffix_token_count
            )

        payload = {"input_ids": input_ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = token_offsets
        return payload


def _chunk(
    chunk_id: str,
    chunk_index: int,
    char_start: int,
    char_end: int,
    token_start: int,
    token_end: int,
):
    return {
        "doc_id": "doc",
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "raw_text": "",
        "char_start": char_start,
        "char_end": char_end,
        "token_start": token_start,
        "token_end": token_end,
    }


def test_qwen_eos_only_mapping_keeps_last_token_spans_in_bounds():
    text = "alpha beta gamma delta"
    chunks = [
        _chunk("first", 0, 0, 10, 0, 2),
        _chunk("second", 1, 11, 22, 2, 4),
    ]
    tokenizer = SpecialTokenTokenizer(prefix_token_count=0, suffix_token_count=1)

    spans, metadata, encoder_token_count = build_encoder_chunk_mappings(
        chunk_records=chunks,
        text=text,
        tokenizer=tokenizer,
        instruction_text="",
    )

    assert encoder_token_count == 4
    assert spans == [(0, 2), (2, 5)]
    assert [end - 1 for _, end in spans] == [1, 4]
    assert all(end <= 5 for _, end in spans)
    assert [
        (row["encoder_model_token_start"], row["encoder_model_token_end"])
        for row in metadata
    ] == spans


def test_bert_mapping_uses_actual_prefix_instruction_and_suffix_offsets():
    text = "alpha beta gamma delta"
    instruction = "Represent: "
    chunks = [
        _chunk("first", 0, 0, 10, 0, 2),
        _chunk("second", 1, 11, 22, 2, 4),
    ]
    tokenizer = SpecialTokenTokenizer(prefix_token_count=1, suffix_token_count=1)

    spans, _, encoder_token_count = build_encoder_chunk_mappings(
        chunk_records=chunks,
        text=text,
        tokenizer=tokenizer,
        instruction_token_count=2,
        instruction_text=instruction,
    )

    assert encoder_token_count == 4
    assert spans == [(0, 5), (5, 8)]
