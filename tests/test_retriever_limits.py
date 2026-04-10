import torch

from chunked_pooling.experiment_retrievers import DenseRetriever


def test_effective_max_tokens_per_forward_is_clamped_to_model_context_window():
    retriever = DenseRetriever(
        name="contriever",
        model_name="facebook/contriever",
        tokenizer_name="facebook/contriever",
        normalize=True,
        distance_metric="cosine",
        model=None,
        tokenizer=None,
        has_instructions=False,
        query_instruction="",
        document_instruction="",
        use_builtin_query_encoder=False,
        device=torch.device("cpu"),
        max_length=512,
        model_context_window=512,
        pooling="mean",
    )

    assert retriever.effective_max_tokens_per_forward(8192) == 512
    assert retriever.effective_max_tokens_per_forward(256) == 256
    assert retriever.effective_max_tokens_per_forward(None) == 512
