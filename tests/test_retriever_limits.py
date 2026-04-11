import numpy as np
import torch

from chunked_pooling.experiment_retrievers import DenseRetriever, _to_numpy


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


def test_to_numpy_upcasts_bfloat16_tensors_before_numpy_conversion():
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    converted = _to_numpy(tensor)

    assert isinstance(converted, np.ndarray)
    assert converted.dtype == np.float32
    assert np.allclose(converted, np.array([[1.0, 2.0]], dtype=np.float32))
