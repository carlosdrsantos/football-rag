"""The embedding model.

BGE models are trained asymmetrically: passages are embedded bare, queries are
embedded behind an instruction prefix. Skipping the prefix costs real recall,
so queries and documents go through separate functions rather than one encode().
"""

from functools import cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def dimensions() -> int:
    return _model().get_embedding_dimension()


def encode_documents(texts: list[str]) -> list[list[float]]:
    vectors = _model().encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]


def encode_queries(texts: list[str]) -> list[list[float]]:
    prefixed = [QUERY_PREFIX + t for t in texts]
    vectors = _model().encode(prefixed, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]
