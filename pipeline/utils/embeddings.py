"""Shared embedding helpers for pipeline stages.

Two embedding spaces are exposed: a *semantic* space (general sentence
embedder, groups text by what it talks about) used by Stage 2 and Stage 3 for
clustering, and a *preference* space (preference-aware embedder, groups text
by what opinion it expresses) used by Stage 5 for the GSC DISC representation
proxy.
"""

import logging
from time import monotonic

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from config import (
    PIPELINE_CPU_THREADS,
    PREFERENCE_EMBEDDING_DEVICE,
    PREFERENCE_EMBEDDING_MODEL,
    TOPIC_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_preference_embedding_model = None
_semantic_embedding_model = None
_torch_threads_configured = False


def configure_torch_threads():
    global _torch_threads_configured
    if _torch_threads_configured:
        return

    torch.set_num_threads(PIPELINE_CPU_THREADS)
    torch.set_num_interop_threads(max(1, min(2, PIPELINE_CPU_THREADS)))
    logger.info("Configured Torch to use up to %d CPU threads", PIPELINE_CPU_THREADS)
    _torch_threads_configured = True


def load_preference_embedder_model():
    logger.info(
        "Loading preference embedding model %s",
        PREFERENCE_EMBEDDING_MODEL,
    )
    start = monotonic()
    model = SentenceTransformer(
        PREFERENCE_EMBEDDING_MODEL,
        device=PREFERENCE_EMBEDDING_DEVICE,
    )
    logger.info("Loaded preference embedding model in %.1fs", monotonic() - start)
    return model


def preference_embedder():
    global _preference_embedding_model
    if _preference_embedding_model is None:
        configure_torch_threads()
        _preference_embedding_model = load_preference_embedder_model()
    return _preference_embedding_model


def embed_preference_texts(texts):
    start = monotonic()
    logger.debug("Embedding %d preference texts", len(texts))
    embeddings = np.asarray(
        preference_embedder().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    logger.debug(
        "Embedded %d preference texts in %.1fs",
        len(texts),
        monotonic() - start,
    )
    return embeddings


def semantic_embedder():
    global _semantic_embedding_model
    if _semantic_embedding_model is None:
        logger.info("Loading semantic embedding model %s", TOPIC_EMBEDDING_MODEL)
        _semantic_embedding_model = SentenceTransformer(TOPIC_EMBEDDING_MODEL)
    return _semantic_embedding_model


def embed_semantic_texts(texts):
    return np.asarray(
        semantic_embedder().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def claim_embedding_text(claim_text, topic_sentence):
    """Combine claim text and topic sentence so the topic signal isn't lost."""
    if topic_sentence:
        return f"{claim_text} | {topic_sentence}"
    return claim_text
