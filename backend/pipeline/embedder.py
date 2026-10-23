"""
backend/pipeline/embedder.py
============================
Embedding pipeline for ARIA — converts ticket text to vector
embeddings for semantic similarity search in ChromaDB.

WHY THIS FILE EXISTS:
The RAG retrieval component needs to find the most semantically
similar resolved tickets when a new ticket arrives. To do this,
all tickets (both incoming and resolved) must be converted to
the same embedding space — a list of 384 numbers that captures
the meaning of the text. Tickets with similar meanings will have
similar embeddings and can be found via cosine similarity search.

This module is a shared utility used by:
- backend/ingestion/pipeline.py (embeds resolved tickets at setup)
- backend/pipeline/retriever.py (embeds incoming tickets at query time)

Both must use the same model so embeddings are comparable.
"""

import os
import logging
from typing import Union
import numpy as np

# sentence-transformers: HuggingFace library specifically designed
# for semantic similarity tasks. Produces embeddings optimized for
# cosine similarity comparison — unlike general BERT embeddings
# which are not optimized for this use case.
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Model name — configurable via environment variable so we can swap models
# without changing code. Default is all-MiniLM-L6-v2:
# - 384-dimensional output vectors — compact and fast
# - Specifically trained for semantic similarity tasks
# - Runs on CPU in ~50ms per sentence — fast enough for real-time use
# - Free, local, no API calls, no rate limits
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "all-MiniLM-L6-v2"
)

# Embedding dimension — must match the model above
# all-MiniLM-L6-v2 always produces 384-dimensional vectors
EMBEDDING_DIMENSION = 384

# ── MODEL SINGLETON ───────────────────────────────────────────────────────────
# We use a module-level singleton so the model loads from disk only once
# when the FastAPI application starts, not on every embedding request.
# Loading the model takes ~2 seconds — doing it per-request would be
# unacceptably slow.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Returns the singleton SentenceTransformer model instance.

    Loads the model from HuggingFace on first call (downloads ~90MB
    and caches locally). Subsequent calls return the cached instance
    instantly.

    WHY SINGLETON: SentenceTransformer models are large objects.
    Creating a new instance for every embedding request would be
    extremely slow. The singleton pattern ensures we pay the loading
    cost once at startup.

    Returns:
        SentenceTransformer: the loaded embedding model
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(
            f"Embedding model loaded. "
            f"Dimension: {EMBEDDING_DIMENSION}"
        )
    return _model


def embed_text(text: str) -> list[float]:
    """
    Converts a single text string to a 384-dimensional embedding vector.

    WHY THIS FUNCTION EXISTS: The RAG retriever needs to embed incoming
    ticket text at query time to find similar resolved tickets. This
    function handles the single-text case.

    The returned embedding is a plain Python list (not a numpy array)
    because ChromaDB and FastAPI both serialize lists more easily than
    numpy arrays.

    Args:
        text: the ticket text to embed. Should be the raw ticket text
              as the user wrote it — no preprocessing needed.

    Returns:
        list[float]: 384-dimensional embedding vector

    Raises:
        ValueError: if text is empty or None
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty or whitespace-only text")

    model    = get_model()
    # encode() returns a numpy array — we convert to list for JSON serialization
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Converts a list of texts to embedding vectors in batches.

    WHY BATCH EMBEDDING: When ingesting thousands of resolved tickets
    into ChromaDB at startup, embedding one at a time would be slow.
    sentence-transformers' encode() is optimized for batch processing —
    it processes batch_size texts in parallel on the CPU.

    Args:
        texts: list of text strings to embed
        batch_size: how many texts to process at once.
                    64 is a good default for CPU — balances
                    speed and memory usage.

    Returns:
        list[list[float]]: list of 384-dimensional embedding vectors,
                           one per input text, in the same order

    Raises:
        ValueError: if texts list is empty
    """
    if not texts:
        raise ValueError("Cannot embed empty list of texts")

    # Filter out any empty strings and track their positions
    # so we can return a zero vector for them rather than crashing
    valid_texts  = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)
        else:
            logger.warning(f"Empty text at index {i} — will use zero vector")

    model = get_model()

    # encode() with batch_size processes texts in batches internally
    # show_progress_bar=True displays a tqdm bar for large batches
    embeddings_array = model.encode(
        valid_texts,
        batch_size        = batch_size,
        show_progress_bar = len(valid_texts) > 100,
        convert_to_numpy  = True,
    )

    # Build output list — insert zero vectors for empty inputs
    result = [None] * len(texts)
    for out_idx, orig_idx in enumerate(valid_indices):
        result[orig_idx] = embeddings_array[out_idx].tolist()

    # Fill any remaining None positions (from empty texts) with zero vectors
    for i in range(len(result)):
        if result[i] is None:
            result[i] = [0.0] * EMBEDDING_DIMENSION

    return result


def get_embedding_dimension() -> int:
    """
    Returns the dimension of embeddings produced by the current model.

    WHY THIS EXISTS: ChromaDB needs to know the embedding dimension when
    creating a collection. Rather than hardcoding 384 in multiple places,
    all components call this function so changing the model only requires
    updating EMBEDDING_MODEL_NAME.

    Returns:
        int: embedding dimension (384 for all-MiniLM-L6-v2)
    """
    return EMBEDDING_DIMENSION


def health_check() -> dict:
    """
    Verifies the embedding model is loaded and working correctly.

    WHY THIS EXISTS: FastAPI's /health endpoint calls this to confirm
    the embedding pipeline is operational before the application
    accepts traffic. Returns a dict with status information.

    Returns:
        dict: health status with model name, dimension, and test result
    """
    try:
        # Embed a simple test sentence to verify the model works end to end
        test_embedding = embed_text("test sentence for health check")

        return {
            "status":           "healthy",
            "model":            EMBEDDING_MODEL_NAME,
            "dimension":        EMBEDDING_DIMENSION,
            "test_embed_dim":   len(test_embedding),
            "dimension_match":  len(test_embedding) == EMBEDDING_DIMENSION,
        }
    except Exception as e:
        logger.error(f"Embedding health check failed: {e}")
        return {
            "status":  "unhealthy",
            "model":   EMBEDDING_MODEL_NAME,
            "error":   str(e),
        }
