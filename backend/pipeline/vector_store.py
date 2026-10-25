"""
backend/pipeline/vector_store.py
================================
ChromaDB vector store implementation for ARIA.

Manages the 'resolved_tickets' collection with cosine similarity distance,
custom embeddings from backend.pipeline.embedder, idempotent upserts,
and semantic search filtering for RAG retrieval.
"""

import os
import math
import json
import inspect
import logging
from pathlib import Path
from typing import Union
import numpy as np
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# ── ENVIRONMENT & EMBEDDER IMPORTS ────────────────────────────────────────────
# load_dotenv() MUST be called BEFORE importing backend.pipeline.embedder
# because embedder reads EMBEDDING_MODEL_NAME from the environment at import time.
load_dotenv()

from backend.pipeline.embedder import (
    EMBEDDING_MODEL_NAME as _DEFAULT_EMBEDDING_MODEL_NAME,
    embed_text,
    embed_batch,
    get_embedding_dimension,
)

logger = logging.getLogger(__name__)

# Reference EMBEDDING_MODEL_NAME as a module-level attribute so unit tests
# can patch it (e.g. testing model mismatch protection).
EMBEDDING_MODEL_NAME: str = _DEFAULT_EMBEDDING_MODEL_NAME

# ── CONSTANTS & GLOBAL CONFIGURATION ──────────────────────────────────────────
# Single collection name per fixed design decision 1 (sap_docs dropped)
COLLECTION_NAME = "resolved_tickets"

# PROJECT_ROOT resolved against backend/pipeline/vector_store.py (parents[2] -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Constant placeholder for missing classifications
UNCLASSIFIED = "UNKNOWN"

# Default batch size for upserting records into ChromaDB
UPSERT_BATCH_SIZE = 500

# Cache of PersistentClient instances keyed by resolved persist Path
_clients: dict[Path, chromadb.PersistentClient] = {}


# ── CLIENT & PERSISTENCE HELPERS ──────────────────────────────────────────────
def resolve_persist_path() -> Path:
    """
    Resolves the ChromaDB persistence path from CHROMA_PERSIST_PATH.

    Per Decision 7:
    - If relative, resolves against PROJECT_ROOT (never cwd).
    - If absolute, uses as-is.
    - Ensures the directory exists.

    Returns:
        Path: Absolute path to the ChromaDB directory.
    """
    raw_path = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_db")
    path_obj = Path(raw_path)
    if not path_obj.is_absolute():
        resolved = (PROJECT_ROOT / path_obj).resolve()
    else:
        resolved = path_obj.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_client() -> chromadb.PersistentClient:
    """
    Returns a cached PersistentClient instance for the resolved persist path.

    Re-reads CHROMA_PERSIST_PATH on every call so tests can point to temporary
    directories via monkeypatch.

    Returns:
        chromadb.PersistentClient: Initialized client with telemetry disabled.
    """
    persist_path = resolve_persist_path()
    if persist_path not in _clients:
        logger.info(f"Initializing ChromaDB PersistentClient at {persist_path}")
        settings = Settings(anonymized_telemetry=False)
        _clients[persist_path] = chromadb.PersistentClient(
            path=str(persist_path),
            settings=settings,
        )
    return _clients[persist_path]


def _extract_distance_space(collection) -> str:
    """
    Extracts the distance metric (space) from the Chroma collection across versions.

    Args:
        collection: The Chroma collection object.

    Returns:
        str: 'cosine', 'l2', 'ip', or 'unknown'.
    """
    meta = collection.metadata or {}
    if "hnsw:space" in meta:
        return str(meta["hnsw:space"])

    # Newer Chroma 1.x stores configuration inside collection.configuration
    config = getattr(collection, "configuration", None)
    if config is not None:
        if isinstance(config, dict):
            hnsw_conf = config.get("hnsw", {})
            if isinstance(hnsw_conf, dict) and "space" in hnsw_conf:
                return str(hnsw_conf["space"])
        elif hasattr(config, "hnsw") and hasattr(config.hnsw, "space"):
            return str(config.hnsw.space)

    return "l2"


# ── COLLECTION ACCESS & VALIDATION ────────────────────────────────────────────
def get_collection():
    """
    Retrieves or initializes the 'resolved_tickets' collection with cosine space.

    At creation:
      - Sets cosine distance metric.
      - Stores embedding_model and embedding_dimension in metadata.
    When opening an existing collection:
      - Validates that distance metric is cosine.
      - Validates that stored embedding_model matches EMBEDDING_MODEL_NAME.
      - Validates that stored dimension matches get_embedding_dimension().
      - On mismatch, raises RuntimeError with reset instructions.

    Returns:
        chromadb.Collection: The verified collection instance.

    Raises:
        RuntimeError: If distance metric, model, or dimension mismatches.
    """
    client = get_client()

    try:
        collection = client.get_collection(COLLECTION_NAME)
        is_new = False
    except Exception:
        is_new = True

    if is_new:
        logger.info(f"Creating collection '{COLLECTION_NAME}' with cosine space")
        initial_meta = {
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": get_embedding_dimension(),
        }

        # Check signature support for 'configuration' parameter in chromadb 1.x
        sig = inspect.signature(client.get_or_create_collection)
        kwargs = {
            "name": COLLECTION_NAME,
            "metadata": initial_meta,
        }
        if "configuration" in sig.parameters:
            kwargs["configuration"] = {"hnsw": {"space": "cosine"}}

        collection = client.get_or_create_collection(**kwargs)
        return collection

    # Existing collection: validate configuration & metadata integrity
    meta = collection.metadata or {}
    stored_space = _extract_distance_space(collection)
    stored_model = meta.get("embedding_model")
    stored_dim = meta.get("embedding_dimension")

    reset_cmd = "Run: python -m backend.ingestion.pipeline reset --yes"

    if stored_space != "cosine":
        raise RuntimeError(
            f"Distance space mismatch in '{COLLECTION_NAME}': found '{stored_space}', "
            f"expected 'cosine'. {reset_cmd}"
        )

    if stored_model != EMBEDDING_MODEL_NAME:
        raise RuntimeError(
            f"Embedding model mismatch in '{COLLECTION_NAME}': collection built with "
            f"'{stored_model}', current model is '{EMBEDDING_MODEL_NAME}'. {reset_cmd}"
        )

    expected_dim = get_embedding_dimension()
    if stored_dim is not None and int(stored_dim) != expected_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch in '{COLLECTION_NAME}': collection built with "
            f"{stored_dim}, current model produces {expected_dim}. {reset_cmd}"
        )

    return collection


# ── METADATA UTILITIES ────────────────────────────────────────────────────────
def clean_metadata(metadata: dict) -> dict:
    """
    Sanitizes metadata dictionary for ChromaDB compatibility (Decision 9).

    Ensures all values are scalars (str, int, float, bool). Converts None and NaN
    to "". Lists and dicts are flattened or serialized.

    Args:
        metadata: Raw dictionary of metadata fields.

    Returns:
        dict: Sanitized metadata dictionary.
    """
    cleaned = {}
    for k, v in metadata.items():
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, float) and (np.isnan(v) or math.isnan(v)):
            cleaned[k] = ""
        elif isinstance(v, (str, int, bool)):
            cleaned[k] = v
        elif isinstance(v, float):
            cleaned[k] = v
        elif isinstance(v, (list, tuple)):
            cleaned[k] = ", ".join(str(item) for item in v)
        elif isinstance(v, dict):
            cleaned[k] = json.dumps(v)
        else:
            cleaned[k] = str(v)
    return cleaned


def build_ticket_metadata(record: dict) -> dict:
    """
    Builds canonical ticket metadata dictionary from a raw ticket record.

    Ensures identical keys and types across CSV ingestion and direct ticket closing:
    ticket_id, module, issue_type, error_code, root_cause, fix, tcodes_used,
    resolution_time_minutes (int, -1 if unknown), escalated, created_date,
    resolved_date, ticket_type, source_type="resolved_ticket".

    Args:
        record: Raw ticket dictionary.

    Returns:
        dict: Cleaned, scalar metadata dictionary.
    """
    # Parse resolution_time_minutes as integer (-1 if unparseable or unknown)
    raw_res_time = record.get("resolution_time_minutes")
    res_time = -1
    if raw_res_time is not None and str(raw_res_time).strip() != "":
        try:
            res_time = int(float(str(raw_res_time).strip()))
        except (ValueError, TypeError):
            res_time = -1

    raw_meta = {
        "ticket_id": str(record.get("ticket_id") or "").strip(),
        "module": str(record.get("module") or "").strip(),
        "issue_type": str(record.get("issue_type") or "").strip(),
        "error_code": str(record.get("error_code") or "").strip(),
        "root_cause": str(record.get("root_cause") or "").strip(),
        "fix": str(record.get("fix") or "").strip(),
        "tcodes_used": str(record.get("tcodes_used") or "").strip(),
        "resolution_time_minutes": res_time,
        "escalated": str(record.get("escalated") or "").strip(),
        "created_date": str(record.get("created_date") or "").strip(),
        "resolved_date": str(record.get("resolved_date") or "").strip(),
        "ticket_type": str(record.get("ticket_type") or "").strip(),
        "source_type": "resolved_ticket",
    }
    return clean_metadata(raw_meta)


# ── UPSERT & INGESTION CORE ───────────────────────────────────────────────────
def upsert_records(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    batch_size: int = UPSERT_BATCH_SIZE,
    skip_existing: bool = True,
) -> dict:
    """
    Upserts document records into ChromaDB with deterministic IDs and batching.

    Validates inputs, skips empty texts, de-duplicates IDs keeping the last occurrence,
    checks for existing records if skip_existing is True, generates embeddings
    using embedder.py, and executes idempotent batch upserts.

    Args:
        ids: List of unique document identifiers.
        texts: List of problem/chunk texts to embed.
        metadatas: List of metadata dictionaries.
        batch_size: Processing batch size for embedding and upserting.
        skip_existing: Whether to skip IDs already present in ChromaDB.

    Returns:
        dict: Ingestion summary containing received, skipped_empty, duplicate_ids,
              skipped_existing, and upserted counts.

    Raises:
        ValueError: If input list lengths do not match.
        RuntimeError: If embedding dimension does not match get_embedding_dimension().
    """
    if not (len(ids) == len(texts) == len(metadatas)):
        raise ValueError(
            f"Input length mismatch: ids ({len(ids)}), texts ({len(texts)}), "
            f"metadatas ({len(metadatas)}) must have equal length."
        )

    received = len(ids)

    # Filter out empty or whitespace-only texts (Decision 8: Never store zero vectors)
    non_empty_ids = []
    non_empty_texts = []
    non_empty_metas = []
    skipped_empty = 0

    for i_id, txt, meta in zip(ids, texts, metadatas):
        clean_text = str(txt or "").strip()
        if not clean_text:
            skipped_empty += 1
            continue
        non_empty_ids.append(str(i_id).strip())
        non_empty_texts.append(clean_text)
        non_empty_metas.append(clean_metadata(meta))

    # De-duplicate IDs within this call, keeping the LAST occurrence
    last_seen_index = {}
    for idx, i_id in enumerate(non_empty_ids):
        last_seen_index[i_id] = idx

    duplicate_ids = len(non_empty_ids) - len(last_seen_index)
    chosen_indices = sorted(last_seen_index.values())

    dedup_ids = [non_empty_ids[i] for i in chosen_indices]
    dedup_texts = [non_empty_texts[i] for i in chosen_indices]
    dedup_metas = [non_empty_metas[i] for i in chosen_indices]

    collection = get_collection()
    skipped_existing = 0
    upserted = 0
    first_batch_validated = False

    for start_idx in range(0, len(dedup_ids), batch_size):
        b_ids = dedup_ids[start_idx : start_idx + batch_size]
        b_texts = dedup_texts[start_idx : start_idx + batch_size]
        b_metas = dedup_metas[start_idx : start_idx + batch_size]

        if skip_existing:
            # Check which IDs already exist in ChromaDB
            existing_res = collection.get(ids=b_ids, include=[])
            existing_set = set(existing_res.get("ids", []))

            if existing_set:
                kept_ids, kept_texts, kept_metas = [], [], []
                for i_id, txt, meta in zip(b_ids, b_texts, b_metas):
                    if i_id in existing_set:
                        skipped_existing += 1
                    else:
                        kept_ids.append(i_id)
                        kept_texts.append(txt)
                        kept_metas.append(meta)
                b_ids, b_texts, b_metas = kept_ids, kept_texts, kept_metas

        if not b_ids:
            continue

        # Generate embeddings using embedder.py (never let Chroma generate them)
        b_embeddings = embed_batch(b_texts)

        # Validate embedding vector dimension on the first processed batch
        if not first_batch_validated and len(b_embeddings) > 0:
            dim = len(b_embeddings[0])
            expected_dim = get_embedding_dimension()
            if dim != expected_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: generated {dim}, "
                    f"expected {expected_dim}"
                )
            first_batch_validated = True

        collection.upsert(
            ids=b_ids,
            embeddings=b_embeddings,
            documents=b_texts,
            metadatas=b_metas,
        )
        upserted += len(b_ids)

    return {
        "received": received,
        "skipped_empty": skipped_empty,
        "duplicate_ids": duplicate_ids,
        "skipped_existing": skipped_existing,
        "upserted": upserted,
    }


def add_resolved_ticket(record: dict) -> bool:
    """
    Ingests or overwrites a single resolved ticket in ChromaDB upon ticket closure.

    Requires ticket_id and non-blank problem text. Uses skip_existing=False so
    re-closing the same ticket overwrites previous metadata.

    Args:
        record: Dictionary representing the resolved ticket.

    Returns:
        bool: True if upserted successfully, False if problem text is blank or missing.
    """
    ticket_id = str(record.get("ticket_id") or "").strip()
    problem = str(record.get("problem") or "").strip()

    if not ticket_id or not problem:
        logger.warning(f"add_resolved_ticket failed: blank ticket_id or problem text.")
        return False

    meta = build_ticket_metadata(record)
    summary = upsert_records(
        ids=[ticket_id],
        texts=[problem],
        metadatas=[meta],
        skip_existing=False,
    )
    return summary["upserted"] == 1


# ── DELETION & RETRIEVAL (SEARCH) ─────────────────────────────────────────────
def delete_by_source_file(source_file: str) -> int:
    """
    Deletes all records matching a specific source_file metadata attribute.

    Used when re-ingesting a documentation file to prevent stale chunks.

    Args:
        source_file: Relative source file path identifier.

    Returns:
        int: Number of deleted records.
    """
    collection = get_collection()
    res = collection.get(where={"source_file": source_file}, include=[])
    matching_ids = res.get("ids", [])
    if matching_ids:
        collection.delete(ids=matching_ids)
        logger.info(f"Deleted {len(matching_ids)} records for source_file='{source_file}'")
    return len(matching_ids)


def search_similar(
    query_text: str,
    n_results: int = 5,
    module_filter: str | None = None,
    issue_type_filter: str | None = None,
    include_unclassified: bool = True,
) -> list[dict]:
    """
    Performs semantic similarity search against resolved tickets using cosine distance.

    Args:
        query_text: Query text to embed and search.
        n_results: Maximum number of nearest neighbors to return.
        module_filter: Optional module filter (e.g. 'FI', 'MM').
        issue_type_filter: Optional issue type filter.
        include_unclassified: If True, matches UNKNOWN/unclassified records as well.

    Returns:
        list[dict]: Results sorted best first with id, text, distance, similarity, metadata.
    """
    clean_query = str(query_text or "").strip()
    if not clean_query:
        logger.warning("Blank query passed to search_similar; returning empty list.")
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    effective_n = min(n_results, count)
    if effective_n <= 0:
        return []

    # Construct metadata filtering conditions
    conditions = []
    if module_filter:
        if include_unclassified:
            conditions.append({"module": {"$in": [module_filter, UNCLASSIFIED]}})
        else:
            conditions.append({"module": {"$in": [module_filter]}})

    if issue_type_filter:
        if include_unclassified:
            conditions.append({"issue_type": {"$in": [issue_type_filter, UNCLASSIFIED]}})
        else:
            conditions.append({"issue_type": {"$in": [issue_type_filter]}})

    if not conditions:
        where_clause = None
    elif len(conditions) == 1:
        where_clause = conditions[0]
    else:
        where_clause = {"$and": conditions}

    # Embed query text with embedder
    query_vector = embed_text(clean_query)

    query_res = collection.query(
        query_embeddings=[query_vector],
        n_results=effective_n,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    ids = query_res.get("ids", [[]])[0]
    docs = query_res.get("documents", [[]])[0]
    dists = query_res.get("distances", [[]])[0]
    metas = query_res.get("metadatas", [[]])[0]

    items = []
    for r_id, doc, dist, meta in zip(ids, docs, dists, metas):
        d_val = float(dist)
        items.append({
            "id": r_id,
            "text": doc,
            "distance": d_val,
            "similarity": 1.0 - d_val,
            "metadata": meta,
        })

    items.sort(key=lambda x: x["distance"])
    return items


# ── STATS & HEALTH CHECK ──────────────────────────────────────────────────────
def get_collection_stats(detailed: bool = False) -> dict:
    """
    Computes summary and diagnostic statistics for the vector store.

    Args:
        detailed: If True, pages through metadata to count items per module and source_type.

    Returns:
        dict: Collection stats including counts, path, model, and distance space.
    """
    collection = get_collection()
    count = collection.count()
    meta = collection.metadata or {}
    distance_space = _extract_distance_space(collection)

    stats = {
        "count": count,
        "persist_path": str(resolve_persist_path()),
        "collection_name": COLLECTION_NAME,
        "embedding_model": meta.get("embedding_model", EMBEDDING_MODEL_NAME),
        "embedding_dimension": meta.get("embedding_dimension", get_embedding_dimension()),
        "distance_space": distance_space,
    }

    if detailed:
        module_counts: dict[str, int] = {}
        source_type_counts: dict[str, int] = {}

        limit = 5000
        offset = 0
        while offset < count:
            batch = collection.get(limit=limit, offset=offset, include=["metadatas"])
            metas = batch.get("metadatas", [])
            if not metas:
                break
            for m in metas:
                if m:
                    mod = m.get("module", UNCLASSIFIED)
                    st = m.get("source_type", "unknown")
                    module_counts[mod] = module_counts.get(mod, 0) + 1
                    source_type_counts[st] = source_type_counts.get(st, 0) + 1
            offset += len(batch.get("ids", []))

        stats["modules"] = module_counts
        stats["source_types"] = source_type_counts

    return stats


def reset_collection() -> None:
    """
    Deletes and recreates the collection with fresh cosine space and metadata.
    """
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted collection '{COLLECTION_NAME}'")
    except Exception as e:
        logger.warning(f"Could not delete collection '{COLLECTION_NAME}': {e}")

    # Re-initialize fresh collection
    get_collection()


def health_check() -> dict:
    """
    Evaluates vector store operational status for the /health diagnostic API.

    Returns:
        dict: Health status containing status ('healthy'/'unhealthy'), count, space, and error.
    """
    try:
        collection = get_collection()
        count = collection.count()
        space = _extract_distance_space(collection)
        return {
            "status": "healthy",
            "count": count,
            "space": space,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        return {
            "status": "unhealthy",
            "count": 0,
            "space": None,
            "error": str(e),
        }
