"""
backend/tests/test_vector_store.py
==================================
Unit tests for ChromaDB vector store implementation (backend.pipeline.vector_store).
"""

import math
import numpy as np
import pytest


def test_empty_store(isolated_chroma):
    """
    Asserts empty store behavior: initial count is 0 and search_similar returns [].
    """
    from backend.pipeline import vector_store

    stats = vector_store.get_collection_stats()
    assert stats["count"] == 0

    results = vector_store.search_similar("Cannot post invoice", n_results=5)
    assert results == []


def test_add_resolved_ticket_round_trip(isolated_chroma):
    """
    Asserts add_resolved_ticket round-trips metadata including multi-line fix,
    comma-separated tcodes_used, and correct numeric types.
    """
    from backend.pipeline import vector_store

    record = {
        "ticket_id": "T-100",
        "module": "FI",
        "issue_type": "Configuration",
        "problem": "Payment run failed with error F5155 during execution",
        "error_code": "F5155",
        "root_cause": "Missing posting authorization for company code 1000",
        "fix": "Step 1: Open OB52.\nStep 2: Maintain posting period variants.\nStep 3: Save.",
        "tcodes_used": "F110, OB52, SU01",
        "resolution_time_minutes": 45,
        "escalated": "No",
        "created_date": "2026-08-01 10:00:00",
        "resolved_date": "2026-08-01 10:45:00",
        "ticket_type": "incident",
    }
    assert vector_store.add_resolved_ticket(record) is True

    results = vector_store.search_similar("Payment run failed F5155", n_results=1)
    assert len(results) == 1
    meta = results[0]["metadata"]

    assert meta["ticket_id"] == "T-100"
    assert meta["module"] == "FI"
    assert meta["issue_type"] == "Configuration"
    assert meta["error_code"] == "F5155"
    assert meta["fix"] == record["fix"]
    assert meta["tcodes_used"] == "F110, OB52, SU01"
    assert meta["resolution_time_minutes"] == 45
    assert isinstance(meta["resolution_time_minutes"], int)
    assert meta["escalated"] == "No"
    assert meta["source_type"] == "resolved_ticket"


def test_idempotency_and_overwrite(isolated_chroma):
    """
    Asserts idempotency: upserting identical IDs leaves count unchanged,
    and add_resolved_ticket with edited fix overwrites existing data.
    """
    from backend.pipeline import vector_store

    record = {
        "ticket_id": "T-200",
        "problem": "Purchase order price variance exceeds tolerance",
        "module": "MM",
        "fix": "Initial fix instruction",
    }
    assert vector_store.add_resolved_ticket(record) is True
    assert vector_store.get_collection_stats()["count"] == 1

    # Ingesting same record again leaves count at 1
    assert vector_store.add_resolved_ticket(record) is True
    assert vector_store.get_collection_stats()["count"] == 1

    # Updating fix overwrites previous metadata
    record["fix"] = "Updated updated tolerance limits via OMRX"
    assert vector_store.add_resolved_ticket(record) is True
    assert vector_store.get_collection_stats()["count"] == 1

    res = vector_store.search_similar("tolerance price variance", n_results=1)
    assert res[0]["metadata"]["fix"] == "Updated updated tolerance limits via OMRX"


def test_blank_problem_handling(isolated_chroma):
    """
    Asserts blank problem returns False, leaves count at 0, and never stores zero vectors.
    """
    from backend.pipeline import vector_store

    blank_record = {
        "ticket_id": "T-EMPTY",
        "problem": "   ",
        "module": "FI",
    }
    assert vector_store.add_resolved_ticket(blank_record) is False
    assert vector_store.get_collection_stats()["count"] == 0


def test_semantic_search(isolated_chroma):
    """
    Stores 6 hand-written records across modules; asserts a paraphrased query
    ranks the semantically matched record first.
    """
    from backend.pipeline import vector_store

    records = [
        {"ticket_id": "FI-1", "problem": "Vendor invoice blocked due to price tolerance", "module": "FI"},
        {"ticket_id": "MM-1", "problem": "Material valuation master record missing moving average price", "module": "MM"},
        {"ticket_id": "SD-1", "problem": "Sales order delivery block active credit limit exceeded", "module": "SD"},
        {"ticket_id": "HR-1", "problem": "Payroll schema execution failed with time wage type error", "module": "HCM"},
        {"ticket_id": "PP-1", "problem": "Production order missing routing sequence for material BOM", "module": "PP"},
        {"ticket_id": "BASIS-1", "problem": "SAP GUI connection timeout buffer overflow work process killed", "module": "BASIS"},
    ]
    for r in records:
        assert vector_store.add_resolved_ticket(r) is True

    # Paraphrased query for HR payroll issue
    query = "Employee salary calculation job terminated due to wage type issue"
    results = vector_store.search_similar(query, n_results=3)

    assert len(results) >= 1
    assert results[0]["id"] == "HR-1"
    assert results[0]["metadata"]["module"] == "HCM"


def test_module_filter(isolated_chroma):
    """
    Asserts module_filter only returns that module. With include_unclassified=True,
    UNKNOWN records are also returned; with False, UNKNOWN records are omitted.
    """
    from backend.pipeline import vector_store

    records = [
        {"ticket_id": "T-MM", "problem": "MM invoice block issue", "module": "MM"},
        {"ticket_id": "T-SD", "problem": "SD sales order block issue", "module": "SD"},
        {"ticket_id": "T-UNK", "problem": "General system block issue", "module": "UNKNOWN"},
    ]
    for r in records:
        assert vector_store.add_resolved_ticket(r) is True

    # With include_unclassified=True: MM returns T-MM and T-UNK
    res_with_unk = vector_store.search_similar(
        "block issue", n_results=5, module_filter="MM", include_unclassified=True
    )
    ids_with_unk = {r["id"] for r in res_with_unk}
    assert "T-MM" in ids_with_unk
    assert "T-UNK" in ids_with_unk
    assert "T-SD" not in ids_with_unk

    # With include_unclassified=False: MM returns only T-MM
    res_no_unk = vector_store.search_similar(
        "block issue", n_results=5, module_filter="MM", include_unclassified=False
    )
    ids_no_unk = {r["id"] for r in res_no_unk}
    assert "T-MM" in ids_no_unk
    assert "T-UNK" not in ids_no_unk
    assert "T-SD" not in ids_no_unk


def test_cosine_check(isolated_chroma):
    """
    Authoritative test: asserts returned Chroma distances match
    1 - cosine_similarity computed with numpy from embed_text vectors (tolerance 1e-3).
    """
    from backend.pipeline import vector_store
    from backend.pipeline.embedder import embed_text

    t1 = "SAP purchase order goods receipt quantity mismatch"
    t2 = "Employee payroll tax withholding calculation error"
    t_query = "Purchase order discrepancy during goods receipt posting"

    vector_store.upsert_records(
        ids=["T-1", "T-2"],
        texts=[t1, t2],
        metadatas=[{"ticket_id": "T-1"}, {"ticket_id": "T-2"}],
        skip_existing=False,
    )

    v1 = np.array(embed_text(t1), dtype=np.float64)
    v2 = np.array(embed_text(t2), dtype=np.float64)
    vq = np.array(embed_text(t_query), dtype=np.float64)

    def np_cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    expected_dist_1 = 1.0 - float(np_cosine_sim(vq, v1))
    expected_dist_2 = 1.0 - float(np_cosine_sim(vq, v2))

    results = vector_store.search_similar(t_query, n_results=2)
    assert len(results) == 2

    res_dict = {r["id"]: r["distance"] for r in results}
    assert math.isclose(res_dict["T-1"], expected_dist_1, abs_tol=1e-3)
    assert math.isclose(res_dict["T-2"], expected_dist_2, abs_tol=1e-3)


def test_n_results_larger_than_collection(isolated_chroma):
    """
    Asserts n_results larger than collection size returns all available records and does not raise.
    """
    from backend.pipeline import vector_store

    vector_store.add_resolved_ticket({"ticket_id": "T-1", "problem": "First problem text"})
    vector_store.add_resolved_ticket({"ticket_id": "T-2", "problem": "Second problem text"})

    results = vector_store.search_similar("problem", n_results=100)
    assert len(results) == 2


def test_mismatch_protection(isolated_chroma, monkeypatch):
    """
    Asserts that changing EMBEDDING_MODEL_NAME after collection exists raises RuntimeError.
    """
    from backend.pipeline import vector_store

    # Create collection under default model
    vector_store.get_collection()

    # Patch model name
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_NAME", "other-model")

    with pytest.raises(RuntimeError) as exc_info:
        vector_store.get_collection()

    assert "Embedding model mismatch" in str(exc_info.value)
    assert "Run: python -m backend.ingestion.pipeline reset --yes" in str(exc_info.value)


def test_get_collection_stats_and_reset(isolated_chroma):
    """
    Asserts get_collection_stats(detailed=True) counts per module correctly,
    and reset_collection clears records leaving count at 0.
    """
    from backend.pipeline import vector_store

    vector_store.add_resolved_ticket({"ticket_id": "T-1", "problem": "FI problem", "module": "FI"})
    vector_store.add_resolved_ticket({"ticket_id": "T-2", "problem": "FI second problem", "module": "FI"})
    vector_store.add_resolved_ticket({"ticket_id": "T-3", "problem": "MM problem", "module": "MM"})

    stats = vector_store.get_collection_stats(detailed=True)
    assert stats["count"] == 3
    assert stats["modules"]["FI"] == 2
    assert stats["modules"]["MM"] == 1
    assert stats["source_types"]["resolved_ticket"] == 3

    # Reset collection
    vector_store.reset_collection()
    stats_after = vector_store.get_collection_stats()
    assert stats_after["count"] == 0
