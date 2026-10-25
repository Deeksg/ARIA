"""
backend/tests/test_ingestion.py
===============================
Unit tests for data and document ingestion pipeline (backend.ingestion.pipeline).
"""

import pytest
from pathlib import Path


def test_chunk_text():
    """
    Asserts chunk_text behavior:
    - Short text yields 1 chunk.
    - 400-word text yields chunks <= 150 words with exactly 30-word overlap.
    - All words from input are covered.
    - overlap_words >= chunk_words raises ValueError.
    """
    from backend.ingestion.pipeline import chunk_text

    # Short text
    short = "Brief SAP support notification."
    assert len(chunk_text(short)) == 1

    # 400 words text
    words = [f"word_{i}" for i in range(400)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_words=150, overlap_words=30)

    for c in chunks:
        assert len(c.split()) <= 150

    # Verify 30-word overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        c1_words = chunks[i].split()
        c2_words = chunks[i + 1].split()
        assert c1_words[-30:] == c2_words[:30]

    # Reconstruct and check all words are covered
    reconstructed = []
    for i, c in enumerate(chunks):
        c_words = c.split()
        if i == 0:
            reconstructed.extend(c_words)
        else:
            reconstructed.extend(c_words[30:])
    assert reconstructed == words

    # Validation: overlap >= chunk_words must raise ValueError
    with pytest.raises(ValueError):
        chunk_text("Sample", chunk_words=50, overlap_words=50)
    with pytest.raises(ValueError):
        chunk_text("Sample", chunk_words=50, overlap_words=60)


def test_load_resolved_csv(tmp_path):
    """
    Asserts load_resolved_csv correctly parses valid records, keeps empty error_code as "",
    skips blank problem rows, handles non-numeric resolution times as -1,
    and raises ValueError if required columns are missing.
    """
    from backend.ingestion.pipeline import load_resolved_csv

    csv_file = tmp_path / "test_tickets.csv"
    csv_file.write_text(
        "ticket_id,module,issue_type,problem,error_code,root_cause,fix,tcodes_used,resolution_time_minutes,escalated,created_date,resolved_date,ticket_type\n"
        "T-1,FI,Config,Valid invoice error,,Table missing,Fix steps,FB60,60,No,2026-08-01,2026-08-01,incident\n"
        "T-2,MM,Error,   ,,Root cause,Fix steps,ME21N,abc,Yes,2026-08-01,2026-08-01,incident\n"
        "T-3,SD,User,Sales order credit check,ERR99,Limit reached,Release order,VA01,invalid,No,2026-08-01,2026-08-01,incident\n",
        encoding="utf-8",
    )

    records, summary = load_resolved_csv(csv_file)
    assert summary["total_rows"] == 3
    assert summary["skipped_blank"] == 1  # T-2 had blank problem
    assert summary["valid_records"] == 2  # T-1 and T-3

    rec1 = records[0]
    assert rec1["ticket_id"] == "T-1"
    assert rec1["error_code"] == ""
    assert rec1["resolution_time_minutes"] == 60

    rec3 = records[1]
    assert rec3["ticket_id"] == "T-3"
    assert rec3["resolution_time_minutes"] == -1  # Non-numeric converted to -1

    # Missing column check
    bad_csv = tmp_path / "bad_tickets.csv"
    bad_csv.write_text("ticket_id,problem\nT-1,Some problem\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_resolved_csv(bad_csv)
    assert "Missing required columns" in str(exc.value)


def test_ingest_csv(isolated_chroma, tmp_path):
    """
    Asserts ingest_csv records correct count, second run adds 0 records (idempotent),
    and force=True leaves count unchanged.
    """
    from backend.ingestion.pipeline import ingest_csv
    from backend.pipeline import vector_store

    csv_file = tmp_path / "clean_sample.csv"
    csv_file.write_text(
        "ticket_id,module,issue_type,problem,error_code,root_cause,fix,tcodes_used,resolution_time_minutes,escalated,created_date,resolved_date,ticket_type\n"
        "T-A,FI,Config,Posting period closed in variant 0001,F5155,OB52 closed,Open period,OB52,30,No,2026-08-01,2026-08-01,incident\n"
        "T-B,MM,Master,Material master record blocked by user,M3020,Lock table active,Wait unlock,MM02,45,No,2026-08-01,2026-08-01,incident\n",
        encoding="utf-8",
    )

    # First run
    s1 = ingest_csv(csv_path=csv_file, force=False)
    assert s1["newly_added"] == 2
    assert s1["already_present"] == 0
    assert vector_store.get_collection_stats()["count"] == 2

    # Second run without force: 0 added, 2 already present
    s2 = ingest_csv(csv_path=csv_file, force=False)
    assert s2["newly_added"] == 0
    assert s2["already_present"] == 2
    assert vector_store.get_collection_stats()["count"] == 2

    # Third run with force: rewrites 2, count remains 2
    s3 = ingest_csv(csv_path=csv_file, force=True)
    assert s3["newly_added"] == 2
    assert vector_store.get_collection_stats()["count"] == 2


def test_ingest_files(isolated_chroma, tmp_path):
    """
    Asserts ingest_files on folder holding .txt, .docx, blank .pdf, and .doc:
    - txt and docx chunks are ingested and searchable.
    - Blank pdf is skipped with warning.
    - .doc is listed as unsupported.
    - Re-ingesting an edited .txt leaves exactly the new number of chunks (no stale chunks).
    """
    import docx
    from pypdf import PdfWriter
    from backend.ingestion.pipeline import ingest_files
    from backend.pipeline import vector_store

    docs_dir = tmp_path / "sap_docs"
    docs_dir.mkdir()

    # 1. Text file
    txt_file = docs_dir / "user_guide.txt"
    txt_file.write_text(
        "SAP Financial Accounting GL Account Maintenance Guide. "
        "Use transaction FS00 to maintain general ledger account master data centrally.",
        encoding="utf-8",
    )

    # 2. Word document
    docx_file = docs_dir / "troubleshooting.docx"
    doc = docx.Document()
    doc.add_paragraph("SAP Materials Management PO tolerance configuration guide.")
    doc.save(str(docx_file))

    # 3. Blank PDF file
    pdf_file = docs_dir / "blank_note.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with open(pdf_file, "wb") as f:
        writer.write(f)

    # 4. Unsupported format (.doc)
    doc_file = docs_dir / "legacy_note.doc"
    doc_file.write_text("Legacy word format", encoding="utf-8")

    # Ingest folder
    summary = ingest_files(folder=docs_dir, module="FI")
    assert summary["files_ingested"] == 2  # txt and docx
    assert "user_guide.txt" not in summary["skipped_empty"]
    assert "troubleshooting.docx" not in summary["skipped_empty"]
    assert "blank_note.pdf" in summary["skipped_empty"]
    assert "legacy_note.doc" in summary["skipped_unsupported"]

    # Verify search finds the docx content
    res = vector_store.search_similar("materials management tolerance configuration", n_results=1)
    assert len(res) >= 1
    assert "troubleshooting.docx" in res[0]["metadata"]["source_file"]

    # Re-ingest edited txt file: should replace previous chunks with 0 stale chunks
    txt_file.write_text(
        "Updated short single sentence user guide for general ledger.",
        encoding="utf-8",
    )
    summary2 = ingest_files(folder=docs_dir, module="FI")
    assert summary2["files_ingested"] == 2

    # Check count for user_guide.txt specifically
    col = vector_store.get_collection()
    guide_records = col.get(where={"source_file": "user_guide.txt"}, include=["metadatas"])
    assert len(guide_records["ids"]) == 1


def test_document_chunk_metadata(isolated_chroma, tmp_path):
    """
    Asserts document chunk metadata has source_type, source_file (relative, forward slashes),
    chunk_index, and total_chunks.
    """
    from backend.ingestion.pipeline import ingest_files
    from backend.pipeline import vector_store

    docs_dir = tmp_path / "docs_sub"
    docs_dir.mkdir()
    sub_dir = docs_dir / "finance"
    sub_dir.mkdir()

    f = sub_dir / "manual.txt"
    # Create text with > 150 words to produce multiple chunks
    words = [f"word{i}" for i in range(200)]
    f.write_text(" ".join(words), encoding="utf-8")

    ingest_files(folder=docs_dir, module="FI", issue_type="Configuration")

    results = vector_store.search_similar("word0 word1", n_results=5)
    assert len(results) >= 1

    meta = results[0]["metadata"]
    assert meta["source_type"] == "document_chunk"
    assert meta["source_file"] == "finance/manual.txt"  # Relative, forward slashes
    assert "chunk_index" in meta
    assert meta["total_chunks"] >= 2
    assert meta["module"] == "FI"
    assert meta["issue_type"] == "Configuration"
