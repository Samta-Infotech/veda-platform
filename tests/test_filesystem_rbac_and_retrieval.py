"""Coverage for the veda_core-side of this session's filesystem/document (RAG)
pipeline fixes:

  1. veda_core/veda/rbac_filter.py::filter_doc_chunks
     — the document-retrieval mirror of filter_nosql_collections: narrows a
       chunk-retrieval candidate list to allowed documents only.

  2. veda_core/ingestion/chunk_embedder.py::retrieve_top_k_chunks
     — actually calls filter_doc_chunks on the ambient RBAC context before
       truncating to top_k (previously: RBAC data-scope was computed but
       never consulted here at all — a denied document's chunks still
       surfaced in every RAG/hybrid answer, verified live against a real
       docs_contracts source in this session before this fix).

Pure functions + a mocked psycopg2 connection — no Django, no real DB. Runs
against ``veda_core`` on sys.path directly, matching how the inference tier
imports it (same precedent as tests/test_rbac_filter.py).

Run from repo root: ``pytest tests/test_filesystem_rbac_and_retrieval.py``
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

from veda.rbac_filter import filter_doc_chunks  # noqa: E402
from ingestion import chunk_embedder
import numpy as np


def _ctx(source_ids, allowed_resources):
    @dataclass(frozen=True)
    class _Ctx:
        source_ids: tuple
        allowed_resources: tuple = None
    return _Ctx(source_ids=tuple(source_ids), allowed_resources=allowed_resources)


@dataclass
class _Chunk:
    chunk_id: str
    source_id: str
    doc_id: str
    doc_name: str
    chunk_index: int = 0
    text: str = ""
    page_num: int = None
    similarity: float = 0.5


def _chunk(doc_name, source_id="3"):
    return _Chunk(chunk_id=f"c-{doc_name}", source_id=source_id, doc_id=f"d-{doc_name}",
                 doc_name=doc_name)


# ---------------------------------------------------------------------------
# filter_doc_chunks
# ---------------------------------------------------------------------------

def test_doc_chunks_no_ctx_is_identity():
    chunks = [_chunk("msa.pdf")]
    assert filter_doc_chunks(chunks, None) is chunks


def test_doc_chunks_ctx_with_no_allowed_resources_is_identity():
    chunks = [_chunk("msa.pdf")]
    ctx = _ctx([3], None)
    assert filter_doc_chunks(chunks, ctx) is chunks


def test_doc_chunks_empty_input_is_identity():
    ctx = _ctx([3], ((3, (False, (("msa.pdf", None),))),))
    assert filter_doc_chunks([], ctx) == []


def test_doc_chunks_open_source_keeps_every_document():
    chunks = [_chunk("msa.pdf"), _chunk("notes.md")]
    ctx = _ctx([3], ((3, (True, ())),))
    assert filter_doc_chunks(chunks, ctx) is chunks


def test_doc_chunks_restricted_source_drops_the_unlisted_document():
    """The exact bug found live in this session: a document explicitly outside
    the allow-list must never surface, even when the source itself is reachable."""
    chunks = [_chunk("msa.pdf"), _chunk("maintenance_policy.docx")]
    ctx = _ctx([3], ((3, (False, (("msa.pdf", None),))),))
    kept = filter_doc_chunks(chunks, ctx)
    assert [c.doc_name for c in kept] == ["msa.pdf"]


def test_doc_chunks_restricted_source_with_zero_allowed_docs_drops_everything():
    chunks = [_chunk("msa.pdf")]
    ctx = _ctx([3], ((3, (False, ())),))
    assert filter_doc_chunks(chunks, ctx) == []


def test_doc_chunks_a_second_source_unmentioned_in_scope_is_denied():
    chunks = [_chunk("msa.pdf", source_id="4")]
    ctx = _ctx([3], ((3, (True, ())),))  # only source 3 is open; source 4 unmentioned
    assert filter_doc_chunks(chunks, ctx) == []


def test_doc_chunks_multi_source_each_restricted_independently():
    chunks = [_chunk("msa.pdf", source_id="3"), _chunk("readme.txt", source_id="4")]
    ctx = _ctx([3, 4], (
        (3, (False, (("msa.pdf", None),))),
        (4, (False, (("other.txt", None),))),  # readme.txt NOT in source 4's allow-list
    ))
    kept = filter_doc_chunks(chunks, ctx)
    assert [c.doc_name for c in kept] == ["msa.pdf"]


def test_doc_chunks_never_mutates_the_input_list():
    chunks = [_chunk("msa.pdf"), _chunk("notes.md")]
    ctx = _ctx([3], ((3, (False, (("msa.pdf", None),))),))
    filter_doc_chunks(chunks, ctx)
    assert [c.doc_name for c in chunks] == ["msa.pdf", "notes.md"]  # original untouched


# ---------------------------------------------------------------------------
# retrieve_top_k_chunks — over-fetch + RBAC filter wiring
# ---------------------------------------------------------------------------

def _fake_rows(names, source_id="3"):
    return [
        {"chunk_id": f"c{i}", "source_id": source_id, "doc_id": f"d{i}", "doc_name": name,
         "chunk_index": 0, "text": "x", "page_num": None, "similarity": 0.9 - i * 0.01}
        for i, name in enumerate(names)
    ]


def _fake_conn(rows):
    fake_cursor = mock.MagicMock()
    fake_cursor.__enter__.return_value = fake_cursor
    fake_cursor.fetchall.return_value = rows
    fake_conn = mock.MagicMock()
    fake_conn.cursor.return_value = fake_cursor
    return fake_conn


def test_retrieve_top_k_chunks_applies_rbac_filter_and_truncates_to_top_k():
    rows = _fake_rows(["msa.pdf", "denied.docx", "msa.pdf", "denied.docx", "msa.pdf",
                       "denied.docx", "msa.pdf", "denied.docx"])
    ctx = _ctx([3], ((3, (False, (("msa.pdf", None),))),))

    with mock.patch.object(chunk_embedder, "INTERNAL_DB_AVAILABLE", True), \
         mock.patch.object(chunk_embedder, "get_internal_connection",
                           return_value=_fake_conn(rows)), \
         mock.patch.object(chunk_embedder, "release_internal_connection"), \
         mock.patch("storage_adapters.reader._resolve_ef_search", return_value=40), \
         mock.patch.object(chunk_embedder, "_try_current_context", return_value=ctx):
        results = chunk_embedder.retrieve_top_k_chunks(
            query_vector=np.zeros(4, dtype="float32"), source_ids=["3"], top_k=3)

    assert len(results) == 3
    assert all(r.doc_name == "msa.pdf" for r in results)


def test_retrieve_top_k_chunks_no_context_is_unfiltered():
    rows = _fake_rows(["anything.pdf"])

    with mock.patch.object(chunk_embedder, "INTERNAL_DB_AVAILABLE", True), \
         mock.patch.object(chunk_embedder, "get_internal_connection",
                           return_value=_fake_conn(rows)), \
         mock.patch.object(chunk_embedder, "release_internal_connection"), \
         mock.patch("storage_adapters.reader._resolve_ef_search", return_value=40), \
         mock.patch.object(chunk_embedder, "_try_current_context", return_value=None):
        results = chunk_embedder.retrieve_top_k_chunks(
            query_vector=np.zeros(4, dtype="float32"), source_ids=["3"], top_k=5)

    assert [r.doc_name for r in results] == ["anything.pdf"]


def test_retrieve_top_k_chunks_restricted_with_no_matches_returns_empty():
    rows = _fake_rows(["denied.docx", "denied.docx"])
    ctx = _ctx([3], ((3, (False, (("msa.pdf", None),))),))

    with mock.patch.object(chunk_embedder, "INTERNAL_DB_AVAILABLE", True), \
         mock.patch.object(chunk_embedder, "get_internal_connection",
                           return_value=_fake_conn(rows)), \
         mock.patch.object(chunk_embedder, "release_internal_connection"), \
         mock.patch("storage_adapters.reader._resolve_ef_search", return_value=40), \
         mock.patch.object(chunk_embedder, "_try_current_context", return_value=ctx), \
         mock.patch.object(chunk_embedder, "_IN_MEMORY_CHUNKS", None):
        results = chunk_embedder.retrieve_top_k_chunks(
            query_vector=np.zeros(4, dtype="float32"), source_ids=["3"], top_k=5)

    assert results == []
