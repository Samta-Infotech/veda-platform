"""apps.sources.item_profiler — build + profile the uniform SourceItem layer for one source.

The importable service behind both the `backfill_source_items`/`profile_source_items` commands and the
post-ingestion hook (multi-source routing). For one source it:
  1. BACKFILLS SourceItem rows from the kind-specific stores (tables from column_embeddings_v2,
     documents from doc_chunks) — filesystem docs become first-class items like datalake datasets.
  2. PROFILES each item — an SLM one-line summary + topics, and a BGE-M3 embedding of (name + summary)
     into the engine-side `source_item_embeddings` (pgvector), which is the query-time routing PRIOR.

Flag-gated (`SOURCE_ITEM_PROFILER_ENABLED`, default OFF) and never-raises from the hook, so it can
never fail an ingestion job that already succeeded (mirrors source_profiler / catalog sync). SLM +
embedding endpoints come from settings/env; when unreachable the item still gets a structural row
(name/type/child_count), just no summary/embedding — routing then falls back to column/chunk evidence.

See docs/multisource_routing/SOURCE_ITEM_METADATA_DESIGN.md.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

from django.conf import settings
from django.db import connection

from apps.sources.models import Source, SourceItem, SourceItemType

logger = logging.getLogger(__name__)

_SLM_URL = os.environ.get("VEDA_SLM_CHAT_URL", "http://192.168.1.35:11500/api/chat")
_SLM_MODEL = os.environ.get("SLM_MODEL_NAME", "qwen2.5-coder:7b")
_METAL_URL = os.environ.get("METAL_EMBED_URL", "http://192.168.1.39:11435").rstrip("/") + "/encode_dense"

_SYS = ("You describe ONE data item (a table, dataset, or document) for a query router. Given its name "
        "and observed content, reply with STRICT JSON: {\"summary\": \"<one specific sentence: what it "
        "holds and what questions it answers>\", \"topics\": [\"..\"]}. Be specific about the business "
        "domain. No preamble.")


def _is_enabled() -> bool:
    return bool(getattr(settings, "SOURCE_ITEM_PROFILER_ENABLED", False))


def _rows(sql, params):
    with connection.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _post_json(url, payload, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ── 1. backfill structural items ────────────────────────────────────────────────────────────────
def backfill_items(source: Source) -> int:
    """Backfill the SourceItem table for one source from its kind-specific stores (column_embeddings_v2, doc_chunks)"""
    sid = str(source.pk)
    kind = source.source_kind()
    made = 0
    try:
        tabs = _rows("SELECT table_name, count(*) FROM column_embeddings_v2 WHERE source_id=%s "
                     "GROUP BY table_name", [sid])
    except Exception:
        tabs = []
    item_type = (SourceItemType.DATASET if kind == "datalake"
                 else SourceItemType.COLLECTION if kind == "nosql" else SourceItemType.TABLE)
    for table_name, ncols in tabs:
        cols = [r[0] for r in _rows(
            "SELECT col_name FROM column_embeddings_v2 WHERE source_id=%s AND table_name=%s LIMIT 50",
            [sid, table_name])]
        SourceItem.objects.update_or_create(
            source=source, item_type=item_type, item_key=table_name,
            defaults=dict(name=table_name, child_count=ncols,
                          item_metadata={"columns": cols, "column_count": ncols}))
        made += 1
    try:
        docs = _rows("SELECT doc_name, count(*) FROM doc_chunks WHERE source_id=%s GROUP BY doc_name",
                     [sid])
    except Exception:
        docs = []
    for doc_name, nchunks in docs:
        ftype = doc_name.rsplit(".", 1)[-1].lower() if "." in doc_name else ""
        SourceItem.objects.update_or_create(
            source=source, item_type=SourceItemType.DOCUMENT, item_key=doc_name,
            defaults=dict(name=doc_name, child_count=nchunks,
                          item_metadata={"file_type": ftype, "chunk_count": nchunks}))
        made += 1
    return made


# ── 2. profile each item (SLM summary + embedding) ───────────────────────────────────────────────
def _observed(item: SourceItem) -> str:
    """Compact observed content for the SLM prompt, per item type."""
    sid = str(item.source_id)
    if item.item_type == SourceItemType.DOCUMENT:
        sample = " ".join(r[0][:300] for r in _rows(
            "SELECT text FROM doc_chunks WHERE source_id=%s AND doc_name=%s ORDER BY chunk_index LIMIT 3",
            [sid, item.item_key]))
        return f"Document: {item.name}\nText sample: {sample[:800]}"
    cols = ", ".join((item.item_metadata or {}).get("columns", [])[:30])
    return f"{item.item_type}: {item.name}\nColumns: {cols}"


def _slm_summary(observed: str):
    """Call the SLM endpoint to summarise one item. Returns (summary, topics)."""
    out = _post_json(_SLM_URL, {"model": _SLM_MODEL, "keep_alive": "24h", "stream": False,
                                "messages": [{"role": "system", "content": _SYS},
                                             {"role": "user", "content": observed}]})
    raw = (out.get("message") or {}).get("content", "")
    a, b = raw.find("{"), raw.rfind("}")
    try:
        d = json.loads(raw[a:b + 1])
        return (d.get("summary") or "").strip()[:1000], list(d.get("topics") or [])
    except Exception:
        return str(raw or "").strip()[:300], []


def _embed(text: str):
    """Call the embedding endpoint to embed one item (name + summary). Returns a list of floats."""
    return _post_json(_METAL_URL, {"texts": [text]}, timeout=30)["vecs"][0]


def profile_items(source: Source, force: bool = False) -> int:
    """Profile each SourceItem for one source: SLM summary + topics, and embedding into pgvector."""
    qs = source.items.all()
    if not force:
        qs = qs.filter(summary="")
    done = 0
    for item in qs:
        try:
            summary, topics = _slm_summary(_observed(item))
            item.summary, item.topics = summary, topics
            item.save(update_fields=["summary", "topics", "updated_at"])
            vec = _embed(f"{item.name}. {summary}")
            with connection.cursor() as cur:
                cur.execute(
                    "INSERT INTO source_item_embeddings (source_id, item_type, item_key, name, summary, "
                    "embedding, updated_at) VALUES (%s,%s,%s,%s,%s,%s,now()) "
                    "ON CONFLICT (source_id, item_type, item_key) DO UPDATE SET "
                    "name=EXCLUDED.name, summary=EXCLUDED.summary, embedding=EXCLUDED.embedding, "
                    "updated_at=now()",
                    [str(item.source_id), item.item_type, item.item_key, item.name, summary,
                     "[" + ",".join(f"{v:.8f}" for v in vec) + "]"])
            done += 1
        except Exception as e:
            logger.warning("item profile failed source=%s item=%s: %s", source.pk, item.item_key, e)
    return done


def build_source_items(source_id, force: bool = False) -> dict:
    """Backfill + profile one source's items. Returns {items, profiled}."""
    source = Source.objects.get(pk=source_id)
    n_items = backfill_items(source)
    n_prof = profile_items(source, force=force)
    return {"items": n_items, "profiled": n_prof}


def build_source_items_if_enabled(source_id) -> None:
    """Flag-gated, never-raises entry point for the ingestion hook."""
    if not _is_enabled():
        return
    try:
        r = build_source_items(source_id)
        logger.info("source items built source_id=%s items=%s profiled=%s",
                    source_id, r["items"], r["profiled"])
    except Exception:
        logger.exception("source item profiling failed source_id=%s — ingestion still succeeded",
                         source_id)
