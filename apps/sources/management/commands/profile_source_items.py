"""Profile each SourceItem with an SLM summary + topics, and embed it for the query-time routing prior.

For every SourceItem: build a compact view of its observed content (columns for a table, a text sample
+ entities for a document), ask the SLM for a one-line business summary + topics, store them on the
SourceItem, then embed (name + summary) via the Metal BGE-M3 endpoint into the engine-side raw table
`source_item_embeddings`. That embedding is the source-level routing PRIOR the coordinator uses to pick
the source whose ITEM is semantically about a query — robust to a big DB's spurious column match
(docs/multisource_routing/SOURCE_ITEM_METADATA_DESIGN.md).

Idempotent; skips items already summarised unless --force.

    python manage.py profile_source_items [--source N] [--force] [--limit N]
"""
import json
import urllib.request

from django.core.management.base import BaseCommand
from django.db import connection

from apps.sources.models import SourceItem

SLM_URL = "http://192.168.1.35:11500/api/chat"
SLM_MODEL = "qwen2.5-coder:7b"
METAL_URL = "http://192.168.1.39:11435/encode_dense"

_SYS = ("You describe ONE data item (a table, dataset, or document) for a query router. Given its name "
        "and observed content, reply with STRICT JSON: {\"summary\": \"<one specific sentence: what it "
        "holds and what questions it answers>\", \"topics\": [\"..\",\"..\"]}. Be specific about the "
        "business domain. No preamble.")


def _post_json(url, payload, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _slm(content):
    out = _post_json(SLM_URL, {"model": SLM_MODEL, "keep_alive": "24h", "stream": False,
                               "messages": [{"role": "system", "content": _SYS},
                                            {"role": "user", "content": content}]})
    return (out.get("message") or {}).get("content", "")


def _parse(raw):
    s = str(raw or "").strip()
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        s = s[a:b + 1]
    try:
        d = json.loads(s)
        return (d.get("summary") or "").strip(), list(d.get("topics") or [])
    except Exception:
        return str(raw or "").strip()[:300], []


def _embed(text):
    out = _post_json(METAL_URL, {"texts": [text]}, timeout=30)
    return out["vecs"][0]


def _observed(item):
    """Compact observed content for the SLM prompt, per item type."""
    sid = str(item.source_id)
    if item.item_type == "document":
        with connection.cursor() as cur:
            cur.execute("SELECT text FROM doc_chunks WHERE source_id=%s AND doc_name=%s "
                        "ORDER BY chunk_index LIMIT 3", [sid, item.item_key])
            sample = " ".join(r[0][:300] for r in cur.fetchall())
        ents = ", ".join((item.item_metadata or {}).get("entities", [])[:10])
        return f"Document: {item.name}\nText sample: {sample[:800]}\nEntities: {ents}"
    cols = ", ".join((item.item_metadata or {}).get("columns", [])[:30])
    return f"{item.item_type.title()}: {item.name}\nColumns: {cols}"


class Command(BaseCommand):
    help = "SLM-summarise + embed each SourceItem for the routing prior."

    def add_arguments(self, parser):
        parser.add_argument("--source", type=int, default=None)
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--limit", type=int, default=None)

    def handle(self, *args, **opts):
        qs = SourceItem.objects.all().order_by("source_id", "item_type", "name")
        if opts["source"]:
            qs = qs.filter(source_id=opts["source"])
        if not opts["force"]:
            qs = qs.filter(summary="")
        if opts["limit"]:
            qs = qs[:opts["limit"]]
        done = fail = 0
        for item in qs:
            try:
                summary, topics = _parse(_slm(_observed(item)))
                item.summary = summary[:1000]
                item.topics = topics
                item.save(update_fields=["summary", "topics", "updated_at"])
                vec = _embed(f"{item.name}. {summary}")
                with connection.cursor() as cur:
                    cur.execute(
                        "INSERT INTO source_item_embeddings (source_id, item_type, item_key, name, "
                        "summary, embedding, updated_at) VALUES (%s,%s,%s,%s,%s,%s,now()) "
                        "ON CONFLICT (source_id, item_type, item_key) DO UPDATE SET "
                        "name=EXCLUDED.name, summary=EXCLUDED.summary, embedding=EXCLUDED.embedding, "
                        "updated_at=now()",
                        [str(item.source_id), item.item_type, item.item_key, item.name, summary,
                         "[" + ",".join(f"{v:.8f}" for v in vec) + "]"])
                done += 1
                if done % 20 == 0:
                    self.stdout.write(f"  ...{done} profiled")
            except Exception as e:
                fail += 1
                self.stderr.write(f"  item {item.item_key} failed: {e}")
        self.stdout.write(self.style.SUCCESS(f"Profiled {done} items ({fail} failed)."))
