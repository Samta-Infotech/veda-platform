"""ingestion/routing_card.py — the per-source ROUTING CARD (Checkpoint B.1).

The routing SLM currently decides between sources from whatever evidence retrieval
happened to surface for one question: a handful of column names and chunk snippets
(query/routing_slm.py::_build_user_message). That is a keyhole view — it tells the
model which columns matched, never what the source actually IS. The card is the map:
a compact, deterministic, per-source description built ONCE at ingestion time and
handed to the router whole.

WHAT IS AND IS NOT IN THE CARD
------------------------------
The card carries the STRUCTURAL half — what this source holds — because that is what
ingestion can see: entities (tables/datasets/documents), their business naming, size,
key dimensions and measures, real sample values, and the cross-source joins this
source participates in.

It deliberately does NOT carry `name` / `domain_tags` / `description`. Those are
REGISTRY facts that live on the Django `Source` row (apps/sources/models.py — a human
may edit them at any time, long after ingestion), and veda_core must never import
Django (the same boundary chatbot/llm.py and apps/query/inference_client.py document).
They already reach the engine at query time through `context.current_source_profiles()`,
which is where the router composes them onto the card. Baking a stale copy into an
ingestion-time artifact would give the router two disagreeing descriptions of the same
source and no rule for which wins.

"BUSINESS NAME"
---------------
The plan for this checkpoint called for a `business_name` per entity from L3 and
recorded "source 2: 0/178" as an ingestion defect to fix. That reading is wrong, and
re-running L3 for source 2 would change nothing: `semantic_layer_v2` has never emitted
a field by that name, for ANY source — what it emits per table is
`business_purpose` / `primary_entity` / `table_type` (verified in-tree at
veda/understanding/grounding.py:166-170, and confirmed here: source 2 has
`primary_entity` populated on 178/178 tables and `business_purpose` on 178/178).
`primary_entity` IS the business name — "what does one row represent" — and the
grounding layer already treats it as such. So the card reads it, and falls back to
`business_name`/`display_name` for any model that does carry them.

Everything here is a pure transform of artifacts THIS ingestion already produced
(the semantic model, the scan result, the value store, the cross-source edges). No
LLM call, no new scan, no new embedding.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ingestion.routing_card")

CARD_ARTIFACT = "veda_routing_card.json"
CARD_VERSION = 1

# Caps. The whole point is a prompt-sized map: the plan's budget is ~1.5k tokens for
# five sources, so one card has to stay around ~300 tokens when rendered compactly.
MAX_ENTITIES = 20
MAX_DIMS = 6
MAX_MEASURES = 6
MAX_SAMPLE_COLS = 3
MAX_SAMPLE_VALUES = 5
MAX_EXAMPLE_QUESTIONS = 8

# Table types that describe plumbing rather than a business entity. A BRIDGE/junction
# table ("links attachments to ledger entries") is never what a question is about, and
# spending card budget on them is what would push a real entity past MAX_ENTITIES.
_NON_ENTITY_TABLE_TYPES = {"BRIDGE", "JUNCTION", "LINK", "MAPPING", "AUDIT", "LOG"}

# Values that are booleans by any spelling — never useful as card sample values.
_BOOLEANISH = {"true", "false", "t", "f", "0", "1", "yes", "no", "y", "n", "none", "null"}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# Prepositions that begin a QUALIFIER rather than part of the name itself. L3's
# `primary_entity` answers "what does one row represent" as a sentence — "A single
# settlement record for a set of payment transactions." — and the head noun phrase
# before the first such preposition is the actual name ("settlement record").
_QUALIFIER_HEADS = (" for ", " of ", " between ", " with ", " within ", " in ",
                    " to ", " from ", " that ", " which ", " used ", " associated ",
                    " belonging ", " linked ", " related ")
_LEADING_ARTICLES = ("a single ", "an single ", "the single ", "a ", "an ", "the ")


def _noun_phrase(sentence: str) -> str:
    """L3's row-description sentence → the bare noun phrase naming the entity.

    Purely mechanical (article strip, qualifier-clause cut, trailing punctuation) —
    no LLM, no word list beyond the closed set of prepositions above. Without this the
    card reads "how many a single financial transaction. per transaction type", which
    is worse than useless to a router matching natural-language questions.
    """
    t = " ".join(str(sentence or "").split()).rstrip(". ").strip()
    if not t:
        return ""
    low = t.lower()
    for art in _LEADING_ARTICLES:
        if low.startswith(art):
            t, low = t[len(art):], low[len(art):]
            break
    # cut at the first qualifier clause, and at any comma
    cut = len(t)
    for sep in _QUALIFIER_HEADS:
        i = low.find(sep)
        if 0 < i < cut:
            cut = i
    i = t.find(",")
    if 0 < i < cut:
        cut = i
    t = t[:cut].strip()
    return " ".join(t.split()[:5])


def _pluralise(name: str) -> str:
    """Naive, deterministic English plural for the last word of a noun phrase — the
    example questions read as questions ("how many financial transactions"), not as
    schema notes. Wrong on irregulars; harmless, since these strings are routing
    pattern-match material, never user-facing copy."""
    if not name:
        return name
    head, _, last = name.rpartition(" ")
    if not last:
        last, head = name, ""
    lo = last.lower()
    if lo.endswith("s") and not lo.endswith("ss"):
        p = last          # already plural ("vendors"), don't make it "vendorses"
    elif lo.endswith(("x", "z", "ch", "sh", "ss")):
        p = last + "es"
    elif lo.endswith("y") and len(lo) > 1 and lo[-2] not in "aeiou":
        p = last[:-1] + "ies"
    else:
        p = last + "s"
    return f"{head} {p}".strip()


def _business_name(meta: Dict[str, Any], table: str) -> str:
    """The human-facing name of one entity. See the module docstring: `primary_entity`
    is what L3 actually emits and is the business name in all but label; the other two
    are read for any model that carries them. L3's value is a SENTENCE, so it is
    reduced to its head noun phrase — `business_name`/`display_name`, when a model
    carries them, are already names and are taken verbatim."""
    for key in ("business_name", "display_name"):
        v = str((meta or {}).get(key) or "").strip()
        if v:
            return v
    np = _noun_phrase((meta or {}).get("primary_entity"))
    return np or table


def _is_entity_table(meta: Dict[str, Any]) -> bool:
    ttype = str((meta or {}).get("table_type") or "").strip().upper()
    return ttype not in _NON_ENTITY_TABLE_TYPES


def _columns_by_table(sm: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """semantic_model["columns"] is keyed "<table>.<col>"; group it by table once."""
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for key, col in (sm.get("columns") or {}).items():
        tbl = str((col or {}).get("table_name") or "").strip()
        if not tbl and "." in str(key):
            tbl = str(key).rsplit(".", 1)[0]
        if tbl:
            out[tbl].append(col or {})
    return out


def _role_names(cols: List[Dict[str, Any]], role: str, limit: int) -> List[str]:
    """The business-facing names of this table's columns in one analytics role.

    Prefers `business_role` ("Transaction Type") over the raw column name ("label") —
    the router is matching a natural-language question, and the raw name is frequently
    an abbreviation that matches nothing. Stable order: the semantic model's own.
    """
    out: List[str] = []
    seen = set()
    for c in cols:
        if str(c.get("analytics_role") or "").strip().upper() != role:
            continue
        name = str(c.get("business_role") or "").strip() or str(c.get("col_name") or "").strip()
        k = name.lower()
        if name and k not in seen:
            seen.add(k)
            out.append(name)
        if len(out) >= limit:
            break
    return out


def _sample_values(source_id: str, tables: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Real values per (table, dimension column), read from the engine's own
    `column_values` store — the same sampled values the value-grounding layer matches
    against, so a value that appears in a card is by construction a value a query can
    actually filter on.

    Read through the internal-store connection, NEVER Django's `connection`: the engine
    tables live in a SEPARATE database (veda_engine) and a Django-routed query against
    them returns "relation does not exist", which callers here would swallow as "no
    values" (the failure mode that hit storage_adapters/reader.py::ann_search).
    Best-effort: any failure yields no samples and the card is still written.
    """
    if not tables:
        return {}
    out: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    try:
        from ingestion.db_abstraction import internal_connection
        with internal_connection() as conn:
            with conn.cursor() as cur:
                # DISTINCT ON keeps this one indexed pass instead of a query per column.
                cur.execute(
                    """
                    SELECT table_name, col_name, value_raw
                      FROM column_values
                     WHERE table_name = ANY(%s)
                       AND semantic_type IN ('CATEGORY', 'STATUS', 'LOCATION', 'NAME', 'TYPE')
                     ORDER BY table_name, col_name, value_raw
                    """,
                    (list(tables),),
                )
                rows = cur.fetchall() or []
    except Exception as exc:                       # pragma: no cover - store may be cold
        logger.warning("routing_card: sample values unavailable (%s: %s)",
                       type(exc).__name__, exc)
        return {}

    for tbl, col, val in rows:
        # Booleans are typed CATEGORY by the sampler but carry no routing signal: a
        # card line reading "e.g. email_verified: False" spends budget saying nothing,
        # and crowds out the values that DO disambiguate a source ("status: APPROVED,
        # NEW"). Skip a column whose sampled values are entirely boolean-ish.
        if str(val).strip().lower() in _BOOLEANISH:
            continue
        per_table = out[tbl]
        if col not in per_table and len(per_table) >= MAX_SAMPLE_COLS:
            continue
        vals = per_table.setdefault(col, [])
        if len(vals) < MAX_SAMPLE_VALUES and val is not None:
            s = str(val).strip()
            if s and s not in vals:
                vals.append(s)
    return {t: dict(c) for t, c in out.items()}


def _fk_centrality() -> Dict[str, int]:
    """FK in-degree per table: how many columns across the schema point AT this table.

    This is the ranking signal for "which entities belong on the card". Without it the
    only available signals are row_count (absent on a backfill — it lives on the
    in-memory scan result, nowhere else) and column roles, which tie for most tables
    and leave the top-20 decided by ALPHABETICAL order: source 2's card was then 20
    `accounts_*` tables, with `assets_asset` — the property entity most of its
    questions are about — never appearing at all.

    In-degree is the schema's own statement of what is central: on this deployment it
    ranks users_user (278) and assets_asset (90) far above any accounts_* table. Read
    from the graph store's existing fk_to/discovered_fk edges; no new computation.
    Best-effort — {} simply falls back to the previous ordering.
    """
    try:
        from ingestion.db_abstraction import internal_connection
        with internal_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dst.table_name, count(*)
                      FROM graph_edges e
                      JOIN (SELECT DISTINCT col_id, table_name
                              FROM column_embeddings_v2) dst
                        ON dst.col_id = replace(e.dst_node_id, 'col:', '')
                     WHERE e.edge_type IN ('fk_to', 'discovered_fk')
                     GROUP BY 1
                    """
                )
                return {t: int(n) for t, n in (cur.fetchall() or [])}
    except Exception as exc:                       # pragma: no cover
        logger.warning("routing_card: fk centrality unavailable (%s: %s)",
                       type(exc).__name__, exc)
        return {}


def _joins_to(source_id: str) -> List[Dict[str, Any]]:
    """The cross-source joins this source participates in, from the `cross_source_fk`
    edges L5 has just (re)discovered. Column ids are resolved to (table, column) names
    through column_embeddings_v2 so the card names joins the way a question would.

    This is the field that lets the router answer "can these two sources be combined at
    all" structurally, instead of inferring it from the wording of the question.
    """
    try:
        from ingestion.db_abstraction import internal_connection
        with internal_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.src_node_id, e.dst_node_id, e.attrs,
                           cs.table_name, cs.col_name, cd.table_name, cd.col_name
                      FROM graph_edges e
                      LEFT JOIN (SELECT DISTINCT col_id, table_name, col_name
                                   FROM column_embeddings_v2) cs
                        ON cs.col_id = replace(e.src_node_id, 'col:', '')
                      LEFT JOIN (SELECT DISTINCT col_id, table_name, col_name
                                   FROM column_embeddings_v2) cd
                        ON cd.col_id = replace(e.dst_node_id, 'col:', '')
                     WHERE e.edge_type = 'cross_source_fk'
                    """
                )
                rows = cur.fetchall() or []
    except Exception as exc:                       # pragma: no cover
        logger.warning("routing_card: cross-source edges unavailable (%s: %s)",
                       type(exc).__name__, exc)
        return []

    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _src, _dst, attrs, s_tbl, s_col, d_tbl, d_col in rows:
        try:
            a = json.loads(attrs) if isinstance(attrs, str) else (attrs or {})
        except (ValueError, TypeError):
            a = {}
        frm, to = str(a.get("from_source") or ""), str(a.get("to_source") or "")
        if str(source_id) not in (frm, to):
            continue
        # orient the edge so "from" is always THIS source
        if str(source_id) == frm:
            other, mine, theirs = to, (s_tbl, s_col), (d_tbl, d_col)
        else:
            other, mine, theirs = frm, (d_tbl, d_col), (s_tbl, s_col)
        if not other or not mine[0] or not theirs[0]:
            continue
        via = {"from_table": mine[0], "from_col": mine[1],
               "to_table": theirs[0], "to_col": theirs[1],
               "tier": a.get("tier")}
        if via not in by_source[other]:
            by_source[other].append(via)
    # HIGH-tier joins first — they are the ones a federated plan can rely on.
    return [{"source_id": sid,
             "via": sorted(v, key=lambda x: (x.get("tier") != "HIGH", x["from_table"]))[:6]}
            for sid, v in sorted(by_source.items())]


def _example_questions(entities: List[Dict[str, Any]]) -> List[str]:
    """Deterministic question TEMPLATES over this source's own key dimensions and
    measures — no LLM, no invention.

    These are not for a user to read; they are for the router to pattern-match against.
    A question's shape ("how many X per Y", "total M by D") is exactly the signal that
    distinguishes "this source can answer it" from "this source merely mentions the
    word", and the deterministic templates carry that shape without a generation call
    that could invent a dimension the source does not have.
    """
    out: List[str] = []
    for e in entities:
        name = _pluralise(e["business_name"].lower())
        dims, meas = e.get("key_dims") or [], e.get("key_measures") or []
        if e.get("chunk_count") and not dims and not meas:
            # a document entity: the shape of question it answers is "what does X say
            # about …", not an aggregate
            out.append(f"what does {e['business_name'].lower()} say about ...")
            if len(out) >= MAX_EXAMPLE_QUESTIONS:
                break
            continue
        if dims:
            out.append(f"how many {name} per {dims[0].lower()}")
        if meas and dims:
            out.append(f"total {meas[0].lower()} by {dims[0].lower()}")
        elif meas:
            out.append(f"total {meas[0].lower()} across all {name}")
        if len(out) >= MAX_EXAMPLE_QUESTIONS:
            break
    return out[:MAX_EXAMPLE_QUESTIONS]


def build_routing_card(ctx, state: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Build (and return) the routing card for the source `ctx` describes.

    Pure transform of what this ingestion already computed. Never raises on missing
    inputs — a source with no semantic model still gets a valid, mostly-empty card,
    which the router reads as "this source has no described entities" rather than
    crashing or silently having no card at all.
    """
    source_id = str(ctx.source_id)
    sm = state.get("semantic_model") or {}
    tables_meta = sm.get("tables") or {}
    cols_by_table = _columns_by_table(sm)

    # Row counts come from THIS run's scan result (the only place they exist — the
    # engine's table_metadata carries no row_count).
    row_counts: Dict[str, int] = {}
    scan = state.get("scan_result")
    for t in (getattr(scan, "tables", None) or []):
        try:
            row_counts[t.table_name] = int(t.row_count or 0)
        except (TypeError, ValueError):
            continue

    # Rank candidate entities: a real business entity, then by how much it can actually
    # answer (has measures / dimensions), then by size. Deterministic — no scoring model.
    centrality = _fk_centrality()
    candidates = []
    for tbl, meta in tables_meta.items():
        if not _is_entity_table(meta):
            continue
        cols = cols_by_table.get(tbl, [])
        dims = _role_names(cols, "DIMENSION", MAX_DIMS)
        meas = _role_names(cols, "MEASURE", MAX_MEASURES)
        candidates.append((tbl, meta, dims, meas, row_counts.get(tbl, 0),
                           centrality.get(tbl, 0)))
    # CENTRALITY first, then answerability (has a measure / has a dimension), then size,
    # then name for a stable tie-break. Answerability was tried first and was wrong: it
    # demotes every table with no MEASURE column below every table that has one, which
    # dropped `users_user` — the single most-referenced entity in source 2 (in-degree
    # 278) — off the card entirely, because "how many users" needs no measure. What a
    # router must know first is which entities a source is ABOUT; whether a given
    # question aggregates them is the query planner's problem, not the router's.
    candidates.sort(key=lambda c: (-c[5], -(len(c[3]) > 0), -(len(c[2]) > 0), -c[4], c[0]))
    chosen = candidates[:MAX_ENTITIES]

    samples = _sample_values(source_id, [c[0] for c in chosen])

    entities: List[Dict[str, Any]] = []
    for tbl, meta, dims, meas, rc, _cen in chosen:
        entities.append({
            "table": tbl,
            "business_name": _business_name(meta, tbl),
            "purpose": str((meta or {}).get("business_purpose") or "").strip() or None,
            "row_count": rc or None,
            "key_dims": dims,
            "key_measures": meas,
            "sample_values": samples.get(tbl, {}),
        })

    card = {
        "version": CARD_VERSION,
        "source_id": source_id,
        "tenant": str(getattr(ctx, "tenant", "default")),
        # Structural kind (relational / datalake / document / nosql) — known to
        # ingestion. The business `name`/`domain_tags`/`description` stay on the
        # registry row and are composed in at query time; see the module docstring.
        "kind": str(getattr(ctx, "type", "") or ""),
        "generated_at": _now(),
        "entity_count_total": len(tables_meta),
        "entities": entities,
        "example_questions": _example_questions(entities),
        "joins_to": _joins_to(source_id),
    }
    if verbose:
        logger.info("routing_card: source=%s entities=%d/%d joins_to=%d",
                    source_id, len(entities), len(tables_meta), len(card["joins_to"]))
    return card


# semantic_type → analytics role, for sources that have no L3 semantic model. The L3
# model states `analytics_role` directly; the engine store only carries the coarser
# `semantic_type`, so it is mapped here. Same vocabulary, one level less resolved.
_STORE_MEASURE_TYPES = {"MONETARY", "METRIC", "QUANTITY", "NUMERIC", "MEASURE"}
_STORE_DIM_TYPES = {"CATEGORY", "STATUS", "TYPE", "LOCATION", "NAME", "FREE_TEXT", "DATE"}


def build_card_from_store(source_id, tenant: str = "default",
                          kind: str = "", verbose: bool = False) -> Dict[str, Any]:
    """A routing card for a source that has NO L3 semantic model.

    Only relational sources get the full L3 treatment. A datalake source (parquet
    tables) builds a "lite" model at QUERY time and never persists one, and a document
    source has documents rather than tables at all — so for both, the source of truth
    about what they hold is the engine store itself (column_embeddings_v2 / doc_chunks),
    which every source kind populates.

    This matters beyond tidiness: without it only source 2 would have a card, and the
    router would be comparing a rich description of one source against bare column
    matches for the others — a systematically biased comparison, and worse than giving
    none of them cards.
    """
    entities: List[Dict[str, Any]] = []
    resolved_kind = kind
    try:
        from ingestion.db_abstraction import internal_connection
        with internal_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT table_name, col_name, semantic_type
                         FROM column_embeddings_v2 WHERE source_id = %s
                        ORDER BY table_name, col_name""",
                    (str(source_id),))
                col_rows = cur.fetchall() or []
                cur.execute(
                    """SELECT doc_name, count(*) FROM doc_chunks WHERE source_id = %s
                        GROUP BY 1 ORDER BY 2 DESC""",
                    (str(source_id),))
                doc_rows = cur.fetchall() or []
    except Exception as exc:
        logger.warning("routing_card: store read failed for source %s (%s: %s)",
                       source_id, type(exc).__name__, exc)
        col_rows, doc_rows = [], []

    by_table: Dict[str, List[tuple]] = defaultdict(list)
    for tbl, col, stype in col_rows:
        by_table[tbl].append((col, str(stype or "").upper()))

    if by_table:
        resolved_kind = resolved_kind or "datalake"
        samples = _sample_values(str(source_id), list(by_table))
        for tbl, cols in list(by_table.items())[:MAX_ENTITIES]:
            dims = [c for c, t in cols if t in _STORE_DIM_TYPES][:MAX_DIMS]
            meas = [c for c, t in cols if t in _STORE_MEASURE_TYPES][:MAX_MEASURES]
            entities.append({
                "table": tbl,
                # no L3 naming for these sources — the table name IS the business name
                # (a parquet dataset is named by whoever produced it, e.g. "vendors")
                "business_name": tbl.replace("_", " "),
                "purpose": None,
                "row_count": None,
                "key_dims": dims,
                "key_measures": meas,
                "sample_values": samples.get(tbl, {}),
            })
    elif doc_rows:
        resolved_kind = resolved_kind or "document"
        for doc_name, n_chunks in doc_rows[:MAX_ENTITIES]:
            entities.append({
                "table": doc_name,
                "business_name": str(doc_name).rsplit(".", 1)[0].replace("_", " "),
                "purpose": None,
                "row_count": None,
                "chunk_count": int(n_chunks),
                "key_dims": [],
                "key_measures": [],
                "sample_values": {},
            })

    return {
        "version": CARD_VERSION,
        "source_id": str(source_id),
        "tenant": str(tenant),
        "kind": resolved_kind,
        "generated_at": _now(),
        "entity_count_total": len(by_table) or len(doc_rows),
        "built_from": "engine_store",     # vs the L3 semantic model
        "entities": entities,
        "example_questions": _example_questions(entities),
        "joins_to": _joins_to(str(source_id)),
    }


def write_routing_card(ctx, state: Dict[str, Any], verbose: bool = False) -> str:
    """Build + persist the card at the per-(tenant, source) artifact path. Returns the
    path written. Uses `source_artifact_path` (never the flat path) — this is a NEW
    per-source derived artifact, and config.py's own guidance is that new artifacts use
    the scoped path so two sources can never shadow each other."""
    from config import source_artifact_path

    _tenant = getattr(ctx, "tenant", "default")
    if (state.get("semantic_model") or {}).get("tables"):
        card = build_routing_card(ctx, state, verbose=verbose)
    else:
        # datalake / document / any source that never persists an L3 model
        card = build_card_from_store(ctx.source_id, _tenant,
                                     kind=str(getattr(ctx, "type", "") or ""),
                                     verbose=verbose)
    path = source_artifact_path(CARD_ARTIFACT, ctx.source_id, _tenant)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(card, f, indent=2, default=str)
    os.replace(tmp, path)          # atomic: a reader never sees a half-written card
    return path


def render_card(card: Dict[str, Any], profile: Optional[Dict[str, Any]] = None,
                max_entities: int = 8) -> str:
    """The card as PROMPT TEXT — the compact projection, not the stored artifact.

    The stored card is deliberately rich (20 entities, 6 dims + 6 measures each, real
    sample values) because it is also an operator-readable description of the source
    and the input to the routing eval. Serialised whole it is ~2.3k tokens, and the
    routing prompt's budget is ~1.5k for FIVE sources — so what reaches the model is
    this projection: the top entities by the same deterministic ranking, two dims and
    one measure each, and a handful of sample values only where they disambiguate.

    `profile` is the live registry row (name / description / domain_tags) from
    `context.current_source_profiles()`. It is composed in HERE rather than stored in
    the card, so a description edited in the admin takes effect on the next query
    instead of on the next ingestion (see the module docstring).
    """
    p = profile or {}
    sid = card.get("source_id")
    head = f"- source_id={sid}"
    kind = p.get("source_type") or card.get("kind") or ""
    if kind:
        head += f" kind={kind}"
    tags = list(p.get("domain_tags") or [])
    if tags:
        head += f" domains={','.join(str(t) for t in tags)}"
    lines = [head]
    if p.get("name"):
        lines.append(f"    name: {p['name']}")
    if p.get("description"):
        lines.append(f"    description: {str(p['description'])[:300]}")

    for e in (card.get("entities") or [])[:max_entities]:
        bits = [f"    entity: {e.get('business_name') or e.get('table')} ({e.get('table')})"]
        if e.get("row_count"):
            bits.append(f"~{e['row_count']:,} rows")
        dims = (e.get("key_dims") or [])[:2]
        meas = (e.get("key_measures") or [])[:1]
        if dims:
            bits.append("by " + ", ".join(dims))
        if meas:
            bits.append("measures " + ", ".join(meas))
        lines.append(" — ".join(bits) if len(bits) > 1 else bits[0])
        sv = e.get("sample_values") or {}
        if sv:
            col, vals = next(iter(sv.items()))
            if vals:
                lines.append(f"        e.g. {col}: {', '.join(str(v) for v in vals[:3])}")

    qs = (card.get("example_questions") or [])[:3]
    if qs:
        lines.append("    answers questions like: " + "; ".join(qs))
    for j in (card.get("joins_to") or []):
        via = (j.get("via") or [])[:2]
        if via:
            joins = ", ".join(f"{v['from_table']}.{v['from_col']}={v['to_table']}.{v['to_col']}"
                              for v in via)
            lines.append(f"    joins to source {j['source_id']} via {joins}")
    return "\n".join(lines)


def load_routing_card(source_id, tenant: str = "default") -> Optional[Dict[str, Any]]:
    """Read one source's card, or None when it has not been ingested since cards
    existed. Callers MUST treat None as "no card" and fall back to their previous
    behaviour — a missing card is the normal state for a source ingested before this
    stage landed, not an error."""
    try:
        from config import source_artifact_path
        path = source_artifact_path(CARD_ARTIFACT, source_id, tenant)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
