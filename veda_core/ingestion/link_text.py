"""ingestion/link_text.py — real sentences for link tables and foreign keys.

A table that only CONNECTS two entities ("worklists_ticketuser": a ticket and the user it
is assigned to) used to reach the embedding stores as its own name plus a column list, and
its card as the business name 'ticketuser' with no aliases — so a question about
"assigned users" or "subscribers" never landed on it. Everything here is deterministic,
derived from the relationship graph, the column names and the other tables' names:

  link_tables(...)      which tables are links, their two endpoints, the verbs their FK
                        names imply ("assigned_by_id" → assigned), aliases
                        ("assigned", "assignee", "ticket assignment", "subscription"…)
                        and the card's one_row_is ("links a ticket to a user")
  compound_name(...)    'ticketuser' → 'ticket user', 'leaselistinglead' → 'lease
                        listing lead' — split on the other tables' names
  fk_phrases(...)       per FK column: the endpoint's business name + the relationship
                        ("the user this ticket is assigned to", "the user who created
                        this ticket")
  table_sentence / column_sentence   the text appended to the embedded passages

Used by ingestion/vocabulary.py (cards, L5), ingestion/biencoder.py + sparse_index.py
(passages, L4), ingestion/rerank_docs.py, and scripts/refresh_link_semantics.py (the
targeted re-embed of an already-ingested source).

A link table: ≥ 2 non-self FK endpoints on DISTINCT entity tables, and FKs make up
≥ 65 % of its columns once the id and the timestamps are set aside (a link row is its
keys plus bookkeeping). The rule's stated 70 % over ALL columns misses exactly the tables
that failed (worklists_ticketuser is 7/15 FKs — its timestamps and flags are not
content), hence the denominator.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

LINK_FK_SHARE = 0.65
_AUDIT_VERBS = {"created", "updated", "modified", "deleted", "changed", "edited"}
_TEMPORAL = "TEMPORAL"
# relationship words an FK column name carries ("<verb>_by_id", "<verb>_to_id")
_VERB_RE = re.compile(r"^([a-z]+?)_(by|to|for)(?:_[a-z]+)?_id$")
_SUBSCRIBABLE = {"plan", "pricing", "package", "subscription", "bundle", "membership"}
_LOG_SUFFIX = {"activity", "activities", "update", "updates", "history", "log", "logs", "event", "events"}


def _bare_phrase(table: str) -> str:
    parts = [p for p in str(table).lower().split("_") if p]
    return "".join(parts[1:]) if len(parts) >= 2 else "".join(parts)


def _words(s: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", str(s or "").lower()) if w]


def _stype(meta: dict) -> str:
    return str((meta or {}).get("semantic_type") or "").upper()


def _cols(sm, table) -> Dict[str, dict]:
    return {k.split(".", 1)[1]: (v or {}) for k, v in ((sm or {}).get("columns") or {}).items()
            if k.split(".", 1)[0] == table}


def _fk_edges(graph) -> List[dict]:
    return [e for e in (graph or {}).get("edges") or []
            if e.get("discovery") == "declared_fk" and not e.get("polymorphic")
            and e.get("target_column") == "id" and e.get("cardinality") in ("N:1", "1:1", None)]


# ── names ────────────────────────────────────────────────────────────────────────────
def compound_name(table: str, known: Dict[str, str]) -> Optional[str]:
    """Split a run-together table phrase on the names of OTHER tables of the source.
    `known`: bare phrase ('ticket', 'leaselisting') → business name ('ticket',
    'lease listing'). 'ticketuser' → 'ticket user'. None when nothing splits."""
    p = _bare_phrase(table)
    if not p or len(p) < 7:
        return None

    def split(s: str, depth: int = 0) -> Optional[List[str]]:
        if not s:
            return []
        if depth > 3:
            return None
        best = None
        for q in sorted(known, key=len, reverse=True):
            if len(q) >= 4 and s.startswith(q) and q != p:
                rest = split(s[len(q):], depth + 1)
                if rest is not None:
                    best = [known[q]] + rest
                    break
        if best is None and depth > 0 and len(s) >= 3:
            best = [s]                                       # a trailing word ('lead')
        return best
    parts = split(p)
    if not parts or len(parts) < 2:
        return None
    return " ".join(parts)


def known_names(cards: Dict[str, dict], sm) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for t in ((sm or {}).get("tables") or {}):
        b = _bare_phrase(t)
        c = (cards or {}).get(t) or {}
        nm = c.get("business_name") if c.get("drafted_by") not in (None, "deterministic") else None
        out.setdefault(b, nm or b)
    return out


# ── link tables ──────────────────────────────────────────────────────────────────────
def link_tables(sm, graph, cards: Optional[Dict[str, dict]] = None) -> Dict[str, Dict[str, Any]]:
    """{table: {a, a_name, b, b_name, verbs, aliases, one_row_is, business_name}}."""
    cards = cards or {}
    edges = _fk_edges(graph)
    by_src: Dict[str, List[dict]] = {}
    for e in edges:
        by_src.setdefault(e["source_table"], []).append(e)
    known = known_names(cards, sm)
    out: Dict[str, Dict[str, Any]] = {}
    for t in ((sm or {}).get("tables") or {}):
        cols = _cols(sm, t)
        content = [c for c, m in cols.items() if c != "id" and _stype(m) != _TEMPORAL]
        fk_cols = {e["source_column"] for e in by_src.get(t, [])}      # self-references are keys too
        fks = [e for e in by_src.get(t, []) if e["target_table"] != t]
        if not content or not fks:
            continue
        share = len([c for c in content if c in fk_cols]) / len(content)
        ents = [e for e in fks if e.get("relationship_type") != "audit"
                and not _is_audit_col(e["source_column"])]
        targets = list(dict.fromkeys(e["target_table"] for e in ents))
        if len(targets) < 2 or share < LINK_FK_SHARE:
            continue
        if any(c in ("name", "title", "label") or c.endswith(("_name", "_title")) for c in content
               if c not in fk_cols):
            continue                                  # a row with its own name is an entity
        # endpoints: the targets the table's OWN name mentions (longest match first), then
        # entities before lookup lists (list_of_values_*, generics_*)
        tp = _bare_phrase(t)

        def mention(x):
            bx = _bare_phrase(x)
            full = bx in tp
            k = len(bx) if full else max((n for n in range(4, len(bx) + 1) if bx[:n] in tp), default=0)
            lookup = x.startswith(("list_of_values", "generics_")) or "listofvalue" in bx
            return (not full, -k, lookup, tp.find(bx[:k]) if k else 99)
        targets.sort(key=mention)
        a, b = targets[0], targets[1]
        an, bn = _name(a, cards, known), _name(b, cards, known)
        verbs = []
        for e in fks:
            m = _VERB_RE.match(e["source_column"])
            if m and m.group(1) not in _AUDIT_VERBS and m.group(1) not in verbs:
                verbs.append(m.group(1))
        aliases: List[str] = []
        for v in verbs:
            aliases += [v, _agent_noun(v), f"{v} {bn}", f"{an} {_noun(v)}", f"{bn} {v} to {an}"]
        sub_b = bool((set(_words(bn)) | set(_words(b))) & _SUBSCRIBABLE)
        sub_a = bool((set(_words(an)) | set(_words(a))) & _SUBSCRIBABLE)
        if (sub_a or sub_b) and not verbs:
            aliases += ["subscription", "subscriber", "subscribers", "subscribed plan"]
            verbs = ["subscribed"]
            if sub_a and not sub_b:                     # the subscriber first, the plan second
                a, b, an, bn = b, a, bn, an
        if set(re.findall(r"[a-z]+", tp)) & _LOG_SUFFIX or any(tp.endswith(s) for s in _LOG_SUFFIX):
            aliases += ["activity", "update", "history"]
        verb = verbs[0] if verbs else None
        one = (f"links a {an} to the {bn} they subscribed to" if verb == "subscribed"
               else f"links a {an} to the {bn} it is {verb} to" if verb
               else f"links a {an} to a {bn}")
        cname = compound_name(t, known)
        out[t] = {"a": a, "a_name": an, "b": b, "b_name": bn, "verbs": verbs,
                  "aliases": list(dict.fromkeys(x for x in aliases if x)),
                  "one_row_is": one, "business_name": cname or f"{an} {bn} link"}
    return out


def _is_audit_col(c: str) -> bool:
    m = _VERB_RE.match(c)
    return bool(m and m.group(1) in _AUDIT_VERBS)


def _name(t: str, cards: Dict[str, dict], known: Dict[str, str]) -> str:
    c = cards.get(t) or {}
    if c.get("business_name") and c.get("drafted_by") != "deterministic":
        return c["business_name"]
    return compound_name(t, known) or known.get(_bare_phrase(t)) or _bare_phrase(t)


def _agent_noun(v: str) -> str:
    return {"assigned": "assignee", "subscribed": "subscriber", "raised": "raiser",
            "approved": "approver", "owned": "owner", "managed": "manager",
            "reviewed": "reviewer", "requested": "requester", "paid": "payer",
            "handled": "handler", "closed": "closer"}.get(v, v.rstrip("ed") + "er" if v.endswith("ed") else v)


def _noun(v: str) -> str:
    return {"assigned": "assignment", "subscribed": "subscription", "approved": "approval",
            "requested": "request", "reviewed": "review", "paid": "payment"}.get(v, v)


# ── FK phrases ───────────────────────────────────────────────────────────────────────
def fk_phrases(sm, graph, cards: Optional[Dict[str, dict]] = None,
               links: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, str]:
    """{"table.col": "the user this ticket is assigned to"} for every declared FK column."""
    cards = cards or {}
    known = known_names(cards, sm)
    links = links if links is not None else link_tables(sm, graph, cards)
    out: Dict[str, str] = {}
    for e in _fk_edges(graph):
        t, c, tgt = e["source_table"], e["source_column"], e["target_table"]
        rn = links[t]["business_name"] if t in links else _name(t, cards, known)
        en = _name(tgt, cards, known)
        m = _VERB_RE.match(c)
        if t in links and not m:
            lk = links[t]
            v = lk["verbs"][0] if lk["verbs"] else None
            if v == "subscribed" and tgt == lk["a"]:
                ph = f"the {en} who subscribed to this {lk['b_name']}"
            elif v == "subscribed" and tgt == lk["b"]:
                ph = f"the {en} this {lk['a_name']} subscribed to"
            elif tgt == lk["b"] and v:
                ph = f"the {en} this {lk['a_name']} is {v} to"
            elif tgt == lk["a"] and v:
                ph = f"the {en} {v} to this {lk['b_name']}"
            else:
                ph = f"the {en} this {rn} links to"
        elif m:
            verb, prep = m.group(1), m.group(2)
            ph = (f"the {en} who {verb} this {rn}" if prep == "by"
                  else f"the {en} this {rn} is {verb} {prep}")
        else:
            skip = set(_words(en)) | set(_words(tgt.replace("_", " "))) | {_bare_phrase(tgt)}
            qual = [w for w in _words(c[:-3] if c.endswith("_id") else c) if w not in skip]
            ph = (f"the {' '.join(qual)} {en} of this {rn}" if qual else f"the {en} this {rn} belongs to")
        out[f"{t}.{c}"] = re.sub(r"\s+", " ", ph).strip()
    return out


def column_sentence(key: str, phrases: Dict[str, str], target: Optional[str] = None) -> str:
    ph = phrases.get(key)
    return f"RELATIONSHIP: {ph}" if ph else ""


def table_sentence(table: str, links: Dict[str, Dict[str, Any]], card: Optional[dict] = None) -> str:
    lk = links.get(table)
    if lk:
        return (f"LINK TABLE: {lk['one_row_is']}; also called "
                + ", ".join([lk["business_name"]] + lk["aliases"][:8]))
    if card and card.get("business_name") and card.get("_renamed"):
        return f"also called {card['business_name']}"
    return ""


def for_source(source_id, tenant: str = "default", sm=None, cards=None):
    """(links, phrases) for an ingested source — the artifacts on disk."""
    from ingestion.vocabulary import _graph, _read_json
    if sm is None:
        from config import resolve_source_artifact
        sm = _read_json(resolve_source_artifact("veda_semantic_model.json", source_id, tenant)) or {}
    if cards is None:
        from config import source_artifact_path
        cards = _read_json(source_artifact_path("veda_entity_cards.json", source_id, tenant)) or {}
        cards = {k: v for k, v in cards.items() if not str(k).startswith("_")}
    g = _graph(source_id, tenant)
    links = link_tables(sm, g, cards)
    return links, fk_phrases(sm, g, cards, links)
