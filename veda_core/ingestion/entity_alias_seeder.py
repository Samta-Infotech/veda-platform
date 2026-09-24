"""L5 stage — publish the tracked entity-alias SEED into the per-source artifact.

`veda_entity_aliases.json` maps an everyday business noun to the canonical table it
names ("property" -> assets_asset). It is the ONLY evidence the engine has for nouns L3
never emits: `primary_entity` for `assets_asset` is just "Asset", so nothing in the
derived model connects the word a user actually types to that table.

It is also hand-curated, and it used to exist ONLY as a derived-looking artifact under
`veda_core/data/`, which `.gitignore` excludes wholesale and which every reingest moves
aside (`scripts/seed_and_reingest.sh`'s artifact-reset list names it explicitly). On
2026-09-22 that combination deleted it — `veda_entity_aliases.json.bak-20260922-140816`
is what was left — and with the glossary empty:

  * the bare-count anchor gate refused "how many properties are there" (the per-source
    battery's src2/src3 failures), and
  * `property` was offered to the user as a missing VALUE, producing the
    "'['property']' doesn't match any value in this data" clarify that hit 5 of the 20
    questions in the 2026-09-23 `question.txt` run.

So the curated data now lives in source control at
`veda_core/data/seeds/<source_id>/veda_entity_aliases.seed.json` and this stage copies
it into the artifact the readers already use. A reingest may still delete the artifact;
the next L5 puts it back, and the seed itself can no longer be lost.

Merge policy: the SEED WINS on conflict. Anything already in the artifact that the seed
does not mention is preserved, so an operator's out-of-band addition survives, but the
tracked, reviewed file is authoritative for the keys it defines.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

ARTIFACT = "veda_entity_aliases.json"
SEED_NAME = "veda_entity_aliases.seed.json"


def seed_dir() -> str:
    """`veda_core/data/seeds` — resolved from THIS file, so it is correct in the
    container (`/app/veda_core/...`) and on the host alike."""
    here = os.path.dirname(os.path.abspath(__file__))          # .../veda_core/ingestion
    return os.path.join(os.path.dirname(here), "data", "seeds")


def seed_path(source_id) -> str:
    return os.path.join(seed_dir(), str(source_id), SEED_NAME)


def load_seed(source_id) -> Dict[str, str]:
    """The tracked seed for this source, `_`-prefixed doc keys stripped. {} if absent."""
    p = seed_path(source_id)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            raw = json.load(f)
        return {str(k).lower(): v for k, v in raw.items() if not str(k).startswith("_")}
    except Exception:
        return {}


def publish(source_id, tenant: str = "default",
            known_tables: Optional[set] = None) -> Tuple[Optional[str], int, int]:
    """Merge the seed into the per-source artifact.

    `known_tables`, when given, drops any alias whose target table is not in this
    source's schema — a seed entry that names a table this source does not have is a
    stale line, and publishing it would make the resolver ground a noun onto nothing.

    Returns (path_written | None, n_from_seed, n_dropped)."""
    seed = load_seed(source_id)
    if not seed:
        return None, 0, 0

    dropped = 0
    if known_tables:
        kept = {}
        for k, v in seed.items():
            if v in known_tables:
                kept[k] = v
            else:
                dropped += 1
        seed = kept

    from config import source_artifact_path
    path = source_artifact_path(ARTIFACT, source_id, tenant)

    existing: Dict[str, str] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = {k: v for k, v in json.load(f).items()
                            if not str(k).startswith("_")}
        except Exception:
            existing = {}

    merged = {**existing, **seed}            # seed wins; operator extras preserved
    merged["_about"] = (f"Business-noun -> table aliases for source {source_id}. "
                        f"Published by ingestion/entity_alias_seeder.py from the tracked "
                        f"seed at data/seeds/{source_id}/{SEED_NAME}; edit the SEED, not "
                        f"this file (a reingest deletes this one).")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    os.replace(tmp, path)                    # atomic: no reader sees a half-written map
    return path, len(seed), dropped


def publish_from_state(ctx, state=None, verbose: bool = False):
    """L5 entry point. Derives the source's real table set from this run's semantic
    model when there is one, so a stale seed line cannot publish a dangling alias."""
    tables = None
    try:
        sm = (state or {}).get("semantic_model") or {}
        if sm.get("tables"):
            tables = set(sm["tables"].keys())
    except Exception:
        tables = None
    return publish(ctx.source_id, getattr(ctx, "tenant", "default"), known_tables=tables)


# ── value aliases (D.5) ───────────────────────────────────────────────────────────────
VALUE_ARTIFACT = "veda_value_aliases.json"
VALUE_SEED_NAME = "veda_value_aliases.seed.json"


def publish_values(source_id, tenant: str = "default",
                   known_tables: Optional[set] = None) -> Tuple[Optional[str], int]:
    """Same seed->artifact publish for the business-PHRASE -> column-VALUE glossary
    (`query/value_glossary`). Keyed "table.column", so a stale line naming a table this
    source does not have is dropped rather than published."""
    sp = os.path.join(seed_dir(), str(source_id), VALUE_SEED_NAME)
    if not os.path.exists(sp):
        return None, 0
    try:
        with open(sp) as f:
            seed = {k: v for k, v in json.load(f).items() if not str(k).startswith("_")}
    except Exception:
        return None, 0
    if known_tables:
        seed = {k: v for k, v in seed.items() if str(k).split(".")[0] in known_tables}
    if not seed:
        return None, 0

    from config import source_artifact_path
    path = source_artifact_path(VALUE_ARTIFACT, source_id, tenant)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = {k: v for k, v in json.load(f).items()
                            if not str(k).startswith("_")}
        except Exception:
            existing = {}
    merged = {**existing, **seed}
    merged["_about"] = (f"Business phrase -> column value map for source {source_id}. "
                        f"Published by ingestion/entity_alias_seeder.publish_values from "
                        f"data/seeds/{source_id}/{VALUE_SEED_NAME} — edit the SEED.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path, len(seed)
