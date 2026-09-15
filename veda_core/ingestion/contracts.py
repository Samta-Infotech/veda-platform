"""Ingestion layer contracts (Track 2 / P4).

Defines the explicit boundary every layered stage speaks: a ``SourceContext``
(who is being ingested + where artifacts are scoped) and a ``StageOutcome``
(what a stage produced + whether it is fatal). The layer modules
(``layers/l1_extract`` … ``layers/l5_publish``) are thin, importable, individually
testable wrappers over the EXISTING stage functions — the logic is preserved
verbatim (P4 is "a move, not a rewrite"); only the I/O boundary is made explicit.

Nothing here opens a DB connection or invents a path. Connection + artifact
scope are resolved once, here, from the injected source (config.get_source(),
which the ingesting worker populated from the DB Source row — §3.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os


@dataclass
class SourceContext:
    """The one source being ingested, resolved from the injected env source.

    Per-source derived-artifact paths are resolved on demand via
    `config.source_artifact_path(name, ctx.source_id, ctx.tenant)` — there is no
    ambient "artifact scope" carried on this context (P0-7, 2026-09-11: removed
    the `artifact_scope` field/`VEDA_ARTIFACT_SCOPE` env plumbing it depended on;
    it was a second, always-inert scoping mechanism beside the one that's
    actually used everywhere now — see docs/backlog/query-engine-open-items.md).
    """

    source_id: str
    tenant: str = "default"
    type: str = "relational"
    engine: str = "postgresql"
    connection: Dict[str, Any] = field(default_factory=dict)
    exclude_tables: List[str] = field(default_factory=list)
    schema_filter: Optional[str] = None
    skip_llm: bool = False
    resume: bool = False

    @classmethod
    def from_env(cls, skip_llm: bool = False, resume: bool = False) -> "SourceContext":
        """Build from the injected source (config.get_source) + scope/tenant env.

        This is the single place the engine learns *which* source it is running
        for — no config-file registry, no re-derivation of "primary" (§3.1)."""
        from config import get_source

        src = get_source()
        conn = {
            "engine": src.get("engine", "postgresql"),
            "host": src.get("host"), "port": src.get("port"),
            "dbname": src.get("dbname"), "user": src.get("user"),
            "password": src.get("password"),
            # File-backed tabular/document sources carry a path instead of a DSN
            # (Cross-source plan P2.1); harmless/empty for relational sources.
            "path": src.get("path") or src.get("source_path"),
        }
        return cls(
            source_id=str(src.get("id", "primary_db")),
            tenant=os.environ.get("VEDA_TENANT", "default"),
            type=src.get("type", "relational"),
            engine=src.get("engine", "postgresql"),
            connection=conn,
            exclude_tables=list(src.get("exclude_tables", [])),
            schema_filter=src.get("schema"),
            skip_llm=skip_llm or os.environ.get("VEDA_SKIP_LLM") == "1",
            resume=resume or os.environ.get("VEDA_RESUME") == "1",
        )


@dataclass
class StageOutcome:
    """Result of one stage. ``fatal`` marks a stage whose failure must abort the
    pipeline (matches run_ingestion's per-stage raise-vs-continue semantics)."""

    name: str
    ok: bool
    fatal: bool = False
    detail: str = ""
    error: Optional[str] = None
