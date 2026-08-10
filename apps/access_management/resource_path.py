"""Canonical resource paths — the addressing scheme every grant is written against.

Implements ADR-0001 (``docs/adr/0001-rbac-resource-path.md``). Read it before changing
anything here: this string appears in every grant row, every resolver result, every
cache key and every gate decision, so a change to its shape is a data migration across
all of them.

    <kind>:<source>[:<segment>]*

    db:crm_postgres                     the whole source
    db:crm_postgres:employee            one table
    db:crm_postgres:employee:salary     one column

DELIBERATELY PURE
    No Django imports, no model imports, no I/O. Everything here is a total function
    over strings, which is what lets the models, the discovery service, the resolver
    and the gates all share one definition without an import cycle — and what makes it
    exhaustively testable without a database.

    The dialect→kind map is therefore keyed by plain strings rather than importing
    ``sources.Source.Dialect``. A test asserts the two stay in step, so the decoupling
    cannot silently drift.

WHY SEGMENT-BOUNDARY MATCHING MATTERS
    ``db:crm`` must NOT match ``db:crm_postgres``. A naive ``startswith`` grants every
    source whose name begins with another's — the classic prefix-authorization bug.
    ``is_prefix_of`` compares segment tuples, never raw strings.
"""
from __future__ import annotations

import re

SEPARATOR = ":"

#: A path always names at least a kind and a source; ``db`` alone is not a resource.
MIN_SEGMENTS = 2
#: Bounds the resolver's prefix expansion (``prefixes()`` is O(depth)).
MAX_SEGMENTS = 8
#: Matches the ``CatalogResource.path`` column width.
MAX_LENGTH = 512

#: Lowercase alphanumerics plus the three separators real table and file names use.
#: Anything else — whitespace, ``:``, unicode — is rejected rather than escaped: an
#: escaping scheme would make every comparison in the resolver and every cache key
#: subtly wrong (ADR §3.6).
_SEGMENT_RE = re.compile(r"^[a-z0-9_.\-]+$")

# --- kinds -----------------------------------------------------------------
# Coarse families, derived from Source.dialect rather than invented alongside it.
# Two independent taxonomies for "what sort of thing is this" would drift the first
# time a dialect is added (ADR §3.2).

KIND_DB = "db"
KIND_NOSQL = "nosql"
KIND_FILES = "files"
KIND_LAKE = "lake"

KIND_BY_DIALECT = {
    # relational
    "postgres": KIND_DB,
    "mysql": KIND_DB,
    "sqlite": KIND_DB,
    "oracle": KIND_DB,
    "sqlserver": KIND_DB,
    "duckdb": KIND_DB,
    # non-relational
    "mongo": KIND_NOSQL,
    "es": KIND_NOSQL,
    "dynamo": KIND_NOSQL,
    # documents
    "filesystem": KIND_FILES,
    "s3_docs": KIND_FILES,
    # data lake
    "delta": KIND_LAKE,
    "parquet": KIND_LAKE,
    "csv_lake": KIND_LAKE,
    "iceberg": KIND_LAKE,
}


class InvalidResourcePath(ValueError):
    """A path is not expressible in the canonical grammar.

    Raised rather than silently coerced. A path that cannot be represented must fail
    loudly at write time: a half-canonicalised path would be granted under one string
    and checked under another, which is a silent authorization hole.
    """


class UnknownDialect(ValueError):
    """A ``Source.dialect`` with no mapped kind.

    Means a dialect was added to ``sources`` without extending ``KIND_BY_DIALECT``.
    Fails closed: an unmappable source is unaddressable, so nothing can be granted on
    it — rather than guessing a kind and granting against the wrong namespace.
    """


def kind_for_dialect(dialect: str) -> str:
    """The resource kind for a ``Source.dialect``.

    Raises:
        UnknownDialect: the dialect has no mapping. Deliberately not a default.
    """
    try:
        return KIND_BY_DIALECT[(dialect or "").strip().lower()]
    except KeyError as exc:
        raise UnknownDialect(
            f"no resource kind mapped for dialect {dialect!r}; "
            f"add it to KIND_BY_DIALECT (see ADR-0001 §3.2)") from exc


def normalize_segment(value: str) -> str:
    """One segment, canonicalised — lowercased and trimmed.

    Raises:
        InvalidResourcePath: blank, or containing anything outside the charset.
    """
    segment = (value or "").strip().lower()
    if not segment:
        raise InvalidResourcePath("resource path segments may not be blank")
    if not _SEGMENT_RE.match(segment):
        raise InvalidResourcePath(
            f"segment {value!r} contains characters that cannot appear in a resource "
            f"path (allowed: lowercase letters, digits, '_', '.', '-')")
    return segment


def build(*parts: str) -> str:
    """A canonical path from its segments.

    ``build("db", "CRM_Postgres", "Employee")`` -> ``"db:crm_postgres:employee"``.

    Raises:
        InvalidResourcePath: on a bad segment, or a path outside the size bounds.
    """
    segments = [normalize_segment(part) for part in parts]
    _check_shape(segments)
    path = SEPARATOR.join(segments)
    if len(path) > MAX_LENGTH:
        raise InvalidResourcePath(
            f"resource path exceeds {MAX_LENGTH} characters ({len(path)})")
    return path


def validate(path: str) -> str:
    """Parse a path and return it in canonical form.

    The single entry point for untrusted input — a stored or submitted path goes
    through here before it is compared against anything.

    Raises:
        InvalidResourcePath: if the path is not expressible.
    """
    if not isinstance(path, str):
        raise InvalidResourcePath("resource path must be a string")
    return build(*path.split(SEPARATOR))


def segments(path: str) -> tuple[str, ...]:
    """The canonical path's segments."""
    return tuple(validate(path).split(SEPARATOR))


def kind_of(path: str) -> str:
    """The path's kind — its first segment."""
    return segments(path)[0]


def source_of(path: str) -> str:
    """The path's source name — its second segment."""
    return segments(path)[1]


def parent(path: str) -> str | None:
    """The immediate parent path, or None when the path names a whole source.

    ``db:crm:employee:salary`` -> ``db:crm:employee`` -> ``db:crm`` -> ``None``.
    A source is the root of its own tree; there is no ``db``-only resource.
    """
    parts = segments(path)
    if len(parts) <= MIN_SEGMENTS:
        return None
    return SEPARATOR.join(parts[:-1])


def prefixes(path: str) -> list[str]:
    """Every path that is a prefix-or-equal of this one, broadest first.

    ``db:crm:employee:salary`` ->
        ``["db:crm", "db:crm:employee", "db:crm:employee:salary"]``

    This is the set a resolver matches grants against (ADR §3.4/§3.5): a grant on any
    of these covers the requested resource. Starts at ``MIN_SEGMENTS`` because a bare
    kind is not a grantable resource — otherwise a grant on ``db`` would silently
    cover every database source in the platform.
    """
    parts = segments(path)
    return [SEPARATOR.join(parts[:i]) for i in range(MIN_SEGMENTS, len(parts) + 1)]


def is_prefix_of(ancestor: str, descendant: str) -> bool:
    """Whether ``ancestor`` covers ``descendant`` under prefix inheritance.

    Segment-wise, never string-wise::

        is_prefix_of("db:crm", "db:crm:employee")     -> True
        is_prefix_of("db:crm", "db:crm")              -> True   (prefix-or-equal)
        is_prefix_of("db:crm", "db:crm_postgres")     -> False  <- the bug this prevents
        is_prefix_of("db:crm:employee", "db:crm")     -> False

    A string ``startswith`` would return True for the third case and grant every
    source whose name merely begins with another's.
    """
    ancestor_parts = segments(ancestor)
    descendant_parts = segments(descendant)
    if len(ancestor_parts) > len(descendant_parts):
        return False
    return descendant_parts[:len(ancestor_parts)] == ancestor_parts


def _check_shape(segments_: list[str]) -> None:
    if len(segments_) < MIN_SEGMENTS:
        raise InvalidResourcePath(
            f"a resource path needs at least {MIN_SEGMENTS} segments "
            f"(<kind>:<source>), got {len(segments_)}")
    if len(segments_) > MAX_SEGMENTS:
        raise InvalidResourcePath(
            f"a resource path may have at most {MAX_SEGMENTS} segments, "
            f"got {len(segments_)}")
    if segments_[0] not in set(KIND_BY_DIALECT.values()):
        raise InvalidResourcePath(
            f"unknown resource kind {segments_[0]!r}; "
            f"expected one of {sorted(set(KIND_BY_DIALECT.values()))}")
