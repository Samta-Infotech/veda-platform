# =============================================================================
# connectors/base.py
# VEDA — Universal Connector Interface
#
# Defines the abstract base class every connector must implement.
# The ingestion dispatcher and query router work against this interface only —
# they never import a concrete connector directly.
#
# Design principles:
#   - One interface, four implementations (relational, document, datalake, nosql)
#   - Each method is independently optional — connectors declare capability via
#     supports_* properties rather than raising NotImplementedError everywhere
#   - All methods are synchronous — async is the caller's responsibility
#   - All methods return typed dataclasses — no raw dicts passed between layers
#   - Every connector is stateless after connect() — no session state held
#
# Connector lifecycle:
#   1. Instantiate with source config dict from VEDA_SOURCES
#   2. Call connect() — verifies credentials, returns ConnectorStatus
#   3. Call any supported method (get_schema, get_chunks, etc.)
#   4. Call disconnect() when done — releases any held resources
#
# File structure:
#   connectors/
#     base.py          ← this file (interface + shared dataclasses)
#     relational.py    ← PostgreSQL, MySQL, SQLite, Oracle, SQL Server
#     document.py      ← PDF, Word, HTML, TXT, Markdown
#     datalake.py      ← Delta Lake, Parquet, CSV, Iceberg
#     nosql.py         ← MongoDB, Elasticsearch, DynamoDB, Cassandra
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


# =============================================================================
# Shared enums
# =============================================================================

class SourceType(str, Enum):
    RELATIONAL = "relational"
    DOCUMENT   = "document"
    DATALAKE   = "datalake"
    NOSQL      = "nosql"


class ConnectorState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED    = "connected"
    ERROR        = "error"


class ColumnRole(str, Enum):
    """What role a column plays in its table."""
    PRIMARY_KEY  = "pk"
    FOREIGN_KEY  = "fk"
    REGULAR      = "regular"
    DISPLAY      = "display"    # human-readable identifier (e.g. incident_no)


# =============================================================================
# Shared output dataclasses
# These are the normalised structures every connector produces.
# schema_unifier.py maps them to VedaSchema — the internal format.
# =============================================================================

@dataclass
class ConnectorStatus:
    """Result of a connect() call."""
    ok:           bool
    source_id:    str
    source_type:  str
    engine:       str
    message:      str          # human-readable status or error
    latency_ms:   float        # connection latency for health monitoring
    metadata:     dict = field(default_factory=dict)


@dataclass
class RawColumn:
    """
    A single column from any relational or datalake source.
    Produced by get_schema() on relational and datalake connectors.
    Consumed by schema_scanner.py → semantic_type_inference.py.
    """
    col_id:          str            # stable UUID assigned by connector
    col_name:        str            # raw column name
    table_id:        str            # UUID of parent table
    table_name:      str            # name of parent table
    data_type:       str            # normalised type string (see DATA_TYPE_MAP below)
    role:            ColumnRole     # pk | fk | regular | display
    is_pk:           bool
    is_fk:           bool
    fk_ref_table:    Optional[str]  # referenced table name (None if not FK)
    fk_ref_col:      Optional[str]  # referenced column name (None if not FK)
    fk_ref_table_id: Optional[str]  # referenced table UUID (None if not FK)
    nullable:        bool
    cardinality:     Optional[int]  # distinct value count (None if not sampled)
    source_id:       str            # which VEDA_SOURCE this came from


@dataclass
class RawTable:
    """
    A single table from any relational or datalake source.
    Produced by get_schema() on relational and datalake connectors.
    """
    table_id:   str
    table_name:  str
    row_count:   int
    columns:     List[RawColumn] = field(default_factory=list)
    source_id:   str = ""
    metadata:    dict = field(default_factory=dict)


@dataclass
class RawSchema:
    """
    Complete schema from a relational or datalake source.
    Produced by get_schema(). Passed to schema_scanner.py.
    """
    source_id:    str
    source_type:  str
    engine:       str
    tables:       List[RawTable]
    fk_edges:     List[dict]       # {from_col_id, from_col_name, from_table_id,
                                   #  from_table, to_col_id, to_col_name,
                                   #  to_table_id, to_table}
    stats:        dict = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """
    A text chunk from a document source.
    Produced by get_chunks() on document connectors.
    Consumed by ingestion/chunk_embedder.py.
    """
    chunk_id:    str            # stable UUID
    source_id:   str            # which VEDA_SOURCE
    doc_id:      str            # UUID of parent document
    doc_name:    str            # filename or title
    doc_path:    str            # full path or URI
    doc_format:  str            # pdf | docx | txt | html | md
    chunk_index: int            # position within document (0-based)
    text:        str            # chunk text content
    page_num:    Optional[int]  # page number (PDFs only, None otherwise)
    metadata:    dict = field(default_factory=dict)


@dataclass
class NoSQLCollection:
    """
    A collection/index from a NoSQL source.
    Produced by get_schema() on NoSQL connectors.
    """
    collection_id:   str
    collection_name: str
    source_id:       str
    engine:          str            # mongodb | elasticsearch | dynamodb
    inferred_fields: List[dict]     # [{name, type, nullable, sample_values}]
    doc_count:       int
    metadata:        dict = field(default_factory=dict)


@dataclass
class QueryResult:
    """
    Result of executing a query against any source.
    Produced by execute_query() on all connector types.
    """
    source_id:    str
    source_type:  str
    rows:         List[Dict[str, Any]]
    row_count:    int
    columns:      List[str]         # column names in order
    sql_or_query: str               # the actual query that was executed
    duration_ms:  float
    truncated:    bool              # True if result was capped at row limit
    error:        Optional[str]     # None on success
    metadata:     dict = field(default_factory=dict)
    answer:       Optional[str] = None  # NL summarization (query/result_explainer.py), when computed
    analytics:    Optional[dict] = None  # deterministic post-exec analysis (result_analyzer.analytics_summary), when computed


# =============================================================================
# Data type normalisation map
#
# Maps engine-specific type strings to VEDA's normalised type vocabulary.
# Used by all connectors to produce consistent RawColumn.data_type values.
# Schema_scanner.py and semantic_type_inference.py expect these normalised types.
# =============================================================================

DATA_TYPE_MAP: Dict[str, str] = {
    # Integer family
    "int":              "integer",
    "int2":             "smallint",
    "int4":             "integer",
    "int8":             "bigint",
    "integer":          "integer",
    "smallint":         "smallint",
    "bigint":           "bigint",
    "tinyint":          "smallint",     # MySQL
    "mediumint":        "integer",      # MySQL
    "serial":           "integer",
    "bigserial":        "bigint",
    "number":           "numeric",      # Oracle

    # Float / decimal family
    "float":            "numeric",
    "float4":           "numeric",
    "float8":           "double",
    "real":             "numeric",
    "double":           "double",
    "double precision": "double",
    "decimal":          "numeric",
    "numeric":          "numeric",
    "money":            "numeric",

    # String family
    "varchar":                "varchar",
    "character varying":      "varchar",
    "nvarchar":               "varchar",    # SQL Server
    "char":                   "character",
    "bpchar":                 "character",
    "text":                   "text",
    "ntext":                  "text",       # SQL Server
    "string":                 "varchar",    # Spark/Delta
    "clob":                   "text",       # Oracle

    # Boolean
    "bool":             "boolean",
    "boolean":          "boolean",
    "bit":              "boolean",          # SQL Server

    # Date / time family
    "date":             "date",
    "time":             "time",
    "timetz":           "time",
    "timestamp":        "timestamp",
    "timestamptz":      "timestamptz",
    "timestamp without time zone": "timestamp",
    "timestamp with time zone":    "timestamptz",
    "datetime":         "timestamp",        # MySQL / SQL Server
    "datetime2":        "timestamp",        # SQL Server
    "interval":         "text",

    # UUID / identifier
    "uuid":             "uuid",
    "uniqueidentifier": "uuid",             # SQL Server

    # Binary / blob
    "bytea":            "bytea",
    "blob":             "bytea",
    "binary":           "bytea",
    "varbinary":        "bytea",

    # JSON / structured
    "json":             "json",
    "jsonb":            "jsonb",

    # Network
    "inet":             "inet",
    "cidr":             "inet",
    "macaddr":          "text",

    # Arrays / special
    "array":            "text",
    "hstore":           "text",
    "xml":              "text",
    "enum":             "varchar",
}


def normalise_data_type(raw_type: str) -> str:
    """
    Normalises an engine-specific data type string to VEDA's vocabulary.
    Case-insensitive. Falls back to 'varchar' for unknown types.
    Strips any precision/scale qualifiers: varchar(255) → varchar.
    """
    if not raw_type:
        return "varchar"
    # Strip precision: varchar(255) → varchar, numeric(10,2) → numeric
    base = raw_type.lower().split("(")[0].strip()
    return DATA_TYPE_MAP.get(base, "varchar")


# =============================================================================
# Abstract base connector
# =============================================================================

class BaseConnector(ABC):
    """
    Abstract interface every VEDA connector must implement.

    Connectors are instantiated with a source config dict from VEDA_SOURCES.
    They produce normalised output (RawSchema, DocumentChunk, etc.) that the
    ingestion dispatcher hands to the existing VEDA pipeline unchanged.

    Capability declaration:
        Each connector declares what it can do via supports_* properties.
        Callers check these before calling methods — no NotImplementedError
        is ever raised for unsupported capabilities.
    """

    def __init__(self, source_config: dict) -> None:
        """
        Parameters
        ----------
        source_config : dict
            One entry from VEDA_SOURCES in config.py.
            Must contain at minimum: id, type, engine, role.
        """
        self._config    = source_config
        self._source_id = source_config["id"]
        self._engine    = source_config.get("engine", "unknown")
        self._state     = ConnectorState.DISCONNECTED

    # ------------------------------------------------------------------
    # Capability flags — override in subclasses as needed
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def state(self) -> ConnectorState:
        return self._state

    @property
    def supports_schema(self) -> bool:
        """True if this connector can return a RawSchema (relational, datalake)."""
        return False

    @property
    def supports_chunks(self) -> bool:
        """True if this connector can return DocumentChunks (document)."""
        return False

    @property
    def supports_nosql_schema(self) -> bool:
        """True if this connector can return NoSQLCollections (nosql)."""
        return False

    @property
    def supports_query(self) -> bool:
        """True if this connector can execute queries and return QueryResult."""
        return False

    @property
    def supports_value_sampling(self) -> bool:
        """True if this connector can sample column values for the value sampler."""
        return False

    # ------------------------------------------------------------------
    # Lifecycle — must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> ConnectorStatus:
        """
        Verifies connectivity and credentials.
        Sets self._state to CONNECTED or ERROR.
        Returns ConnectorStatus with ok=True on success.
        Must not raise — all errors captured in ConnectorStatus.message.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """
        Releases any held connections or file handles.
        Safe to call multiple times.
        Sets self._state to DISCONNECTED.
        """

    # ------------------------------------------------------------------
    # Schema access — relational + datalake connectors implement this
    # ------------------------------------------------------------------

    def get_schema(self) -> RawSchema:
        """
        Returns the complete schema as a RawSchema object.
        Called by ingestion/source_dispatcher.py → schema_scanner.py.

        Only meaningful when supports_schema = True.
        Default implementation raises a clear error if called on wrong type.
        """
        raise NotImplementedError(
            f"Connector '{self._source_id}' (engine={self._engine}) "
            f"does not support get_schema(). "
            f"Only relational and datalake connectors support this."
        )

    def sample_column_values(
        self,
        table_name: str,
        col_name:   str,
        n:          int = 100,
    ) -> List[str]:
        """
        Samples up to n distinct non-null values from table.col.
        Used by ingestion/value_sampler.py and ingestion/data_graph.py.
        Only meaningful when supports_value_sampling = True.
        Returns empty list if unsupported or on error.
        """
        return []

    def get_row_count(self, table_name: str) -> int:
        """
        Returns approximate row count for a table.
        Used by schema scanner for display and REG builder for node features.
        Returns 0 if unsupported or on error.
        """
        return 0

    # ------------------------------------------------------------------
    # Document access — document connectors implement this
    # ------------------------------------------------------------------

    def get_chunks(
        self,
        chunk_size:    int = 512,
        chunk_overlap: int = 64,
    ) -> Iterator[DocumentChunk]:
        """
        Yields DocumentChunk objects for all documents in this source.
        Called by ingestion/chunk_embedder.py.
        Only meaningful when supports_chunks = True.
        """
        raise NotImplementedError(
            f"Connector '{self._source_id}' does not support get_chunks(). "
            f"Only document connectors support this."
        )

    def get_document_count(self) -> int:
        """Returns total number of documents in this source. 0 if unsupported."""
        return 0

    # ------------------------------------------------------------------
    # NoSQL access — nosql connectors implement this
    # ------------------------------------------------------------------

    def get_nosql_schema(self) -> List[NoSQLCollection]:
        """
        Infers schema from sampled documents/records.
        Called by ingestion/source_dispatcher.py for NoSQL sources.
        Only meaningful when supports_nosql_schema = True.
        """
        raise NotImplementedError(
            f"Connector '{self._source_id}' does not support get_nosql_schema(). "
            f"Only nosql connectors support this."
        )

    # ------------------------------------------------------------------
    # Query execution — all queryable connectors implement this
    # ------------------------------------------------------------------

    def execute_query(
        self,
        query:      str,
        params:     Optional[list] = None,
        row_limit:  int = 1000,
        timeout_sec: int = 30,
    ) -> QueryResult:
        """
        Executes a query against this source.
        For relational sources: parameterised SQL.
        For datalake sources: DuckDB/Spark SQL.
        For nosql sources: native query dict (JSON-serialisable).
        Only meaningful when supports_query = True.
        """
        raise NotImplementedError(
            f"Connector '{self._source_id}' does not support execute_query(). "
            f"Set role='queryable' and implement execute_query()."
        )

    # ------------------------------------------------------------------
    # Health check — convenience method
    # ------------------------------------------------------------------

    def health_check(self) -> ConnectorStatus:
        """
        Re-verifies connectivity. Cheaper than full connect() on warm connectors.
        Default implementation calls connect() — override for efficiency.
        """
        return self.connect()

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"source_id='{self._source_id}', "
            f"engine='{self._engine}', "
            f"state={self._state.value})"
        )


# =============================================================================
# Connector registry — maps source type strings to connector classes
# Populated by each connector module at import time via register_connector().
# Used by ingestion/source_dispatcher.py to instantiate the right connector.
# =============================================================================

_CONNECTOR_REGISTRY: Dict[str, type] = {}


def register_connector(source_type: str, engine: str, cls: type) -> None:
    """
    Registers a connector class for a given source_type + engine combination.
    Called at module level in each connector file.

    Example:
        register_connector("relational", "postgresql", PostgreSQLConnector)
        register_connector("relational", "mysql",      MySQLConnector)
    """
    key = f"{source_type}:{engine}"
    _CONNECTOR_REGISTRY[key] = cls


def get_connector_class(source_type: str, engine: str) -> type:
    """
    Returns the connector class for a given source_type + engine.
    Falls back to the generic class for that source_type if no exact match.
    Raises KeyError if no connector is registered for this combination.
    """
    key         = f"{source_type}:{engine}"
    generic_key = f"{source_type}:generic"

    if key in _CONNECTOR_REGISTRY:
        return _CONNECTOR_REGISTRY[key]
    if generic_key in _CONNECTOR_REGISTRY:
        return _CONNECTOR_REGISTRY[generic_key]
    raise KeyError(
        f"No connector registered for source_type='{source_type}' "
        f"engine='{engine}'. "
        f"Available: {list(_CONNECTOR_REGISTRY.keys())}"
    )


def build_connector(source_config: dict) -> BaseConnector:
    """
    Factory function. Instantiates the right connector for a source config.
    Called by ingestion/source_dispatcher.py.

    Imports the connector module lazily to avoid import errors when
    optional dependencies (psycopg2, pdfplumber, delta-rs) are not installed.
    """
    source_type = source_config.get("type", "")
    engine      = source_config.get("engine", "generic")

    # Lazy import the connector module so missing deps don't break startup
    if source_type == SourceType.RELATIONAL:
        from connectors.relational import _ensure_registered
        _ensure_registered()
    elif source_type == SourceType.DOCUMENT:
        from connectors.document import _ensure_registered    # Phase 2
        _ensure_registered()
    elif source_type == SourceType.DATALAKE:
        from connectors.datalake import _ensure_registered    # Phase 3
        _ensure_registered()
    elif source_type == SourceType.NOSQL:
        from connectors.nosql import _ensure_registered       # Phase 4
        _ensure_registered()
    else:
        raise ValueError(
            f"Unknown source type '{source_type}'. "
            f"Must be one of: {[e.value for e in SourceType]}"
        )

    cls = get_connector_class(source_type, engine)
    return cls(source_config)


# =============================================================================
# Smoke test — python connectors/base.py
# =============================================================================

if __name__ == "__main__":
    # Verify dataclasses instantiate correctly
    status = ConnectorStatus(
        ok=True, source_id="test", source_type="relational",
        engine="postgresql", message="OK", latency_ms=1.2,
    )
    print(f"ConnectorStatus: {status}")

    col = RawColumn(
        col_id="col-1", col_name="workflow_state", table_id="tbl-1",
        table_name="incident", data_type="varchar", role=ColumnRole.REGULAR,
        is_pk=False, is_fk=False, fk_ref_table=None, fk_ref_col=None,
        fk_ref_table_id=None, nullable=True, cardinality=5, source_id="primary_db",
    )
    print(f"RawColumn: {col.table_name}.{col.col_name} ({col.data_type})")

    # Verify data type normalisation
    tests = [
        ("varchar(255)",          "varchar"),
        ("CHARACTER VARYING(100)", "varchar"),
        ("INT",                   "integer"),
        ("BIGINT",                "bigint"),
        ("TIMESTAMP WITH TIME ZONE", "timestamptz"),
        ("DOUBLE PRECISION",      "double"),
        ("NUMBER",                "numeric"),
        ("NVARCHAR(MAX)",         "varchar"),
        ("unknown_type",          "varchar"),   # fallback
    ]
    print("\nData type normalisation:")
    all_ok = True
    for raw, expected in tests:
        got = normalise_data_type(raw)
        ok  = got == expected
        if not ok: all_ok = False
        print(f"  {'✓' if ok else '✗'} {raw:<35} → {got}  (expected {expected})")

    print()
    print("Connector registry empty (no connectors loaded yet):")
    print(f"  {_CONNECTOR_REGISTRY}")
    print()
    print("All base.py checks passed ✓" if all_ok else "SOME CHECKS FAILED ✗")