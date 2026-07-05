# =============================================================================
# connectors/nosql.py
# VEDA — NoSQL Connector (Phase 4)
#
# Implements BaseConnector for NoSQL sources.
# Supports: MongoDB, Elasticsearch, DynamoDB
#
# Schema inference:
#   MongoDB / DynamoDB — sample NOSQL_SCHEMA_SAMPLE_SIZE documents, flatten
#     nested fields up to NOSQL_MAX_NESTING_DEPTH, infer types from values.
#   Elasticsearch — uses index mapping API (no sampling needed).
#
# Query execution:
#   Each connector's execute_query() accepts a JSON-serialised native query dict
#   produced by query/nosql_builder.py. Format per engine:
#     mongodb        — {"collection": "...", "intent": "find|count|aggregate",
#                        "filter": {...}, "pipeline": [...]}
#     elasticsearch  — {"index": "...", "body": {...}}
#     dynamodb       — {"table": "...", "intent": "scan|count",
#                        "FilterExpression": "...", ...}
#
# All deps are optional — connectors degrade gracefully when not installed.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from connectors.base import (
    BaseConnector,
    ConnectorState,
    ConnectorStatus,
    NoSQLCollection,
    QueryResult,
    register_connector,
)
from config import NOSQL_SCHEMA_SAMPLE_SIZE, NOSQL_MAX_NESTING_DEPTH, VEDA_INTERNAL_TABLES


# =============================================================================
# Optional dependency availability flags
# =============================================================================

try:
    import pymongo as _pymongo
    _MONGO_AVAILABLE = True
except ImportError:
    _MONGO_AVAILABLE = False

try:
    from elasticsearch import Elasticsearch as _ES
    _ES_AVAILABLE = True
except ImportError:
    _ES_AVAILABLE = False

try:
    import boto3 as _boto3
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False


# =============================================================================
# Shared schema-inference helpers
# =============================================================================

def _infer_python_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "numeric"
    if isinstance(value, str):
        return "varchar"
    if isinstance(value, dict):
        return "json"
    if isinstance(value, list):
        return "text"
    return "varchar"


def _flatten_doc(doc: dict, prefix: str = "", max_depth: int = 3) -> Dict[str, Any]:
    result = {}
    if max_depth <= 0:
        return result
    for key, val in doc.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and max_depth > 1:
            result.update(_flatten_doc(val, full_key, max_depth - 1))
        else:
            result[full_key] = val
    return result


def _infer_schema_from_docs(docs: List[dict], max_depth: int = 3) -> List[dict]:
    """
    Returns [{name, type, nullable, sample_values}] from a document sample.
    Counts type occurrences per field; dominant type wins.
    """
    field_types:  Dict[str, Dict[str, int]] = {}
    field_values: Dict[str, List[str]]      = {}

    for doc in docs:
        flat = _flatten_doc(doc, max_depth=max_depth)
        for field, val in flat.items():
            if field in ("_id",):
                continue
            t = _infer_python_type(val)
            if field not in field_types:
                field_types[field]  = {}
                field_values[field] = []
            field_types[field][t] = field_types[field].get(t, 0) + 1
            if len(field_values[field]) < 5 and val is not None:
                sv = str(val)[:64]
                if sv not in field_values[field]:
                    field_values[field].append(sv)

    result = []
    for field, type_counts in sorted(field_types.items()):
        dominant_type  = max(type_counts, key=type_counts.get)
        total_seen     = sum(type_counts.values())
        nullable       = total_seen < len(docs)
        result.append({
            "name":          field,
            "type":          dominant_type,
            "nullable":      nullable,
            "sample_values": field_values.get(field, []),
        })
    return result


def _bson_to_dict(doc: dict) -> dict:
    result = {}
    for k, v in doc.items():
        if k == "_id":
            result["_id"] = str(v)
        elif hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        elif isinstance(v, dict):
            result[k] = _bson_to_dict(v)
        else:
            result[k] = v
    return result


# =============================================================================
# Elasticsearch type mapping
# =============================================================================

_ES_TYPE_MAP = {
    "text":      "text",     "keyword":   "varchar",
    "integer":   "integer",  "long":      "bigint",
    "short":     "smallint", "byte":      "smallint",
    "float":     "numeric",  "double":    "double",
    "half_float":"numeric",  "scaled_float": "numeric",
    "boolean":   "boolean",
    "date":      "timestamp",
    "ip":        "inet",
    "object":    "json",     "nested":    "json",
    "binary":    "bytea",
}


def _es_type_to_veda(es_type: str) -> str:
    return _ES_TYPE_MAP.get(es_type, "varchar")


# =============================================================================
# MongoDB connector
# =============================================================================

class MongoDBConnector(BaseConnector):
    """
    MongoDB connector. One source entry = one MongoDB database.
    Schema inference samples documents per collection.
    Queries run as native MongoDB find / count / aggregate operations.
    """

    def __init__(self, source_config: dict) -> None:
        super().__init__(source_config)
        self._host     = source_config.get("host", "localhost")
        self._port     = int(source_config.get("port", 27017))
        self._dbname   = source_config.get("dbname", "")
        self._user     = source_config.get("user")
        self._password = source_config.get("password")
        self._client   = None

    @property
    def supports_nosql_schema(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return _MONGO_AVAILABLE

    def connect(self) -> ConnectorStatus:
        t0 = time.time()
        if not _MONGO_AVAILABLE:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="nosql",
                engine="mongodb",
                message="pymongo not installed — pip install pymongo",
                latency_ms=0.0,
            )
        try:
            if self._user:
                uri = (
                    f"mongodb://{self._user}:{self._password}"
                    f"@{self._host}:{self._port}/"
                )
            else:
                uri = f"mongodb://{self._host}:{self._port}/"
            self._client = _pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
            self._client.server_info()
            self._state = ConnectorState.CONNECTED
            return ConnectorStatus(
                ok=True, source_id=self._source_id, source_type="nosql",
                engine="mongodb",
                message=f"Connected to {self._host}:{self._port}/{self._dbname}",
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception as e:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="nosql",
                engine="mongodb", message=str(e),
                latency_ms=round((time.time() - t0) * 1000, 2),
            )

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._state = ConnectorState.DISCONNECTED

    def get_nosql_schema(self) -> List[NoSQLCollection]:
        if not _MONGO_AVAILABLE or self._client is None:
            return []
        db          = self._client[self._dbname]
        collections = []
        for col_name in db.list_collection_names():
            if col_name.lower() in VEDA_INTERNAL_TABLES:
                continue
            try:
                col       = db[col_name]
                doc_count = col.count_documents({})
                sample    = list(col.find({}, limit=NOSQL_SCHEMA_SAMPLE_SIZE))
                fields    = _infer_schema_from_docs(sample, NOSQL_MAX_NESTING_DEPTH)
                collections.append(NoSQLCollection(
                    collection_id   = str(uuid.uuid4()),
                    collection_name = col_name,
                    source_id       = self._source_id,
                    engine          = "mongodb",
                    inferred_fields = fields,
                    doc_count       = doc_count,
                ))
            except Exception:
                continue
        return collections

    def execute_query(
        self,
        query:       str,
        params:      Optional[list] = None,
        row_limit:   int = 1000,
        timeout_sec: int = 30,
    ) -> QueryResult:
        t0 = time.time()
        if not _MONGO_AVAILABLE or self._client is None:
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=[], row_count=0, columns=[],
                sql_or_query=query, duration_ms=0.0, truncated=False,
                error="MongoDB not connected",
            )
        try:
            q          = json.loads(query) if isinstance(query, str) else query
            col_name   = q.get("collection", "")
            intent     = q.get("intent", "find")
            filter_doc = q.get("filter", {})
            projection = q.get("projection")

            db  = self._client[self._dbname]
            col = db[col_name]

            if intent == "count":
                count = col.count_documents(filter_doc)
                rows  = [{"count": count}]
                cols  = ["count"]
                return QueryResult(
                    source_id=self._source_id, source_type="nosql",
                    rows=rows, row_count=1, columns=cols,
                    sql_or_query=query,
                    duration_ms=round((time.time() - t0) * 1000, 2),
                    truncated=False, error=None,
                )

            if intent == "aggregate":
                pipeline = q.get("pipeline", [])
                cursor   = col.aggregate(pipeline, maxTimeMS=timeout_sec * 1000)
                all_rows = list(cursor)[: row_limit + 1]
                truncated = len(all_rows) > row_limit
                rows = [_bson_to_dict(r) for r in all_rows[:row_limit]]
                cols = list(rows[0].keys()) if rows else []
                return QueryResult(
                    source_id=self._source_id, source_type="nosql",
                    rows=rows, row_count=len(rows), columns=cols,
                    sql_or_query=query,
                    duration_ms=round((time.time() - t0) * 1000, 2),
                    truncated=truncated, error=None,
                )

            # find
            cursor    = (
                col.find(filter_doc, projection=projection)
                   .limit(row_limit + 1)
                   .max_time_ms(timeout_sec * 1000)
            )
            all_rows  = list(cursor)
            truncated = len(all_rows) > row_limit
            rows      = [_bson_to_dict(r) for r in all_rows[:row_limit]]
            cols      = list(rows[0].keys()) if rows else []
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=rows, row_count=len(rows), columns=cols,
                sql_or_query=query,
                duration_ms=round((time.time() - t0) * 1000, 2),
                truncated=truncated, error=None,
            )
        except Exception as e:
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=[], row_count=0, columns=[],
                sql_or_query=query,
                duration_ms=round((time.time() - t0) * 1000, 2),
                truncated=False, error=str(e),
            )


# =============================================================================
# Elasticsearch connector
# =============================================================================

class ElasticsearchConnector(BaseConnector):
    """
    Elasticsearch connector. One source entry = one ES cluster.
    Schema is read from the index mapping API (no sampling needed).
    Queries run as native ES query DSL via the search API.
    """

    def __init__(self, source_config: dict) -> None:
        super().__init__(source_config)
        self._host    = source_config.get("host", "localhost")
        self._port    = int(source_config.get("port", 9200))
        self._index   = source_config.get("index") or source_config.get("dbname", "*")
        self._user    = source_config.get("user")
        self._password = source_config.get("password")
        self._client  = None

    @property
    def supports_nosql_schema(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return _ES_AVAILABLE

    def connect(self) -> ConnectorStatus:
        t0 = time.time()
        if not _ES_AVAILABLE:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="nosql",
                engine="elasticsearch",
                message="elasticsearch not installed — pip install elasticsearch",
                latency_ms=0.0,
            )
        try:
            kwargs: dict = {"hosts": [f"http://{self._host}:{self._port}"]}
            if self._user:
                kwargs["basic_auth"] = (self._user, self._password or "")
            self._client = _ES(**kwargs)
            info = self._client.info()
            self._state = ConnectorState.CONNECTED
            return ConnectorStatus(
                ok=True, source_id=self._source_id, source_type="nosql",
                engine="elasticsearch",
                message=(
                    f"Connected to ES {info['version']['number']}"
                    f" at {self._host}:{self._port}"
                ),
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception as e:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="nosql",
                engine="elasticsearch", message=str(e),
                latency_ms=round((time.time() - t0) * 1000, 2),
            )

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._state = ConnectorState.DISCONNECTED

    def get_nosql_schema(self) -> List[NoSQLCollection]:
        if not _ES_AVAILABLE or self._client is None:
            return []
        collections = []
        try:
            if self._index and self._index != "*":
                indices = [self._index]
            else:
                cat     = self._client.cat.indices(format="json")
                indices = [i["index"] for i in cat if not i["index"].startswith(".")]

            for idx in indices:
                if idx.lower() in VEDA_INTERNAL_TABLES:
                    continue
                try:
                    mapping   = self._client.indices.get_mapping(index=idx)
                    props     = mapping[idx]["mappings"].get("properties", {})
                    fields    = [
                        {
                            "name":          fname,
                            "type":          _es_type_to_veda(fmeta.get("type", "text")),
                            "nullable":      True,
                            "sample_values": [],
                        }
                        for fname, fmeta in props.items()
                    ]
                    doc_count = self._client.count(index=idx)["count"]
                    collections.append(NoSQLCollection(
                        collection_id   = str(uuid.uuid4()),
                        collection_name = idx,
                        source_id       = self._source_id,
                        engine          = "elasticsearch",
                        inferred_fields = fields,
                        doc_count       = doc_count,
                    ))
                except Exception:
                    continue
        except Exception:
            pass
        return collections

    def execute_query(
        self,
        query:       str,
        params:      Optional[list] = None,
        row_limit:   int = 1000,
        timeout_sec: int = 30,
    ) -> QueryResult:
        t0 = time.time()
        if not _ES_AVAILABLE or self._client is None:
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=[], row_count=0, columns=[],
                sql_or_query=query, duration_ms=0.0, truncated=False,
                error="Elasticsearch not connected",
            )
        try:
            q     = json.loads(query) if isinstance(query, str) else query
            index = q.get("index", self._index)
            body  = q.get("body", {"query": {"match_all": {}}})

            resp  = self._client.search(
                index  = index,
                body   = body,
                size   = min(row_limit, 10000),
                timeout = f"{timeout_sec}s",
            )
            hits  = resp["hits"]["hits"]
            rows  = [{**h["_source"], "_id": h["_id"]} for h in hits]
            cols  = list(rows[0].keys()) if rows else []
            total = resp["hits"]["total"]
            total_val = total["value"] if isinstance(total, dict) else total
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=rows, row_count=len(rows), columns=cols,
                sql_or_query=query,
                duration_ms=round((time.time() - t0) * 1000, 2),
                truncated=total_val > row_limit,
                error=None,
            )
        except Exception as e:
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=[], row_count=0, columns=[],
                sql_or_query=query,
                duration_ms=round((time.time() - t0) * 1000, 2),
                truncated=False, error=str(e),
            )


# =============================================================================
# DynamoDB connector
# =============================================================================

class DynamoDBConnector(BaseConnector):
    """
    DynamoDB connector. One source entry = one AWS region / table set.
    Schema inference samples items per table.
    Queries use DynamoDB Scan with optional FilterExpression.
    """

    def __init__(self, source_config: dict) -> None:
        super().__init__(source_config)
        self._region     = source_config.get("region", "us-east-1")
        self._table      = source_config.get("table") or source_config.get("dbname", "")
        self._access_key = source_config.get("aws_access_key")
        self._secret_key = source_config.get("aws_secret_key")
        self._resource   = None

    @property
    def supports_nosql_schema(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return _BOTO3_AVAILABLE

    def _boto_kwargs(self) -> dict:
        kwargs: dict = {"region_name": self._region}
        if self._access_key:
            kwargs["aws_access_key_id"]     = self._access_key
            kwargs["aws_secret_access_key"] = self._secret_key
        return kwargs

    def connect(self) -> ConnectorStatus:
        t0 = time.time()
        if not _BOTO3_AVAILABLE:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="nosql",
                engine="dynamodb",
                message="boto3 not installed — pip install boto3",
                latency_ms=0.0,
            )
        try:
            kw             = self._boto_kwargs()
            self._resource = _boto3.resource("dynamodb", **kw)
            client         = _boto3.client("dynamodb", **kw)
            client.list_tables(Limit=1)
            self._state = ConnectorState.CONNECTED
            return ConnectorStatus(
                ok=True, source_id=self._source_id, source_type="nosql",
                engine="dynamodb",
                message=f"Connected to DynamoDB region={self._region}",
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception as e:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="nosql",
                engine="dynamodb", message=str(e),
                latency_ms=round((time.time() - t0) * 1000, 2),
            )

    def disconnect(self) -> None:
        self._resource = None
        self._state    = ConnectorState.DISCONNECTED

    def get_nosql_schema(self) -> List[NoSQLCollection]:
        if not _BOTO3_AVAILABLE or self._resource is None:
            return []
        collections = []
        try:
            tables: List[str] = []
            if self._table:
                tables = [self._table]
            else:
                client    = self._resource.meta.client
                paginator = client.get_paginator("list_tables")
                for page in paginator.paginate():
                    tables.extend(page["TableNames"])

            for tbl_name in tables:
                if tbl_name.lower() in VEDA_INTERNAL_TABLES:
                    continue
                try:
                    tbl   = self._resource.Table(tbl_name)
                    tbl.load()
                    resp  = tbl.scan(Limit=NOSQL_SCHEMA_SAMPLE_SIZE)
                    items = resp.get("Items", [])
                    fields = _infer_schema_from_docs(items, NOSQL_MAX_NESTING_DEPTH)
                    collections.append(NoSQLCollection(
                        collection_id   = str(uuid.uuid4()),
                        collection_name = tbl_name,
                        source_id       = self._source_id,
                        engine          = "dynamodb",
                        inferred_fields = fields,
                        doc_count       = tbl.item_count,
                    ))
                except Exception:
                    continue
        except Exception:
            pass
        return collections

    def execute_query(
        self,
        query:       str,
        params:      Optional[list] = None,
        row_limit:   int = 1000,
        timeout_sec: int = 30,
    ) -> QueryResult:
        t0 = time.time()
        if not _BOTO3_AVAILABLE or self._resource is None:
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=[], row_count=0, columns=[],
                sql_or_query=query, duration_ms=0.0, truncated=False,
                error="DynamoDB not connected",
            )
        try:
            q        = json.loads(query) if isinstance(query, str) else query
            tbl_name = q.get("table", self._table)
            intent   = q.get("intent", "scan")
            tbl      = self._resource.Table(tbl_name)

            if intent == "count":
                tbl.load()
                return QueryResult(
                    source_id=self._source_id, source_type="nosql",
                    rows=[{"count": tbl.item_count}], row_count=1, columns=["count"],
                    sql_or_query=query,
                    duration_ms=round((time.time() - t0) * 1000, 2),
                    truncated=False, error=None,
                )

            scan_kwargs: dict = {"Limit": row_limit + 1}
            for key in ("FilterExpression", "ExpressionAttributeValues", "ExpressionAttributeNames"):
                if q.get(key):
                    scan_kwargs[key] = q[key]

            resp      = tbl.scan(**scan_kwargs)
            items     = resp.get("Items", [])
            truncated = len(items) > row_limit
            rows      = items[:row_limit]
            cols      = list(rows[0].keys()) if rows else []
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=rows, row_count=len(rows), columns=cols,
                sql_or_query=query,
                duration_ms=round((time.time() - t0) * 1000, 2),
                truncated=truncated, error=None,
            )
        except Exception as e:
            return QueryResult(
                source_id=self._source_id, source_type="nosql",
                rows=[], row_count=0, columns=[],
                sql_or_query=query,
                duration_ms=round((time.time() - t0) * 1000, 2),
                truncated=False, error=str(e),
            )


# =============================================================================
# Connector registration
# =============================================================================

def _ensure_registered() -> None:
    pass  # registration happens at module level below


register_connector("nosql", "mongodb",       MongoDBConnector)
register_connector("nosql", "elasticsearch", ElasticsearchConnector)
register_connector("nosql", "dynamodb",      DynamoDBConnector)
register_connector("nosql", "generic",       MongoDBConnector)   # fallback
