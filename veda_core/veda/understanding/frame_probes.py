"""veda.understanding.frame_probes — Stage 4: verify a grounded frame against the DATA
before its SQL runs.

ONE read-only round trip per frame (a single `SELECT COUNT(*), COUNT(*) FILTER (…)…`),
statement_timeout FRAME_PROBE_TIMEOUT_MS (2 s):

  probe                     signal                         action (frame_path decides)
  ─────────────────────────  ─────────────────────────────  ───────────────────────────────
  filtered vs anchor count   a filter keeps 0 of N > 0      alternate value/column, else
                                                            clarify with the counts
  numeric range hit rate     0 % or 100 % of rows           wrong column → alternate measure,
                                                            else clarify
  order column distinctness  COUNT(DISTINCT) ≤ 1            wrong measure → clarify
  (entity coverage and the low-confidence 0-row rule are checked in frame_path / after
   execution — they need the compiled SQL and the result.)

Observations are returned as plain sentences too, so one revision round can hand them to
the extractor as evidence for a re-extraction of the FRAME (never of the SQL).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from veda.understanding.frame_grounding import GroundedFrame, _bare


@dataclass
class ProbeResult:
    ran: bool = False
    total: Optional[int] = None
    filtered: Optional[int] = None
    per_filter: Dict[str, int] = field(default_factory=dict)       # "col op" → rows kept
    order_distinct: Optional[int] = None
    problems: List[Dict[str, Any]] = field(default_factory=list)   # [{kind, slot, detail}]
    observations: List[str] = field(default_factory=list)
    error: Optional[str] = None
    ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.ran and not self.problems


def _timeout() -> int:
    try:
        import config
        return int(getattr(config, "FRAME_PROBE_TIMEOUT_MS", 2000))
    except Exception:
        return 2000


def _run(sql: str):
    from veda.execution import execute_sql
    from veda.validation import validate_and_parameterize  # noqa: F401  (literals are ours)
    return execute_sql(sql, None, timeout_ms=_timeout())


def distinct_values(table: str, column: str, limit: int = 41) -> Optional[List[str]]:
    cols, rows, err = _run(f'SELECT DISTINCT CAST("{column}" AS TEXT) AS v FROM "{table}" '
                           f'WHERE "{column}" IS NOT NULL LIMIT {int(limit)}')
    if err or rows is None:
        return None
    return [str(r[0]) for r in rows if r and r[0] is not None]


def probe(gf: GroundedFrame) -> ProbeResult:
    import time
    from veda.understanding.frame_compiler import predicate, _alias_map, _q
    pr = ProbeResult()
    try:
        import config
        if not getattr(config, "FRAME_PROBES_ENABLED", True):
            return pr
    except Exception:
        pass
    # the probe runs on the anchor alone; filters on joined parents are probed through the
    # join (LEFT JOIN keeps the anchor row count honest)
    am = _alias_map(gf)
    frm = f'FROM {_q(_bare(gf.anchor))} t0' + "".join(
        f' LEFT JOIN {_q(_bare(j.table))} {am[j.table]} ON {am[j.table]}."{j.pk_column}" = t0."{j.fk_column}"'
        for j in gf.joins)
    sel = ["COUNT(*) AS n_all"]
    keys = []
    preds = []
    for i, f in enumerate(gf.filters):
        p = predicate(f, am.get(f.table, "t0"))
        if not p:
            continue
        preds.append(p)
        sel.append(f"COUNT(*) FILTER (WHERE {p}) AS n_f{i}")
        keys.append((f"{f.column} {f.op}", i, f))
    if gf.time:
        a = am.get(gf.time["table"], "t0")
        tp = []
        if gf.time.get("start"):
            tp.append(f"{a}.\"{gf.time['column']}\" >= '{gf.time['start']}'")
        if gf.time.get("end"):
            tp.append(f"{a}.\"{gf.time['column']}\" <= '{gf.time['end']}'")
        if tp:
            preds.append(" AND ".join(tp))
            sel.append(f"COUNT(*) FILTER (WHERE {' AND '.join(tp)}) AS n_time")
    if preds:
        sel.append(f"COUNT(*) FILTER (WHERE {' AND '.join(preds)}) AS n_where")
    ord_measure = False
    if gf.order and gf.order[0] == gf.anchor and not gf.group_by:
        oc = gf.order[1]
        sel.append(f'COUNT(DISTINCT t0."{oc}") AS n_order_distinct')
        ord_measure = oc != "id"
    sql = f"SELECT {', '.join(sel)} {frm}"
    t0 = time.time()
    cols, rows, err = _run(sql)
    pr.ms = round((time.time() - t0) * 1000.0, 1)
    if err or not rows:
        pr.error = (err or "no rows")[:200]
        return pr
    pr.ran = True
    rec = dict(zip([c.lower() for c in cols], rows[0]))
    pr.total = int(rec.get("n_all") or 0)
    if preds:
        pr.filtered = int(rec.get("n_where") or 0)
    for label, i, f in keys:
        n = int(rec.get(f"n_f{i}") or 0)
        pr.per_filter[label] = n
        if pr.total > 0 and n == 0 and f.grounding == "temporal":
            # nothing in a date range is an ANSWER ("no users signed up last month"), not a
            # sign the range was misread — say so instead of asking back
            pr.observations.append(f"No {gf.anchor} rows fall in the requested period.")
            gf.notes.append("nothing falls in that period")
        elif pr.total > 0 and n == 0:
            pr.problems.append({"kind": "empty_filter", "slot": f"filter:{f.column}", "filter": f,
                                "detail": f"{f.column} {f.op} {f.value!r} keeps 0 of {pr.total}"})
            pr.observations.append(f"The condition {f.column.replace('_', ' ')} {f.op} {f.value} "
                                   f"matches 0 of {pr.total} {gf.anchor} rows.")
        elif f.grounding == "numeric" and pr.total > 20 and n == pr.total and f.op != "between":
            pr.problems.append({"kind": "vacuous_range", "slot": f"filter:{f.column}", "filter": f,
                                "detail": f"{f.column} {f.op} {f.value} keeps all {pr.total} rows"})
            pr.observations.append(f"The condition {f.column.replace('_', ' ')} {f.op} {f.value} "
                                   f"is true for every row — probably the wrong column.")
    if "n_order_distinct" in rec:
        pr.order_distinct = int(rec.get("n_order_distinct") or 0)
        if pr.total > 1 and pr.order_distinct <= 1 and ord_measure:
            pr.problems.append({"kind": "constant_order", "slot": "order",
                                "detail": f"{gf.order[1]} has {pr.order_distinct} distinct value(s)"})
            pr.observations.append(f"Sorting by {gf.order[1].replace('_', ' ')} is meaningless: "
                                   f"it has {pr.order_distinct} distinct value(s).")
    only_time = gf.time is not None and all(f.grounding == "temporal" for f in gf.filters)
    if preds and pr.total and pr.filtered == 0 and only_time:
        if "nothing falls in that period" not in gf.notes:
            gf.notes.append("nothing falls in that period")
    elif preds and pr.total and pr.filtered == 0 and not pr.problems:
        pr.problems.append({"kind": "empty_conjunction", "slot": "filters",
                            "detail": f"all filters together keep 0 of {pr.total}"})
        pr.observations.append(f"Together the conditions match 0 of {pr.total} rows.")
    return pr
