# =============================================================================
# tests/tools/explainability_invariants.py
# VEDA — invariant checker for a CAPTURED chat SSE stream.
#
# Unit tests pin one behaviour each. This pins the SHAPE OF A WHOLE TURN, against
# a real stream off the live stack, which is where the defects this session found
# actually lived: a denial displayed as a partial-access warning, two refusals of
# identical shape reporting different terminal statuses, a raw source id in the
# user-facing block on one of the two assembler paths. None of those is visible
# from inside a single unit test.
#
# USE
#   1. capture a stream:
#        curl -s -N -X POST http://localhost:8080/api/v1/conversations/query \
#          -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
#          -d '{"message":"how many assets are there","source_id":2,"stream":true}' \
#          -o /tmp/one.sse
#   2. check it:
#        python tests/tools/explainability_invariants.py /tmp/one.sse::sql_count
#
#   Exit code is the violation count, so it drops into CI as-is.
#
# The vocabularies below are the contract's own closed lists (CHAT_API_CONTRACT.md
# §1c/§1d). When one legitimately grows, it grows here too — that is the point:
# a new value cannot reach the UI without someone editing this list.
#
# VERIFY THE CHECKER, NOT JUST THE STREAM. It is easy to write a checker that
# passes everything. `--self-test` feeds it a deliberately broken payload and
# fails unless every planted defect is caught.
# =============================================================================
import json, re, sys

# ---- closed vocabularies (the contract's own lists) -------------------------
STEP_IDS   = ["understanding", "finding", "analyzing", "preparing"]
STEP_STATE = {"pending","active","completed","warning","failed","skipped"}
DETAIL_T   = {"access","source","evidence","operation","validation","output"}
EXEC_T     = {"sql","documents","multi_source","unknown"}
FLOW_STAGE = {"request","access","sources","evidence","validation","operations",
              "result","answer"}
STATUS     = {"active","completed","failed"}

# ---- things that must NEVER reach a user-facing string ----------------------
LEAKS = [
    (r"\bpsycopg2\b", "driver name"), (r"\bsqlstate\b", "driver detail"),
    (r"\bpg_[a-z_]+", "postgres catalog"), (r"\bduckdb\b", "storage engine"),
    (r"\bselect\s+.*\bfrom\b", "SQL statement"),
    (r"\[(?:DB|DOC)\]", "internal provenance marker"),
    (r"\bX-Data-Scope\b", "internal header"),
    (r"\bsource_id\b", "internal identifier name"),
    (r"catalog error|binder error|parser error", "raw engine error"),
    (r'did you mean "', "engine catalog hint"),
]
# The generated SQL is published DELIBERATELY at exactly one path — the
# `sql.query` field of the explainability event, whose block is
# {"enabled": true, "query": "<sql>"} — because the user asked to see the
# statement that produced their numbers. That one path is exempt from the
# "SQL statement" leak rule and nothing else is: SQL turning up in a summary,
# a limitation, a step label, an answer, a warning or a source name is still
# an accident and still a leak. The key is the (rule description, exact path)
# pair rather than a prefix, so "anything under explainability" stays a leak.
LEAK_EXEMPT = {("SQL statement", "explainability.sql.query")}
# backend phase names — allowed ONLY inside `audit`
PHASES = ["supervisor_classify","supervisor_followup","source_selection",
          "schema_linking","sql_planning","rag_retrieve","rag_synthesize",
          "data_retrieval","result_preparation","access_check","execution_plan",
          "hybrid_retrieve","hybrid_synthesize","cross_source_processing",
          "tier2_intent","tier2_entity","tier2_columns","tier2_filters",
          "tier2_assemble","nosql_build","sql_probe"]

def strings(o, path="", skip=()):
    """Every string in the payload, with its path — skipping named subtrees."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in skip:
                continue
            yield from strings(v, f"{path}.{k}", skip)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from strings(v, f"{path}[{i}]", skip)
    elif isinstance(o, str):
        yield path, o

def parse(fn):
    evs = []
    for blk in open(fn).read().split("\n\n"):
        e = re.search(r"^event:\s*(\S+)", blk, re.M)
        d = re.search(r"^data:\s*(.*)$", blk, re.M | re.S)
        if e and d:
            try: evs.append((e.group(1), json.loads(d.group(1))))
            except Exception: pass
    return evs

def check(fn, label):
    V = []
    def bad(rule, detail): V.append((rule, detail))
    evs = parse(fn)
    kinds = [k for k, _ in evs]
    th  = [d for k, d in evs if k == "thinking"]
    fr  = [d["steps"] for d in th if isinstance(d.get("steps"), dict)]
    ex  = [d for k, d in evs if k == "explainability"]
    ct  = [d for k, d in evs if k == "content"]

    # ---- 1. terminal-event ordering -------------------------------------
    if "completed" in kinds and "content" in kinds:
        if kinds.index("content") > kinds.index("completed"):
            bad("ORDER", "content arrived AFTER completed")
    if ct and ex and kinds.index("explainability") < kinds.index("content"):
        pass  # explainability before content is intended

    # ---- 2. thinking model ----------------------------------------------
    lens = [len(f["steps"]) for f in fr]
    if lens != sorted(lens):
        bad("STEPS", f"the array shrank: {lens}")
    for f in fr:
        if f.get("total_steps") != 4:
            bad("STEPS", f"total_steps={f.get('total_steps')}, expected 4")
        if f.get("status") not in STATUS:
            bad("STEPS", f"status={f.get('status')!r} outside the closed set")
        ids = [s["id"] for s in f["steps"]]
        if ids != [i for i in STEP_IDS if i in ids]:
            bad("STEPS", f"steps out of canonical order: {ids}")
        if len(set(ids)) != len(ids):
            bad("STEPS", f"duplicate step ids: {ids}")
        for s in f["steps"]:
            if s["state"] not in STEP_STATE:
                bad("STEPS", f"state={s['state']!r} outside the closed set")
            if not (s.get("summary") or "").strip():
                bad("STEPS", f"{s['id']} has a blank summary")
            if s["state"] == "pending" and s.get("details"):
                bad("STEPS", f"{s['id']} is pending but already shows content")
            if s["state"] == "skipped" and s.get("details"):
                bad("STEPS", f"{s['id']} is skipped but shows content")
            labels = [r.get("label") for r in s.get("details") or []]
            if len(set(labels)) != len(labels):
                bad("STEPS", f"{s['id']} lists a duplicate detail: {labels}")
            for r in s.get("details") or []:
                if r.get("type") not in DETAIL_T:
                    bad("STEPS", f"detail type={r.get('type')!r} outside the closed set")
                if r.get("duration_ms") is not None and r["duration_ms"] < 0:
                    bad("STEPS", f"negative duration on {r.get('label')}")
            if s.get("duration_ms") is not None and s["duration_ms"] < 0:
                bad("STEPS", f"{s['id']} has a negative duration")
        if (f.get("execution") or {}).get("type") not in EXEC_T:
            bad("STEPS", f"execution.type={(f.get('execution') or {}).get('type')!r} outside the closed set")
        for k, v in (f.get("evidence") or {}).items():
            if not isinstance(v, int) or v < 0:
                bad("STEPS", f"evidence.{k}={v!r} is not a count")
    # terminal consistency
    if fr:
        last = fr[-1]
        if last.get("status") in ("completed", "failed"):
            if last.get("current_step") is not None:
                bad("TERMINAL", f"terminal frame still names current_step={last['current_step']!r}")
            for s in last["steps"]:
                if s["state"] == "active":
                    bad("TERMINAL", f"terminal frame still has {s['id']} active")
        term = [f for f in fr if f.get("status") in ("completed", "failed")]
        if len(term) > 1 and len({f["status"] for f in term}) > 1:
            bad("TERMINAL", "the turn reported two DIFFERENT terminal statuses")
    for d in th:
        if not (d.get("message") or "").strip():
            bad("LEGACY", f"blank legacy message on phase={d.get('phase')!r}")

    # A progress frame AFTER the turn has already reported its one terminal state
    # contradicts it. MISSED BY THIS CHECKER until 2026-09-11: it only compared the
    # order of `content` against `completed`, so a post-terminal `thinking` frame —
    # which is what `visualization_prep` was on every charted turn — sailed through.
    _seen_terminal = False
    for k, d in evs:
        if k != "thinking":
            continue
        _st = (d.get("steps") or {}).get("status")
        if _seen_terminal:
            bad("TERMINAL", f"a progress frame (phase={d.get('phase')!r}) arrived "
                            f"AFTER the turn reported a terminal status")
        if _st in ("completed", "failed"):
            _seen_terminal = True

    # ---- 3. explainability ----------------------------------------------
    for e in ex:
        if not e.get("version"):
            bad("EXPLAIN", "no version stamp")
        val = e.get("validation") or {}
        if isinstance(val, dict) and not val.get("checks") and val.get("passed") is True:
            bad("EXPLAIN", "validation.passed=true with an EMPTY check list (false assurance)")
        res = e.get("result") or {}
        rc = res.get("row_count")
        ev_rows = None
        if fr:
            ev_rows = (fr[-1].get("evidence") or {}).get("rows")
        if isinstance(rc, int) and isinstance(ev_rows, int) and rc != ev_rows:
            bad("EXPLAIN", f"result.row_count={rc} contradicts evidence.rows={ev_rows}")
        for s in e.get("sources") or []:
            if "id" in s:
                bad("EXPLAIN", f"user-facing source carries a raw id: {s}")
            if not s.get("name"):
                bad("EXPLAIN", f"source with no display name: {s}")
        # `limitations` is PROJECTED FROM `warnings`, so a limiting code present in
        # one and absent from the other means the two blocks were built at different
        # moments. Measured live: `warnings: ["no_results"]` beside `limitations: []`,
        # because the post-answer refresh updated warnings and not limitations.
        _LIMITING = {"no_results", "result_truncated", "restricted_data",
                     "partial_source_failure", "source_conflict",
                     "unmatched_records", "low_evidence"}
        _codes = {w.get("code") for w in (e.get("warnings") or [])
                  if isinstance(w, dict)} & _LIMITING
        if _codes and not (e.get("limitations") or []):
            bad("EXPLAIN", f"warnings carry {sorted(_codes)} but `limitations` is "
                           f"empty — the two blocks disagree about the same turn")
        for st in (e.get("flow") or {}).get("stages", []):
            if st.get("stage") not in FLOW_STAGE:
                bad("EXPLAIN", f"flow stage={st.get('stage')!r} outside the closed set")
        sql = e.get("sql") or {}
        if sql.get("enabled") is False and sql.get("query"):
            bad("EXPLAIN", "sql.enabled is false but a query is present")
        # The mirror case, which matters now that the SQL is published again:
        # enabled=true is the UI's cue to render a "view SQL" affordance, so an
        # empty or null query behind it reads to the user as "the statement
        # exists and we are withholding it". Both halves of the block have to
        # agree or the block is broken.
        if sql.get("enabled") is True and not (sql.get("query") or "").strip():
            bad("EXPLAIN", "sql.enabled is true but the query is missing or empty")
        if e.get("confidence") is not None and not isinstance(e["confidence"], (int, float)):
            bad("EXPLAIN", f"confidence={e['confidence']!r} is not a number")
        # v1 operations must agree with the flow the reader follows
        ops = [o.get("summary") for o in (e.get("operations") or [])
               if isinstance(o, dict)]
        fops = next((st.get("items") for st in (e.get("flow") or {}).get("stages", [])
                     if st.get("stage") == "operations"
                     and st.get("label") == "Operations applied"), None)
        if ops and fops is not None and list(fops) != ops[:8]:
            bad("EXPLAIN", f"flow operations {fops} disagree with v1 operations {ops}")

    # ---- 4. leakage (everything EXCEPT audit) ---------------------------
    for k, d in evs:
        for path, s in strings(d, k, skip=("audit", "trace", "timeline_summary")):
            low = s.lower()
            for pat, what in LEAKS:
                if (what, path) in LEAK_EXEMPT:
                    continue
                if re.search(pat, low):
                    bad("LEAK", f"{what} at {path}: {s[:90]!r}")
            for ph in PHASES:
                # a phase NAME as a standalone token, not as prose
                if re.search(rf"(?<![a-z_]){ph}(?![a-z_])", low) and path.split(".")[-1] not in ("phase",):
                    bad("LEAK", f"backend phase name '{ph}' at {path}: {s[:70]!r}")
    # answer text
    for c in ct:
        t = c.get("content") or ""
        if t.rstrip().endswith("STOP"):
            bad("ANSWER", "the answer ends with the prompt's own imperative token")
        if not t.strip():
            bad("ANSWER", "empty content block")

    print(f"{'✓' if not V else '✗'} {label}  ({len(V)} violation{'' if len(V)==1 else 's'})")
    for rule, detail in V:
        print(f"    [{rule}] {detail}")
    return len(V)

# --------------------------------------------------------------------- self-test
#: One payload with a KNOWN defect per rule family. If the checker stops catching
#: any of these it has gone blind, and a clean report means nothing.
def _self_test() -> int:
    import tempfile
    step = lambda i, idx, st, summ="ok", det=None: {
        "id": i, "index": idx, "title": i, "state": st, "duration_ms": 0,
        "summary": summ, "details": det or [], "expandable": bool(det)}
    bad = [
        ("thinking", {"phase": "x", "message": "", "steps": {
            "type": "thinking", "status": "running", "current_step": "finding",
            "total_steps": 3, "steps": [
                step("finding", 2, "active"),
                step("understanding", 1, "pending",
                     det=[{"type": "bogus", "label": "z", "state": "completed"}])],
            "evidence": {"rows": -1}, "execution": {"type": "weird"}, "timing": {}}}),
        ("thinking", {"phase": "y", "message": "m", "steps": {
            "type": "thinking", "status": "completed", "current_step": "finding",
            "total_steps": 4, "steps": [
                step("understanding", 1, "completed"), step("finding", 2, "active"),
                step("analyzing", 3, "skipped",
                     det=[{"type": "operation", "label": "q", "state": "completed"}]),
                step("preparing", 4, "completed", summ="", det=[
                    {"type": "operation", "label": "dup", "state": "completed"},
                    {"type": "operation", "label": "dup", "state": "completed"}])],
            "evidence": {"rows": 7}, "execution": {"type": "sql"}, "timing": {}}}),
        ("explainability", {
            "understanding": {"summary": "data_retrieval finished"},
            "validation": {"passed": True, "checks": []},
            "result": {"row_count": 99},
            "sources": [{"id": "2", "name": "homzhub"}, {"name": ""}],
            "flow": {"stages": [
                {"stage": "teleport", "label": "?"},
                {"stage": "operations", "label": "Operations applied",
                 "items": ["List records"]}]},
            "operations": [{"type": "retrieval", "summary": "Retrieved 5 passages"}],
            # enabled=false WITH a query is still an inconsistency (existing
            # rule), but the query string itself sitting at sql.query is no
            # longer a leak — that is the one place SQL is published on purpose,
            # so this planted case must now score EXPLAIN only, not LEAK.
            "sql": {"enabled": False, "query": "SELECT * FROM assets"},
            "confidence": "high",
            # The second limitation is the same statement in the WRONG field:
            # prose the user reads. Exempting sql.query must not exempt this.
            "limitations": ['Catalog Error: no table foo! Did you mean "bar"?',
                            "Skipped rows; ran select id from assets first"]}),
        # A second explainability frame carrying only the mirror defect: the
        # block advertises SQL (enabled=true) and then hands over nothing, which
        # the UI renders as a withheld statement. It is otherwise well-formed so
        # that it contributes exactly one planted defect.
        ("explainability", {"version": "1.0",
                            "sql": {"enabled": True, "query": None}}),
        # A third frame carrying only the cross-block disagreement: a LIMITING
        # warning code with an empty `limitations`. Otherwise well-formed, so it
        # contributes exactly one planted defect.
        ("explainability", {"version": "1.0",
                            "warnings": [{"code": "no_results",
                                          "severity": "warning", "message": "x"}],
                            "limitations": []}),
        # a progress frame after the terminal frame above — must be caught
        ("thinking", {"phase": "visualization_prep", "message": "Creating a visual…"}),
        ("completed", {"ok": True}),
        ("content", {"type": "markdown", "content": "[DOC] The answer. STOP"}),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".sse", delete=False) as fh:
        fh.write("\n\n".join(f"event: {k}\ndata: {json.dumps(v)}"
                              for k, v in bad) + "\n\n")
        path = fh.name
    print("--- self-test: every line below is a defect the checker MUST catch ---")
    caught = check(path, "SELF-TEST")
    # 25 -> 29: the checker already caught 28 here; the sql.query exemption
    # removed one of them (the planted SQL that is now published on purpose) and
    # two were added — SQL in a limitation, and the enabled=true/empty-query
    # mirror — so the floor moves to the 29 now measured.
    EXPECTED = 31          # raise this when a new rule adds a planted defect
    if caught < EXPECTED:
        print(f"\n✗ CHECKER IS BLIND: caught {caught}, expected at least {EXPECTED}")
        return 1
    print(f"\n✓ checker caught {caught} planted defects (>= {EXPECTED})")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--self-test"]
    if "--self-test" in sys.argv[1:]:
        sys.exit(_self_test())
    if not args:
        print(__doc__ or "", "\nusage: explainability_invariants.py FILE.sse[::label] ...")
        sys.exit(2)
    total = 0
    for arg in args:
        fn, _, label = arg.partition("::")
        total += check(fn, label or fn)
    print(f"\nTOTAL VIOLATIONS: {total}")
    sys.exit(min(total, 250))
