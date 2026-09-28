"""scripts/_flags.py — the one place every eval harness reads the ENGINE'S EFFECTIVE
flags from, instead of guessing at them from raw `os.environ` or trusting `.env`.

Why this exists (2026-09-26, wiring-fix pass D.1/D.2)
------------------------------------------------------
`veda_core/config.py` is the source of truth for every flag that shapes the pipeline
(docs/QUERY_PIPELINE_FLOW.md §0/§9). Its auto-`.env` load stays OFF (see the comment
block at the top of config.py for the decision and why) — so a harness invoked on the
host gets CODE DEFAULTS (`FRAME_PATH_ENABLED=False`, `SLM_MODEL_NAME=qwen2.5-coder:7b`,
…) unless the caller exported the right env vars first, while the SAME harness run
inside a container gets `.env` via `env_file:`. Two runs of "the same" harness — one on
each side — can silently grade two different pipelines. That is exactly the failure
mode `question-txt-lane-bypass` and `router-dead-missing-module` cost real eval time
to find after the fact.

This module makes that impossible to miss quietly:
  * `effective_flags()` reads config.py's ATTRIBUTES (the values the code actually
    uses — not `os.environ`, which config.py may have already re-defaulted or ignored).
  * `print_flags_header()` puts them at the top of every harness run's stdout.
  * every harness stamps the same dict into its results JSON / summary line, so a
    result file is self-describing months later.
  * `--expect KEY=VALUE` (repeatable) hard-aborts a run that is not wired the way the
    caller assumed, instead of quietly producing a number against the wrong pipeline.

Import from a script with the repo's `scripts/` directory already on `sys.path`
(every harness here already arranges that for its own sibling imports).
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

_REPO = Path(__file__).resolve().parents[1]

# name -> attribute read off veda_core/config.py. Kept as an explicit allow-list (not
# `vars(config)`) so this module documents exactly which flags it claims to track —
# see docs/QUERY_PIPELINE_FLOW.md §9 "Live flag snapshot" for the same set.
_FLAG_ATTRS = (
    "FRAME_PATH_ENABLED",
    "AGENT_PLANNER_ENABLED",
    "AGENT_JUDGE_MODE",
    "SLM_MODEL_NAME",
    "SLM_TEMPERATURE",
    "ROUTING_AUTHORITATIVE_MODES",
    "MULTISOURCE_ROUTING_SHADOW",
    "QUERY_DECOMPOSE_ENABLED",
    "COMPOUND_PART_BUDGET_S",
    "COMPOUND_TOTAL_BUDGET_S",
    "AGENT_PART_BUDGET_S",
    "AGENT_MAX_STEPS",
    "AGENT_MAX_TOOL_CALLS",
    "SLM_JOIN_TIMEOUT_SECS",
)


def _import_config():
    """`import config` the way every veda_core module does: veda_core in front of
    sys.path so a bare `import config` resolves to veda_core/config.py rather than
    the repo-root Django settings package of the same name (config/settings.py,
    config/celery.py, imported as top-level `config` by manage.py / gunicorn).

    Guards against the ambiguity directly: if something already claimed the name
    `config` in sys.modules and it is NOT veda_core's (no FRAME_PATH_ENABLED
    attribute), drop it and re-import fresh now that veda_core leads sys.path."""
    for p in ("/app/veda_core", str(_REPO / "veda_core")):
        if Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)
    cfg = sys.modules.get("config")
    if cfg is not None and not hasattr(cfg, "FRAME_PATH_ENABLED"):
        del sys.modules["config"]
        cfg = None
    if cfg is None:
        import config as cfg  # noqa: F401
    return cfg


def effective_flags(cfg=None) -> dict:
    """{name: value} for the flags that shape the pipeline, read from config.py's
    module attributes. Also includes 'OLLAMA_HOST' — the HOST ONLY of
    `SLM_OLLAMA_BASE_URL` (never the full URL out of an abundance of caution, though
    it carries no credentials today)."""
    cfg = cfg if cfg is not None else _import_config()
    out = {name: getattr(cfg, name, "<missing>") for name in _FLAG_ATTRS}
    base_url = getattr(cfg, "SLM_OLLAMA_BASE_URL", "") or ""
    out["OLLAMA_HOST"] = urlparse(base_url).hostname or base_url or "<unset>"
    return out


def flags_header(flags: dict, title: str = "effective flags") -> str:
    width = max((len(k) for k in flags), default=0)
    bar = "-" * (len(title) + 8)
    lines = [f"--- {title} ---"]
    lines += [f"  {k.ljust(width)} = {v!r}" for k, v in flags.items()]
    lines.append(bar)
    return "\n".join(lines)


def print_flags_header(flags: dict, title: str = "effective flags", file=None) -> None:
    print(flags_header(flags, title), file=file or sys.stdout, flush=True)


class FlagExpectationError(RuntimeError):
    """Raised by check_expect() when an --expect assertion fails."""


def parse_expect(pairs) -> dict:
    """['FRAME_PATH_ENABLED=1', 'AGENT_JUDGE_MODE=enforce'] -> {'FRAME_PATH_ENABLED':
    '1', 'AGENT_JUDGE_MODE': 'enforce'}. Raises FlagExpectationError on a malformed
    entry rather than silently dropping it."""
    out = {}
    for p in pairs or ():
        if "=" not in p:
            raise FlagExpectationError(f"--expect {p!r} is not KEY=VALUE")
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    return out


def _norm(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (tuple, list)):
        return ",".join(_norm(x) for x in v)
    return str(v)


def check_expect(flags: dict, expect: dict) -> None:
    """Raise FlagExpectationError, naming every mismatch, unless every `expect` key
    matches the effective value (compared as strings, so True/1/'1' all agree)."""
    bad = []
    for k, want in (expect or {}).items():
        got = flags.get(k, "<not tracked by effective_flags()>")
        if _norm(got) != _norm(want):
            bad.append(f"{k}: expected {want!r}, got {got!r}")
    if bad:
        raise FlagExpectationError(
            "flag expectation failed — this harness is not wired the way the "
            "caller assumed:\n  " + "\n  ".join(bad))


def add_expect_arg(parser) -> None:
    """Attach a repeatable `--expect KEY=VALUE` option to an argparse parser (or
    subparser)."""
    parser.add_argument(
        "--expect", action="append", default=[], metavar="KEY=VALUE",
        help="abort unless the effective flag KEY equals VALUE (repeatable)")


def enforce_expect(flags: dict, expect_args) -> None:
    """CLI convenience: parse --expect args against `flags` and sys.exit(1) with a
    readable message on mismatch, instead of a raised exception / traceback."""
    try:
        check_expect(flags, parse_expect(expect_args))
    except FlagExpectationError as e:
        sys.exit(str(e))


def local_flags_note() -> str:
    """For HTTP harnesses that talk to a live api/inference process over the network:
    there is no flags endpoint (adding one would touch inference/**, out of scope for
    this pass — see docs/QUERY_PIPELINE_FLOW.md). The values below are read from the
    LOCAL config.py import, which is not necessarily the container actually serving
    the request. Treat a mismatch between this and the observed behavior as a signal
    to check the container's own effective env, not a bug in the grader."""
    return ("NOTE: flags below are LOCAL only (no flags endpoint exists on "
            "api/inference) — they may not match the container that served "
            "this run's requests.")
