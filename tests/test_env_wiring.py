"""Every key in .env must be READ by something. (Benchmark finding §0.3.1, 2026-09-23.)

WHY THIS EXISTS
---------------
There is no generic settings bridge in this repo — `grep -rn 'startswith("VEDA_")'`
returns nothing. A `.env` key only takes effect if some module reads that exact name.
On 2026-09-23 eleven keys were read by NOTHING:

    VEDA_TOP_K  VEDA_TOP_K_TO_LLM  VEDA_QUERY_ROUTER_ENABLED  VEDA_IR_JOIN_FREE_ENABLED
    VEDA_FAST_PATH_ENABLED  VEDA_QUERY_DECOMPOSE_ENABLED  VEDA_HNSW_M
    VEDA_HNSW_EF_CONSTRUCTION  VEDA_ENCODER_MODE  SLM_TEMPERATURE  (+ POSTGRES_HOST,
    POSTGRES_PORT, MODEL_CACHE_DIR found by this guard)

Each one's value happened to equal the hardcoded literal in config.py, so nothing
misbehaved — except SLM_TEMPERATURE, where .env said 0 and the live literal was 0.3.
That one made every document answer and every LLM-generated SQL non-deterministic:
measured, the same question produced 5 distinct answers in 5 runs.

storage_adapters/env_drift.py already compares .env against os.environ. That catches a
container created before an edit; it cannot catch a key nothing reads. This asserts the
stronger property.

WHAT COUNTS AS A READ
---------------------
A mention in CODE — .py, .yml, .sh, .ini, .conf, .sql. Deliberately NOT .md: a key
documented in an AGENTS.md but read by no code is exactly the failure this guards
against. env_drift.py itself is excluded (it names every key by definition).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = REPO / ".env"

# Directories/files searched for a reader.
SEARCH_ROOTS = [
    "veda_core", "apps", "inference", "storage_adapters", "chatbot", "config",
    "scripts", "docker", "manage.py", "docker-compose.yml", "docker-compose.prod.yml",
    "docker-compose.demo.yml", "docker-compose.mlflow.yml", "docker-compose.pg17.yml",
]
CODE_SUFFIXES = {".py", ".yml", ".yaml", ".sh", ".ini", ".conf", ".sql", ".cfg", ".toml"}

# env_drift names every key by construction, so it can never be the sole reader.
EXCLUDE_READERS = {"storage_adapters/env_drift.py"}


def _env_keys() -> list[str]:
    keys = []
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            keys.append(key)
    return keys


def _code_files() -> list[Path]:
    out = []
    for root in SEARCH_ROOTS:
        p = REPO / root
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for f in p.rglob("*"):
                if (f.is_file() and f.suffix in CODE_SUFFIXES
                        and "__pycache__" not in f.parts
                        and ".venv" not in f.parts):
                    rel = f.relative_to(REPO).as_posix()
                    if rel not in EXCLUDE_READERS:
                        out.append(f)
    return out


def _readers(key: str, files: list[Path]) -> list[str]:
    pat = re.compile(rf"\b{re.escape(key)}\b")
    hits = []
    for f in files:
        try:
            if pat.search(f.read_text(errors="ignore")):
                hits.append(f.relative_to(REPO).as_posix())
        except OSError:
            continue
    return hits


def test_env_file_exists():
    assert ENV_FILE.is_file(), (
        f"{ENV_FILE} not found — this guard needs the real .env "
        f"(it is gitignored, so CI must provide one)."
    )


def test_no_orphaned_env_keys():
    """Fail with the list of keys nothing reads."""
    files = _code_files()
    keys = _env_keys()
    assert keys, ".env parsed to zero keys — the parser is wrong, not the file."

    orphans = {k: _readers(k, files) for k in keys}
    orphans = {k: v for k, v in orphans.items() if not v}

    assert not orphans, (
        "These .env keys are read by NO code file. Either wire each to a named "
        "constant (use config._env_str/_env_int/_env_float/_env_bool) or delete the "
        "key — a key that reads as live but isn't is how SLM_TEMPERATURE=0 sat inert "
        "while the engine sampled at 0.3:\n  " + "\n  ".join(sorted(orphans))
    )


def test_no_generic_veda_settings_bridge():
    """Guards the ASSUMPTION the orphan test rests on.

    If someone later adds a generic `for k in os.environ: if k.startswith("VEDA_")`
    bridge, the orphan test above becomes misleading (keys would be live without a
    named reader). This fails loudly so the guard gets revisited rather than quietly
    testing the wrong property.
    """
    res = subprocess.run(
        ["grep", "-rn", "--include=*.py", r'startswith("VEDA_")', "veda_core", "apps",
         "config", "inference", "storage_adapters", "chatbot"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert res.returncode != 0 or not res.stdout.strip(), (
        "A generic VEDA_* env bridge now exists:\n" + res.stdout
        + "\nRevisit tests/test_env_wiring.py — its orphan check assumes there is none."
    )
