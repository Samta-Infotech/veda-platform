"""VEDA · Single-shot + REPL entry point."""
import os, re, sys, time, json, logging, threading
from config import SEMANTIC_MODEL_FILE
from veda.pipeline import run_query
from veda.runtime import _ENGINE, _prewarm_ollama, warm_up


def main():
    if not os.path.exists(SEMANTIC_MODEL_FILE):
        print("❌ Run ingestion first: python3 main.py --ingestion-only\n")
        return 1
    with open(SEMANTIC_MODEL_FILE) as f:
        sm = json.load(f)
    all_cols = list(sm.get("columns", {}).keys())

    args = [a for a in sys.argv[1:]]
    flags = {a for a in args if a.startswith("--")}

    # Warm-worker mode: load models ONCE, then answer one query per stdin line.
    # This is the cold-start fix for batch / programmatic use — a thin client pipes
    # queries in (echo "q" | … --serve, or a file) and never pays the engine build
    # per query. Existence/fast-path/full all work exactly as in single-shot.
    if "--serve" in flags:
        print("VEDA — warm worker  (one query per stdin line; blank or 'exit' to stop)")
        warm_up(verbose=True)
        print()
        try:
            for line in sys.stdin:
                q = line.strip()
                if not q or q.lower() in ("exit", "quit", "q"):
                    break
                try:
                    run_query(q, sm, all_cols)
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
        finally:
            if _ENGINE:
                _ENGINE.close()
        return 0

    # Single-shot mode: one query as args.
    query_args = [a for a in args if not a.startswith("--")]
    if query_args:
        print("\n" + "=" * 78 + "\nVEDA  —  Natural Language → SQL\n" + "=" * 78)
        query = " ".join(query_args)
        print(f"\n  Question:  {query}\n")
        # Overlap the Ollama model load with the engine build + retrieval for the
        # full-pipeline tail. Fast-path queries (count / metric / dimension) skip the
        # engine AND the LLM, so only pre-warm when this query is NOT a fast-path hit.
        try:
            from query.fast_path import try_fast_path as _tfp
            if _tfp(query) is None:
                threading.Thread(target=_prewarm_ollama, daemon=True).start()
        except Exception:
            pass
        rc = run_query(query, sm, all_cols)
        if _ENGINE:
            _ENGINE.close()
        return rc

    # Interactive REPL: load models ONCE, then each query is ~3-6s (no re-init).
    print("\n" + "=" * 78)
    print("VEDA  —  Interactive NL → SQL   (type a question, or 'exit')")
    print("=" * 78)
    print("  Loading models (one-time)…")
    warm_up(verbose=True)                  # engine + BGE-M3 + registries + LLM prewarm
    print()
    try:
        while True:
            try:
                query = input("veda> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            print()
            try:
                run_query(query, sm, all_cols)
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    finally:
        if _ENGINE:
            _ENGINE.close()
        print("\nbye.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
