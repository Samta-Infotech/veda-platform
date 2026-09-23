import sys, os
sys.path.insert(0, "/app/veda_core"); sys.path.insert(0, "/app")
os.chdir("/app/veda_core")
from query.resolution import referent_tables
for label, tok in (("STRING (what master passed)", "payment"),
                   ("LIST   (what the branch passes)", ["payment"])):
    try:
        r = referent_tables(tok, {})
        print(f"{label:34s} -> returned {type(r).__name__} len={len(r)}")
    except Exception as e:
        print(f"{label:34s} -> RAISED {type(e).__name__}: {e}")
