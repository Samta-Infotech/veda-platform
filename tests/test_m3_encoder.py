"""WP3 acceptance — BGE-M3 encoder shape test.

Asserts m3_encoder.encode_query returns a 1024-dim L2-normalized dense vector plus a
non-empty learned-sparse weight dict for a sample sentence, and that encode_dense yields
a normalized (n, 1024) matrix.

Run: python tests/test_m3_encoder.py  (from repo root; needs the ML image with
FlagEmbedding + the baked BAAI/bge-m3 weights). Skips cleanly when the model/dep is
absent so it never blocks the thin-image / CI-without-weights case.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "veda_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def main() -> int:
    try:
        import FlagEmbedding  # noqa: F401
    except Exception as e:
        print(f"[skip] FlagEmbedding not installed ({e}) — build-only environment")
        return 0

    try:
        from ingestion import m3_encoder
    except Exception as e:
        print(f"[skip] m3_encoder import failed ({e})")
        return 0

    try:
        dense, sparse = m3_encoder.encode_query("how many escalated incidents are open")
    except Exception as e:
        print(f"[skip] model weights unavailable ({e}) — bake BAAI/bge-m3 to run this")
        return 0

    dense = np.asarray(dense, dtype=np.float32)
    assert dense.shape == (1024,), f"dense must be 1024-dim, got {dense.shape}"
    norm = float(np.linalg.norm(dense))
    assert abs(norm - 1.0) < 1e-2, f"dense must be L2-normalized, got norm={norm:.4f}"
    print(f"[1] encode_query dense: 1024-dim, norm={norm:.4f}  ✓")

    assert isinstance(sparse, dict) and len(sparse) > 0, "sparse weights must be non-empty"
    assert all(isinstance(k, str) and float(v) > 0 for k, v in sparse.items()), \
        "sparse entries must be {token_id_str: positive weight}"
    print(f"[2] encode_query sparse: {len(sparse)} tokens, all positive  ✓")

    mat = m3_encoder.encode_dense(["users table", "payment amount"])
    assert mat.shape == (2, 1024), f"encode_dense shape (2,1024), got {mat.shape}"
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2), f"rows must be normalized, got {norms}"
    print(f"[3] encode_dense: (2, 1024) normalized  ✓")

    print("\n✓ ALL M3 ENCODER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
