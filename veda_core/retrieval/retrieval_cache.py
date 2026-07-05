# =============================================================================
# retrieval/retrieval_cache.py
# VEDA Phase 3f - Retrieval Result Caching
#
# Purpose:
#   Cache retrieval results by enriched query hash
#   TTL: 5 minutes
#   Hit rate: Expected 40-60% on typical workloads
#   Benefit: Reduces 500ms retrieval to 100ms for cached queries
#
# Input: Enriched tokens
# Output: Cached top-15 results (or None for cache miss)
#
# Status: Phase 3f
# =============================================================================

import hashlib
import json
import logging
import time
from typing import List, Tuple, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class RetrievalCache:
    """File-based cache for retrieval results (Redis fallback available)."""

    def __init__(
        self,
        cache_dir: str = "data/retrieval_cache",
        ttl_seconds: int = 300,  # 5 minutes
        use_redis: bool = False
    ):
        """
        Initialize retrieval cache.

        Args:
            cache_dir: Directory for file-based cache
            ttl_seconds: Time-to-live in seconds
            use_redis: Use Redis instead of files (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.use_redis = use_redis
        self.redis_client = None

        self._init_cache()

    def _init_cache(self):
        """Initialize cache backend."""
        if self.use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host="localhost",
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("✓ Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis unavailable, falling back to file cache: {e}")
                self.use_redis = False
        else:
            # File-based cache
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ File cache initialized at {self.cache_dir}")

    def _make_cache_key(self, enriched_tokens: List[str]) -> str:
        """
        Generate cache key from enriched tokens.

        Args:
            enriched_tokens: List of enriched tokens

        Returns:
            MD5 hash of sorted tokens
        """
        # Sort tokens for consistency
        sorted_tokens = "|".join(sorted(enriched_tokens))

        # Hash to MD5
        hash_obj = hashlib.md5(sorted_tokens.encode())
        cache_key = f"retrieval:{hash_obj.hexdigest()}"

        return cache_key

    def get(self, enriched_tokens: List[str]) -> Optional[List[Tuple[str, float]]]:
        """
        Get cached retrieval results.

        Args:
            enriched_tokens: List of enriched query tokens

        Returns:
            Cached top-15 results or None (cache miss)
        """
        cache_key = self._make_cache_key(enriched_tokens)

        if self.use_redis:
            return self._get_redis(cache_key)
        else:
            return self._get_file(cache_key)

    def _get_redis(self, cache_key: str) -> Optional[List[Tuple[str, float]]]:
        """Get from Redis cache."""
        try:
            cached_json = self.redis_client.get(cache_key)
            if cached_json:
                data = json.loads(cached_json)
                logger.info(f"✓ Cache hit: {cache_key}")
                return [(col_id, score) for col_id, score in data]
            else:
                logger.debug(f"✗ Cache miss: {cache_key}")
                return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    def _get_file(self, cache_key: str) -> Optional[List[Tuple[str, float]]]:
        """Get from file cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            logger.debug(f"✗ Cache miss: {cache_key}")
            return None

        # Check TTL
        age = time.time() - cache_file.stat().st_mtime
        if age > self.ttl_seconds:
            logger.debug(f"✗ Cache expired: {cache_key} (age {age:.0f}s)")
            cache_file.unlink()  # Delete expired
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
            logger.info(f"✓ Cache hit: {cache_key} (age {age:.0f}s)")
            return [(col_id, score) for col_id, score in data]
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None

    def set(
        self,
        enriched_tokens: List[str],
        results: List[Tuple[str, float]]
    ):
        """
        Cache retrieval results.

        Args:
            enriched_tokens: List of enriched tokens
            results: Top-K retrieval results
        """
        cache_key = self._make_cache_key(enriched_tokens)

        if self.use_redis:
            self._set_redis(cache_key, results)
        else:
            self._set_file(cache_key, results)

    def _set_redis(self, cache_key: str, results: List[Tuple[str, float]]):
        """Set in Redis cache."""
        try:
            data = [(col_id, score) for col_id, score in results]
            self.redis_client.setex(
                cache_key,
                self.ttl_seconds,
                json.dumps(data)
            )
            logger.debug(f"✓ Cached {cache_key} in Redis ({len(results)} results)")
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    def _set_file(self, cache_key: str, results: List[Tuple[str, float]]):
        """Set in file cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            data = [(col_id, score) for col_id, score in results]
            with open(cache_file, "w") as f:
                json.dump(data, f)
            logger.debug(f"✓ Cached {cache_key} to file ({len(results)} results)")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def clear(self):
        """Clear all cache."""
        if self.use_redis:
            try:
                self.redis_client.flushdb()
                logger.info("✓ Redis cache cleared")
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        else:
            import shutil
            try:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ File cache cleared")
            except Exception as e:
                logger.warning(f"File cache clear failed: {e}")

    def stats(self) -> Dict:
        """Get cache statistics."""
        if self.use_redis:
            try:
                info = self.redis_client.info()
                return {
                    "type": "redis",
                    "keys": self.redis_client.dbsize(),
                    "memory": info.get("used_memory_human", "unknown")
                }
            except:
                return {"type": "redis", "error": "unavailable"}
        else:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            return {
                "type": "file",
                "entries": len(cache_files),
                "size_bytes": total_size,
                "size_mb": total_size / 1024 / 1024
            }


# Global cache instance
_cache_instance = None


def get_cache(use_redis: bool = False) -> RetrievalCache:
    """Get or create global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RetrievalCache(use_redis=use_redis)
    return _cache_instance


def cache_retrieval_results(
    enriched_tokens: List[str],
    results: List[Tuple[str, float]],
    use_redis: bool = False
):
    """Standalone function to cache results."""
    cache = get_cache(use_redis)
    cache.set(enriched_tokens, results)


def get_cached_results(
    enriched_tokens: List[str],
    use_redis: bool = False
) -> Optional[List[Tuple[str, float]]]:
    """Standalone function to get cached results."""
    cache = get_cache(use_redis)
    return cache.get(enriched_tokens)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create cache
    cache = RetrievalCache(use_redis=False)

    # Example tokens
    tokens1 = ["show", "total", "payments", "amount"]
    results1 = [
        ("payment.amount", 0.95),
        ("payment.fee", 0.92),
        ("payment.total", 0.88),
    ]

    # Cache results
    print("\n1. Setting cache...")
    cache.set(tokens1, results1)

    # Retrieve from cache
    print("\n2. Getting from cache (hit)...")
    cached = cache.get(tokens1)
    if cached:
        print(f"   ✓ Got {len(cached)} cached results")
        for col_id, score in cached:
            print(f"     {col_id}: {score:.3f}")

    # Cache miss
    print("\n3. Getting different query (miss)...")
    tokens2 = ["different", "query"]
    result2 = cache.get(tokens2)
    print(f"   {result2}")

    # Cache stats
    print("\n4. Cache statistics:")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
