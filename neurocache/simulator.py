"""Experiment harness: run policies and the hybrid controller on a trace."""

from __future__ import annotations

from .cache import Cache
from .monitor import HybridCache
from .policies import LearnedPolicy, LRUPolicy, RRIPPolicy


def run_policy(policy, trace, num_sets=16, ways=4):
    """Run one policy over a trace; returns a result dict."""
    cache = Cache(num_sets, ways, policy)
    for addr in trace:
        cache.access(addr)
    return {
        "policy": type(policy).__name__,
        "hits": cache.hits,
        "misses": cache.misses,
        "hit_rate": cache.hit_rate,
    }


def run_hybrid(trace, num_sets=16, ways=4, window=512, slack=0.02):
    """Run the hybrid controller; returns live + shadow results."""
    hybrid = HybridCache(
        num_sets, ways, LearnedPolicy(), window=window, slack=slack
    )
    for addr in trace:
        hybrid.access(addr)
    return {
        "policy": "Hybrid(learned->LRU)",
        "mode": hybrid.mode,
        "fallbacks": hybrid.fallbacks,
        "fallback_at": hybrid.fallback_at,
        "hits": hybrid.live.hits,
        "misses": hybrid.live.misses,
        "hit_rate": hybrid.hit_rate,
        "shadow_hit_rate": hybrid.shadow_hit_rate,
        "shadow_misses": hybrid.shadow.misses,
    }


def compare(trace, num_sets=16, ways=4, **hybrid_kw):
    """Run LRU, RRIP, learned, and hybrid on the same trace."""
    results = [
        run_policy(LRUPolicy(), trace, num_sets, ways),
        run_policy(RRIPPolicy(), trace, num_sets, ways),
        run_policy(LearnedPolicy(), trace, num_sets, ways),
        run_hybrid(trace, num_sets, ways, **hybrid_kw),
    ]
    return results
