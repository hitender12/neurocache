"""Tests for NeuroCache.

Run with:  python -m unittest discover -s tests -v
"""

import unittest

from neurocache.cache import Cache
from neurocache.monitor import HybridCache
from neurocache.policies import LearnedPolicy, LRUPolicy, RRIPPolicy
from neurocache.simulator import compare, run_hybrid, run_policy
from neurocache import traces

SETS, WAYS = 16, 4  # 64-line cache


class TestCacheBasics(unittest.TestCase):
    def test_lru_hit_miss_counts(self):
        # 2 sets x 2 ways. 0,2 -> set 0; 1,3 -> set 1.
        cache = Cache(2, 2, LRUPolicy())
        for addr in [0, 1, 2, 3, 0, 1, 2, 3]:
            cache.access(addr)
        self.assertEqual(cache.hits, 4)
        self.assertEqual(cache.misses, 4)

    def test_lru_evicts_least_recently_used(self):
        cache = Cache(1, 2, LRUPolicy())
        # 0 in, 1 in, 2 in (evicts 0), 0 in (evicts 1).
        for addr in [0, 1, 2, 0]:
            cache.access(addr)
        self.assertEqual(cache.hits, 0)  # 0 was evicted before its reuse
        self.assertNotIn(1, cache.resident())
        self.assertIn(0, cache.resident())
        self.assertIn(2, cache.resident())

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            Cache(0, 4, LRUPolicy())
        with self.assertRaises(ValueError):
            Cache(16, 0, LRUPolicy())

    def test_rrip_beats_lru_on_mixed_trace(self):
        trace = traces.mixed_trace(6000, 80, seed=1)
        lru = run_policy(LRUPolicy(), trace, SETS, WAYS)
        rrip = run_policy(RRIPPolicy(), trace, SETS, WAYS)
        self.assertGreater(rrip["hit_rate"], lru["hit_rate"])


class TestLearnedPolicy(unittest.TestCase):
    def test_learned_starts_as_lru(self):
        """With learning rate 0 the weights never move, so the policy must
        behave exactly like LRU."""
        trace = traces.zipf_trace(3000, 500, 1.0, seed=1)
        lru = run_policy(LRUPolicy(), trace, SETS, WAYS)
        frozen = run_policy(LearnedPolicy(lr=0.0), trace, SETS, WAYS)
        self.assertEqual(frozen["hits"], lru["hits"])

    def test_learned_beats_lru_on_loop(self):
        """The headline result: on a loop slightly larger than the cache,
        LRU scores ~0 hits while the learned policy learns to evict the
        most-recently-used line and recovers most of the working set."""
        trace = traces.loop_trace(6000, 80)
        lru = run_policy(LRUPolicy(), trace, SETS, WAYS)
        learned = run_policy(LearnedPolicy(), trace, SETS, WAYS)
        self.assertAlmostEqual(lru["hit_rate"], 0.0, places=3)
        self.assertGreater(learned["hit_rate"], lru["hit_rate"] + 0.3)

    def test_learned_improves_over_time(self):
        """Learning curve: hit rate after the first 100 accesses (cold
        start, still LRU-like) must exceed the first 100 accesses by a
        wide margin once the policy has adapted to the loop."""
        trace = traces.loop_trace(6000, 80)
        cache = Cache(SETS, WAYS, LearnedPolicy())
        hits_at_100 = None
        for i, addr in enumerate(trace, start=1):
            cache.access(addr)
            if i == 100:
                hits_at_100 = cache.hits
        first_rate = hits_at_100 / 100
        later_rate = (cache.hits - hits_at_100) / 5900
        self.assertGreater(later_rate, first_rate + 0.2)

    def test_learned_records_bad_evictions(self):
        policy = LearnedPolicy()
        cache = Cache(SETS, WAYS, policy)
        for addr in traces.loop_trace(2000, 80):
            cache.access(addr)
        self.assertGreater(policy.evictions, 0)
        self.assertGreater(policy.bad_evictions, 0)
        self.assertNotEqual(policy.weights, [0.0, -1.0, 0.0, 0.0])


class TestFallbackGuarantee(unittest.TestCase):
    def test_fallback_triggers_when_learned_loses(self):
        """On Zipf traffic the learned policy is worse than LRU, so the
        regret monitor must switch the live cache to LRU."""
        trace = traces.zipf_trace(6000, 500, 1.0, seed=1)
        result = run_hybrid(trace, SETS, WAYS)
        self.assertEqual(result["mode"], "lru")
        self.assertEqual(result["fallbacks"], 1)
        self.assertIsNotNone(result["fallback_at"])

    def test_worst_case_bound_holds(self):
        """Hybrid misses never exceed LRU misses by more than one monitor
        window (the bounded cost of exploring the learned policy)."""
        window = 512
        for trace in [
            traces.zipf_trace(6000, 500, 1.0, seed=1),
            traces.zipf_trace(6000, 500, 0.8, seed=1),
            traces.phase_shift_trace(6000, 80, 500, seed=1),
            traces.scan_trace(6000),
            traces.reverse_loop_trace(6000, 80),
        ]:
            lru = run_policy(LRUPolicy(), trace, SETS, WAYS)
            hybrid = run_hybrid(trace, SETS, WAYS, window=window)
            self.assertLessEqual(
                hybrid["misses"],
                lru["misses"] + window,
                f"bound violated on trace with LRU misses={lru['misses']}, "
                f"hybrid misses={hybrid['misses']}",
            )

    def test_no_fallback_on_loop(self):
        """Where the learned policy wins, the monitor must not intervene."""
        trace = traces.loop_trace(6000, 80)
        result = run_hybrid(trace, SETS, WAYS)
        self.assertEqual(result["mode"], "learned")
        self.assertEqual(result["fallbacks"], 0)
        self.assertGreater(result["hit_rate"], 0.5)

    def test_hybrid_beats_lru_on_phase_shift(self):
        """The hybrid gets the best of both: learned during the loop
        phase, LRU after the workload shifts."""
        trace = traces.phase_shift_trace(6000, 80, 500, seed=1)
        lru = run_policy(LRUPolicy(), trace, SETS, WAYS)
        hybrid = run_hybrid(trace, SETS, WAYS)
        self.assertGreater(hybrid["hit_rate"], lru["hit_rate"])


if __name__ == "__main__":
    unittest.main()
