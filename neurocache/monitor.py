"""Regret monitor + fallback controller (the worst-case guarantee).

The hybrid cache runs the learned policy as its live policy and an LRU
policy as a *shadow* on the same trace.  A sliding window compares their
hit rates.  If the learned policy's hit rate falls below LRU's by more
than ``slack`` over a full window, the controller switches the live cache
to LRU and stays there.  That gives the guarantee:

    hybrid misses <= LRU misses + (one window of exploration) + slack

i.e. the learned policy can only ever cost you a bounded amount relative
to plain LRU, while it is free to win by as much as it can on workloads
where it is better.
"""

from __future__ import annotations

from .cache import Cache
from .policies import LRUPolicy


class RegretMonitor:
    """Windowed comparison of live vs. shadow hit rates."""

    def __init__(self, window: int = 512, slack: float = 0.02):
        self.window = window
        self.slack = slack
        self.live_hits = 0
        self.shadow_hits = 0
        self.n = 0
        self.windows_evaluated = 0

    def observe(self, live_hit: bool, shadow_hit: bool) -> bool:
        """Record one access. Returns True when a full window shows the
        live policy underperforming the shadow by more than ``slack``."""
        self.n += 1
        self.live_hits += bool(live_hit)
        self.shadow_hits += bool(shadow_hit)
        if self.n < self.window:
            return False
        self.windows_evaluated += 1
        live_rate = self.live_hits / self.window
        shadow_rate = self.shadow_hits / self.window
        regret = shadow_rate - live_rate
        self.live_hits = 0
        self.shadow_hits = 0
        self.n = 0
        return regret > self.slack


class HybridCache:
    """Live cache (learned policy, falls back to LRU) + shadow LRU cache."""

    def __init__(
        self,
        num_sets: int,
        ways: int,
        learned_policy,
        window: int = 512,
        slack: float = 0.02,
    ):
        self.live = Cache(num_sets, ways, learned_policy)
        self.shadow = Cache(num_sets, ways, LRUPolicy())
        self.monitor = RegretMonitor(window=window, slack=slack)
        self.mode = "learned"
        self.fallbacks = 0
        self.fallback_at: int | None = None

    def access(self, addr: int) -> bool:
        live_hit = self.live.access(addr)
        shadow_hit = self.shadow.access(addr)
        if self.mode == "learned" and self.monitor.observe(
            live_hit, shadow_hit
        ):
            self.mode = "lru"
            self.fallbacks += 1
            self.fallback_at = self.live.accesses
            self.live.policy = LRUPolicy()
        return live_hit

    @property
    def hit_rate(self) -> float:
        return self.live.hit_rate

    @property
    def shadow_hit_rate(self) -> float:
        return self.shadow.hit_rate
