"""Eviction policies: LRU, RRIP, and the online learned policy.

The learned policy is the novel piece.  It scores every resident line with a
small linear model over per-line features and evicts the lowest-scoring line.
Weights start at the LRU point (score == age), then a perceptron-style
update runs online: whenever an evicted line is referenced again soon, that
eviction is judged "bad" and the weights move so lines with similar features
score higher next time.  No offline training, no trace-specific tuning -- the
policy adapts to whatever workload is actually running.

Feature vector for a line at time ``now``::

    [1.0, log1p(age), log1p(hit count), log1p(last reuse gap)]

where age is accesses since the line was last touched and the reuse gap is
the distance between its last two references.
"""

from __future__ import annotations

import math


class BasePolicy:
    """No-op hooks; concrete policies override what they need."""

    def choose_victim(self, lines, addr, now):  # pragma: no cover
        raise NotImplementedError

    def on_hit(self, line, now):
        pass

    def on_insert(self, line, now):
        pass

    def on_evict(self, line, now):
        pass

    def observe(self, addr, now, hit):
        pass


class LRUPolicy(BasePolicy):
    """Evict the least-recently-used line."""

    def choose_victim(self, lines, addr, now):
        return min(range(len(lines)), key=lambda i: lines[i].last_access)


class RRIPPolicy(BasePolicy):
    """Static RRIP (SRRIP): 2-bit re-reference prediction values.

    Insert at RRPV 2 (weak), hits promote to 0, eviction takes the first
    line at RRPV 3, ageing all lines when none qualifies.
    """

    MAX_RRPV = 3

    def choose_victim(self, lines, addr, now):
        while True:
            for i, line in enumerate(lines):
                if line.rrpv == self.MAX_RRPV:
                    return i
            for line in lines:
                line.rrpv = min(self.MAX_RRPV, line.rrpv + 1)

    def on_hit(self, line, now):
        line.rrpv = 0

    def on_insert(self, line, now):
        line.rrpv = 2


class LearnedPolicy(BasePolicy):
    """Online linear replacement policy with perceptron updates.

    ``weights`` starts at ``[0, -1, 0, 0]`` -- score equals negative
    log-age, so evicting the lowest-scoring line evicts the oldest line:
    the policy behaves exactly like LRU until it has evidence to do
    otherwise.
    ``window`` is how soon a re-reference must follow an eviction for that
    eviction to count as a mistake.
    """

    def __init__(self, lr: float = 0.5, window: int = 128):
        self.weights = [0.0, -1.0, 0.0, 0.0]
        self.lr = lr
        self.window = window
        self.evicted: list[tuple[int, list[float], int]] = []
        self.evictions = 0
        self.bad_evictions = 0

    # -- model ---------------------------------------------------------
    def features(self, line, now) -> list[float]:
        return [
            1.0,
            math.log1p(now - line.last_access),
            math.log1p(line.hits),
            math.log1p(line.last_gap),
        ]

    def score(self, line, now) -> float:
        feats = self.features(line, now)
        return sum(w * f for w, f in zip(self.weights, feats))

    # -- policy interface ----------------------------------------------
    def choose_victim(self, lines, addr, now):
        return min(range(len(lines)), key=lambda i: self.score(lines[i], now))

    def on_evict(self, line, now):
        self.evictions += 1
        self.evicted.append((line.addr, self.features(line, now), now))
        self.evicted = [
            e for e in self.evicted if now - e[2] <= self.window
        ]

    def observe(self, addr, now, hit):
        """Check whether this access re-references a recently evicted line.

        If so, the eviction was a mistake: move the weights so lines with
        those features score higher (are kept longer) in future.
        """
        for ev_addr, feats, ev_time in self.evicted:
            if ev_addr == addr and now - ev_time <= self.window:
                self.bad_evictions += 1
                self.weights = [
                    w + self.lr * f for w, f in zip(self.weights, feats)
                ]
                self.evicted = [
                    e for e in self.evicted if e[0] != addr
                ]
                return
