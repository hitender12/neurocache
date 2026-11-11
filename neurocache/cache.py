"""Set-associative cache simulator.

A cache is a set of sets; each set holds up to ``ways`` lines.  Policies are
pluggable objects that only decide *which* line to evict and maintain
per-line metadata hooks (``on_hit`` / ``on_insert`` / ``on_evict`` /
``observe``).  All line metadata (age, hit count, reuse gap, RRPV) lives on
the ``Line`` so the eviction policy can be swapped mid-run -- which is what
the fallback controller in :mod:`neurocache.monitor` relies on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Line:
    addr: int
    last_access: int = 0
    insert_time: int = 0
    hits: int = 0
    rrpv: int = 2
    last_gap: int = 0  # accesses between the line's last two references


class Cache:
    """A set-associative cache with a pluggable eviction policy."""

    def __init__(self, num_sets: int, ways: int, policy):
        if num_sets < 1 or ways < 1:
            raise ValueError("num_sets and ways must be >= 1")
        self.num_sets = num_sets
        self.ways = ways
        self.policy = policy
        self.sets: list[list[Line | None]] = [
            [None] * ways for _ in range(num_sets)
        ]
        self.accesses = 0
        self.hits = 0

    @property
    def misses(self) -> int:
        return self.accesses - self.hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def access(self, addr: int) -> bool:
        """Access ``addr``. Returns True on a hit, False on a miss."""
        self.accesses += 1
        now = self.accesses
        s = self.sets[addr % self.num_sets]

        for line in s:
            if line is not None and line.addr == addr:
                line.last_gap = now - line.last_access
                line.last_access = now
                line.hits += 1
                self.hits += 1
                self.policy.on_hit(line, now)
                self.policy.observe(addr, now, hit=True)
                return True

        # Miss: fill an empty slot if there is one.
        for i, line in enumerate(s):
            if line is None:
                s[i] = Line(addr=addr, last_access=now, insert_time=now)
                self.policy.on_insert(s[i], now)
                self.policy.observe(addr, now, hit=False)
                return False

        victim = self.policy.choose_victim(s, addr, now)
        evicted = s[victim]
        self.policy.on_evict(evicted, now)
        s[victim] = Line(addr=addr, last_access=now, insert_time=now)
        self.policy.on_insert(s[victim], now)
        self.policy.observe(addr, now, hit=False)
        return False

    def resident(self) -> set[int]:
        """Addresses currently held in the cache."""
        return {
            line.addr
            for s in self.sets
            for line in s
            if line is not None
        }
