"""Synthetic address-trace generators.

Every generator is deterministic for a given seed, so experiments and tests
are reproducible.  Traces are lists of integer addresses.
"""

from __future__ import annotations

import random


def loop_trace(length: int, working_set: int, start: int = 0) -> list[int]:
    """Cycle through ``working_set`` addresses forever.

    The classic LRU killer: when the working set is slightly larger than
    the cache, LRU evicts exactly the line that is about to be reused and
    scores ~0 hits.
    """
    return [start + (i % working_set) for i in range(length)]


def scan_trace(length: int, start: int = 0) -> list[int]:
    """Streaming trace: every address is new, no reuse at all."""
    return [start + i for i in range(length)]


def zipf_trace(
    length: int, n_items: int, alpha: float = 1.0, seed: int = 0
) -> list[int]:
    """Random accesses with Zipf popularity (a few hot addresses)."""
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** alpha for i in range(n_items)]
    return rng.choices(range(n_items), weights=weights, k=length)


def reverse_loop_trace(length: int, working_set: int) -> list[int]:
    """Forward pass then reverse pass, repeating (palindrome-ish scan)."""
    cycle = list(range(working_set)) + list(range(working_set - 1, -1, -1))
    return [cycle[i % len(cycle)] for i in range(length)]


def hot_scan_trace(
    length: int, hot_addr: int = 0, hot_every: int = 100, seed: int = 0
) -> list[int]:
    """A stream of unique addresses with one hot address revisited often."""
    out = []
    nxt = 1
    for i in range(length):
        if i % hot_every == 0:
            out.append(hot_addr)
        else:
            out.append(nxt)
            nxt += 1
    return out


def phase_shift_trace(
    length: int,
    working_set: int,
    n_items: int,
    alpha: float = 1.0,
    seed: int = 0,
) -> list[int]:
    """First half: tight loop (teaches the learner one regime).
    Second half: Zipf random (a different regime -- the learned model
    trained on loops is mismatched here, which is what the regret monitor
    is designed to catch).
    """
    half = length // 2
    return loop_trace(half, working_set) + zipf_trace(
        length - half, n_items, alpha=alpha, seed=seed
    )


def mixed_trace(length: int, working_set: int, seed: int = 0) -> list[int]:
    """Alternating blocks of looping and scanning."""
    rng = random.Random(seed)
    out = []
    while len(out) < length:
        block = rng.choice(["loop", "scan"])
        n = min(rng.randint(200, 600), length - len(out))
        if block == "loop":
            out.extend(loop_trace(n, working_set))
        else:
            out.extend(scan_trace(n, start=rng.randint(0, 10**6)))
    return out
