# NeuroCache — Architecture

## Components

```
demo.py                  runnable comparison on 3 workloads
neurocache/
  cache.py               set-associative cache; policy plug-in point
  policies.py            LRU, RRIP, LearnedPolicy
  monitor.py             RegretMonitor, HybridCache
  traces.py              deterministic synthetic trace generators
  simulator.py           experiment harness
tests/
  test_neurocache.py     12 tests (unittest, stdlib only)
```

## Design decisions

**Line metadata lives on the `Line`, not in the policy.** The cache owns
`last_access`, `hits`, `last_gap`, and `rrpv`. Policies only read that
state and decide evictions. This is what makes mid-run policy switching
cheap and safe: the fallback controller swaps `cache.policy` and the new
policy immediately sees consistent state.

**The learned policy is linear, not a neural net.** Four features
(`[1, log1p(age), log1p(hits), log1p(last reuse gap)]`), one weight
vector, perceptron updates. Rationale: eviction decisions happen on every
miss, so the model must be microseconds-fast and its behavior explainable.
The feature set is small enough to reason about and large enough to
express "keep old, frequently-hit lines" vs "evict the just-used line".

**Weights start at the LRU point.** Initial weights `[0, -1, 0, 0]` make
the score `-log(age)`, so evicting the argmin evicts the oldest line —
exactly LRU. Learning is therefore strictly an *improvement* process
from a sane default, and with learning rate 0 the policy is bit-identical
to LRU (covered by a test).

**Bad evictions are the only training signal.** When a line is evicted
we record its features and eviction time. If the address is referenced
again within `window` accesses, the eviction was a mistake and the
weights move toward that line's features. This is the same signal
Hawkeye uses (a line evicted and soon re-referenced was cache-friendly),
reduced to a perceptron update. There is no reward shaping, no replay
buffer, and no offline phase.

**The monitor compares against a live shadow, not a fixed threshold.**
A fixed "hit rate must stay above X" rule would false-trigger on hard
workloads (scans score 0 for every policy). Comparing against a shadow
LRU on the same trace isolates *regret*: the learner is only punished
for doing worse than the classical baseline, on the same accesses.

**Fallback is one-way.** Once the controller switches to LRU it stays
there for the run. This keeps the worst-case argument simple (at most
one window of regret plus the cost of converging to LRU behavior), at
the price of not re-probing the learner if the workload shifts back.
Re-probing is the obvious next extension; the monitor already exposes
`windows_evaluated` to build it on.

## Data flow

```
trace ──► HybridCache.access(addr)
              ├─► live Cache (learned policy, or LRU after fallback)
              └─► shadow Cache (LRU, always)
         monitor.observe(live_hit, shadow_hit)
              └─► regret > slack over a full window → switch live to LRU
```

## Invariants tested

- LRU eviction order and hit/miss counts on hand-computed sequences.
- Frozen learned policy (lr=0) is bit-identical to LRU.
- Learned policy beats LRU by > 0.3 hit rate on the loop workload and
  its hit rate rises steeply after the first 100 accesses.
- Fallback triggers on Zipf traffic and never triggers on loops.
- Hybrid misses ≤ LRU misses + one window on five traces (scan, Zipf ×2,
  phase shift, reverse loop).
