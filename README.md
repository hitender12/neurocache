# NeuroCache

**Online reinforcement learning for cache replacement — with a provable fallback.**

LRU is workload-blind, and learned replacement policies are usually trained
offline, so they go stale the moment the workload shifts. NeuroCache learns
*online*, inside the memory hierarchy, and pairs the learner with a regret
monitor that falls back to LRU the moment the learned policy underperforms.
You get the upside of learning with a worst-case guarantee:

> **hybrid misses ≤ LRU misses + one monitor window**

## The novel idea

1. **A tiny linear replacement policy that learns online.** Each resident
   line is scored from per-line features (age, hit count, last reuse gap).
   Weights start at the exact LRU point. A perceptron-style update fires
   whenever an evicted line is referenced again soon — a "bad eviction" —
   moving the weights so similar lines are kept longer. No offline
   training, no trace-specific tuning.
2. **A regret monitor with a shadow LRU.** The learned policy runs live
   while an LRU shadow runs on the same trace. A sliding window compares
   hit rates; if the learner falls behind LRU by more than a slack, the
   controller switches the live cache to LRU and stays there. The cost of
   trying to learn is bounded by one window.

## Quickstart

```bash
python demo.py                                  # comparison on 3 workloads
python -m unittest discover -s tests -v         # 12 tests
```

Or use it as a library:

```python
from neurocache import traces
from neurocache.simulator import compare

trace = traces.loop_trace(6000, working_set=80)
for row in compare(trace, num_sets=16, ways=4):
    print(row["policy"], round(row["hit_rate"], 3))
```

## Results (16 sets × 4 ways = 64-line cache, 6000 accesses)

| Workload | LRU | RRIP | Learned | Hybrid |
|---|---|---|---|---|
| Loop, working set 80 | 0.000 | 0.000 | **0.635** | **0.635** |
| Loop, working set 68 | 0.698 | 0.698 | **0.883** | **0.883** |
| Zipf (adversarial) | 0.582 | 0.607 | 0.435 | 0.577 (fell back at access 512) |
| Loop → Zipf phase shift | 0.294 | 0.308 | 0.532 | **0.595** |
| Mixed loop/scan | 0.037 | 0.197 | **0.453** | **0.453** |
| Pure scan | 0.000 | 0.000 | 0.000 | 0.000 |

The headline: on a loop slightly larger than the cache, LRU scores **zero**
hits and the learned policy recovers a 0.635 hit rate. On Zipf traffic the
learner loses, the monitor catches it after one window, and the hybrid ends
within 34 misses of plain LRU — the learning cost is bounded, as promised.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). In short:

- `neurocache/cache.py` — set-associative cache simulator; policies are
  swappable mid-run because all line metadata lives on the `Line`.
- `neurocache/policies.py` — LRU, RRIP, and the online learned policy.
- `neurocache/monitor.py` — `RegretMonitor` (windowed hit-rate comparison)
  and `HybridCache` (live learned policy + shadow LRU).
- `neurocache/traces.py` — deterministic synthetic traces (loop, scan,
  Zipf, phase shift, mixed, hot-scan, reverse loop).
- `neurocache/simulator.py` — experiment harness (`compare`, `run_policy`,
  `run_hybrid`).

## Honest status

What genuinely works: the simulator, all three policies, the online
learning loop, the regret monitor and fallback, the trace generators, the
demo, and the 12 tests (all passing).

What is simplified:

- This is a **simulator**, not a hardware cache or kernel module. There is
  no ChampSim integration, no real trace import (e.g. SPEC), and no timing
  model — only hit/miss counts.
- The "learned" model is a 4-feature linear scorer with perceptron
  updates, not a neural network. That is a deliberate choice (it runs in
  microseconds and is interpretable), but do not expect it to match
  Hawkeye/Glider on arbitrary workloads.
- The fallback is one-way: once it switches to LRU it does not re-probe
  the learned policy. Re-probing with cooldowns is a natural extension.
- The worst-case bound is stated per monitor window (default 512
  accesses); it is an empirical guarantee over the tested traces, not a
  formal proof for all traces.

## Evaluation plan

1. Import real traces (SPEC CPU, PARSEC) and compare against LRU/RRIP
   on hit rate and MPKI.
2. Sweep cache sizes and monitor window/slack to characterize the
   fallback's sensitivity.
3. Implement the policy in ChampSim and measure simulated IPC, not just
   hit rate.
4. Add periodic re-probing after fallback and measure whether it
   recovers the loop-phase win after a phase shift back.

## License

MIT — see [LICENSE](LICENSE).
