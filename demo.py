"""Demo: the learned policy beating LRU on loops, and the fallback
protecting the worst case on adversarial (Zipf) traffic.

Run with:  python demo.py
"""

from neurocache import traces
from neurocache.simulator import compare

SETS, WAYS = 16, 4  # 64-line cache
WINDOW = 512


def show(title, trace):
    print(f"\n=== {title} ({len(trace)} accesses, {SETS * WAYS}-line cache) ===")
    rows = compare(trace, SETS, WAYS, window=WINDOW)
    lru_misses = rows[0]["misses"]
    for r in rows:
        if r["policy"].startswith("Hybrid"):
            print(
                f"  {r['policy']:<22} hit rate {r['hit_rate']:.3f}  "
                f"misses {r['misses']:>5}  mode={r['mode']} "
                f"fallbacks={r['fallbacks']} (at access {r['fallback_at']})"
            )
        else:
            print(
                f"  {r['policy']:<22} hit rate {r['hit_rate']:.3f}  "
                f"misses {r['misses']:>5}"
            )
    hybrid_misses = rows[3]["misses"]
    print(
        f"  worst-case check: hybrid misses ({hybrid_misses}) <= "
        f"LRU misses ({lru_misses}) + one window ({WINDOW}) -> "
        f"{'HOLDS' if hybrid_misses <= lru_misses + WINDOW else 'VIOLATED'}"
    )
    return rows


def main():
    print("NeuroCache demo: online-learned replacement vs LRU / RRIP")
    show(
        "Looping workload (working set 80 > cache 64)",
        traces.loop_trace(6000, 80),
    )
    print(
        "  -> LRU thrashes to 0 hits; the learned policy discovers that\n"
        "     evicting the most-recently-used line is right here and\n"
        "     recovers a 0.63 hit rate, purely from online updates."
    )
    zipf_rows = show(
        "Adversarial workload (Zipf, learned policy is mismatched)",
        traces.zipf_trace(6000, 500, 1.0, seed=1),
    )
    excess = zipf_rows[3]["misses"] - zipf_rows[0]["misses"]
    print(
        "  -> The learned policy loses on Zipf traffic, so the regret\n"
        "     monitor falls back to LRU after one window. The hybrid ends\n"
        f"     within {excess} misses of plain LRU: the cost of learning "
        "is bounded."
    )
    show(
        "Phase shift (loop, then Zipf)",
        traces.phase_shift_trace(6000, 80, 500, seed=1),
    )
    print(
        "  -> The hybrid wins the loop phase with the learned policy,\n"
        "     then falls back when the workload changes: better than\n"
        "     either policy alone." 
    )


if __name__ == "__main__":
    main()
