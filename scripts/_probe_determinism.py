"""Throwaway probe: is the neural-policy eval bench deterministic? Bench one
fixed checkpoint twice on the same seeds and count per-seed outcome flips.
Run as a FILE (Windows spawn re-imports __main__, so it needs a guard)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phaseb_iter1_full import _bench  # noqa: E402


def main() -> None:
    n = int(os.environ.get("PROBE_N", "256"))
    jobs = int(os.environ.get("PROBE_JOBS", "12"))
    seeds = [f"{5300000 + i:07d}" for i in range(1, n + 1)]
    ckpt = ".data/phaseb_policy_iter1full.pt"
    r1 = _bench(ckpt, seeds, jobs)
    r2 = _bench(ckpt, seeds, jobs)
    w1 = sum(1 for w, _ in r1 if w)
    w2 = sum(1 for w, _ in r2 if w)
    odiff = sum(1 for (a, _), (b, _) in zip(r1, r2) if a != b)
    adiff = sum(1 for (_, a), (_, b) in zip(r1, r2) if a != b)
    print(f"PROBE n={n} jobs={jobs}")
    print(f"PROBE bench1_wins={w1} bench2_wins={w2}")
    print(f"PROBE win/loss flips between identical benches = {odiff}/{n} ({odiff/n:.1%})")
    print(f"PROBE ante differences = {adiff}/{n}")


if __name__ == "__main__":
    main()
