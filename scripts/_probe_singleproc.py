"""Isolate the eval nondeterminism source: run the SAME (seed, ckpt) twice in a
SINGLE process (no ProcessPool). If flips==0 here, the bench logic (sim+policy)
is deterministic and the run-to-run noise lives in the parallel/threading layer;
if flips>0, something in sim/policy is itself nondeterministic."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phaseb_iter1_full import _eval_seed  # noqa: E402


def main() -> None:
    ckpt = ".data/phaseb_policy_iter1full.pt"
    n = int(os.environ.get("PROBE_N", "24"))
    seeds = [f"{5300000 + i:07d}" for i in range(1, n + 1)]
    flips = 0
    for s in seeds:
        r1 = _eval_seed((s, ckpt))
        r2 = _eval_seed((s, ckpt))
        if r1 != r2:
            flips += 1
            print(f"FLIP seed {s}: {r1} vs {r2}")
    print(f"SINGLE-PROC: {flips}/{n} seeds nondeterministic when run twice in one process")


if __name__ == "__main__":
    main()
