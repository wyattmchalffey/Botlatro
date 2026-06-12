# P2 Compute Decision: build a Rust run-level engine vs rent cores

*2026-06-12. The decision memo SUPERHUMAN_ROADMAP.md P2 requires before the
self-play phase starts. Verdict up front: **rent cores; do not build the
engine.** Revisit only if the triggers at the bottom fire.*

## The workload

P2 (self-play value iteration) needs on-policy run generation at scale:

- 10k-100k runs per iteration, 2-4 iterations expected (roadmap P2).
- Current local throughput (8-core box, 12-14 workers): ~9-10k solver-quality
  runs/day; the P1-augmented bot (deep-play delegation, if adopted) is
  slower per run, so call it **5-10k/day local**.
- One 50k-run iteration locally = **5-10 days of a fully-saturated box** that
  also can't run benches, gates, or diagnostics meanwhile (this week showed
  the box is the bottleneck for everything).

## Option A — full-Rust run-level engine

The only architecture that has not already measured ≤1x: GameState + sim +
seed-RNG + beam + shop all native, FFI once per RUN (RUST_PORT_PLAN's own
conclusion at line ~372).

- **What exists**: beam.rs + byte-identical native leaf (Phase 4d), py_random
  bit-exact RNG, the batched scorers, ~150-joker eval coverage.
- **What's missing** (verified by the 2026-06-10 hunt): rng/pseudohash.rs +
  luajit_tw223.rs (seed-faithful RNG is Python-only), the whole shop kernel
  (named "largest unported hot path"), beam divergence-spec items 2-7,
  whole-run orchestration (blind select/cash-out/packs/consumables).
- **Measured track record of Rust speed work here**: every per-decision
  hybrid ≤1x (native beam 0.92x, candidate-gen 0%, leaf wiring slower);
  cumulative wins came only from batched FFI + allocation removal, totalling
  2.2-4.8x across a month of phases. The plan's 10x+ rhetoric has never once
  survived measurement; agent estimates ran 3-5x optimistic.
- **Honest projection**: multi-week single-dev effort for a realistic 2-3x
  data-gen, with parity risk concentrated exactly in the late-ante surfaces
  (packs, vouchers, stochastic bosses) that P0.4 hasn't fixture-validated yet.

## Option B — rent cores

The sim is plain Python + a pip-installable Rust extension; data-gen is
embarrassingly parallel across seeds with JSONL output. Renting is *running
the same code elsewhere*, not a port.

- A 64-vCPU spot instance ≈ $0.7-1.5/hr ≈ **8x this box**. One 50k-run
  iteration: ~18-30 hours ≈ **$15-50**. Four iterations: under $200 — less
  than a day of the engineering time Option A consumes, with zero parity
  risk and the local box left free for gates and diagnostics.
- Scales linearly with instance count if iteration latency matters
  (4 instances → iteration in ~5-8 hours).
- One-time setup cost: an image/venv bootstrap script + seed-range sharding +
  result download. Days of work at most, reusable forever.

## Verdict

**Rent.** Option A's realistic 2-3x is worth less than $50 per iteration and
costs weeks; the measured record says the 10x dream is not real on this
codebase. Local engineering effort belongs in the P1 planner and the
quality-gated trims (data-gen beam_depth 3→2, SHOP_BEAM_WIDTH=2 for bulk
generation — both pre-vetted, gate-ready, and stack with renting).

## Reopen triggers (any one)

1. P2 ends up needing **>6 full regenerations** (engine amortizes).
2. Cloud spend is off the table for any reason.
3. A P2 design change makes per-decision latency critical (e.g. live MCTS at
   inference time rather than offline generation) — that's the one workload
   shape where a native engine wins by necessity, not economics.
