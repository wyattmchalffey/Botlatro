# Phase 8–15: The Superhuman Bot (Neural-Guided Search)

**Status:** Active (started 2026-05-31). This is the end-goal architecture
plan — the path from "exact forward model + hand-tuned heuristic search" to a
**superhuman, self-improving, neural-guided planner.** It supersedes the
*offline-solver-as-end-goal* framing in
[`PHASE7_OFFLINE_SOLVER_PLAN.md`](PHASE7_OFFLINE_SOLVER_PLAN.md): the solver
and `basic_strategy_bot` are now **bootstraps**, not the destination.

**Cross-refs:** [`PLAN.md`](PLAN.md) (phases 8–15), the Rust core
([`RUST_PORT_PLAN.md`](RUST_PORT_PLAN.md)), the sim/RNG foundation
([`PHASE7_OFFLINE_SOLVER_PLAN.md`](PHASE7_OFFLINE_SOLVER_PLAN.md)).

---

## 1. The goal and the key reframe

Goal: a bot that plays Balatro at superhuman strength (win rates and scores
beyond strong human play across random seeds and high stakes).

**The reframe that makes this tractable:** with seed-faithful RNG, Balatro is a
**deterministic, perfect-information, single-agent planning problem** — once the
seed is fixed, every shop, boss, pack, and draw is known. There is no opponent
and there are no chance nodes. So this is *not* chess (adversarial → minimax);
it is closer to **optimal solitaire / a deterministic puzzle solver**. The right
algorithms are **best-first / A* / IDA* / PUCT with a strong learned heuristic**,
not minimax.

The single hardest part of a strong game engine — a perfect, fast forward model
with make/unmake — is **already built** (`forward_sim`, 99.9% exact, Rust-backed).
What's missing is the *brain*: a learned evaluation and a search that uses it.

**Why "faster" and "stronger" are the same change.** Today the leaf evaluator is
a greedy rollout (`state_value.clear_probability`) plus a saturating hand-tuned
shop score. Rollouts are the expensive *and* weak way to evaluate a position
(28%+ of runtime; can't see late-game scaling → the ante-8 ceiling). Every
strong modern engine deleted rollouts in favor of a **cheap learned static
evaluation** (Stockfish-NNUE; AlphaZero value+policy net). That one change makes
search simultaneously *cheaper* (µs eval vs ms rollout) and *better-guided*
(learned eval captures synergy/scaling). We adopt the same move.

---

## 2. What we already have (the moat)

- **Exact forward model:** `search/forward_sim.py` + `rust/botlatro-core`
  (PyO3), 99.9% exact vs the real game on the audited surface; ~150 jokers, all
  bosses, vouchers, tags, packs modeled against extracted Lua source.
- **Seed-faithful RNG:** `rng/` — deck, shop, boss, vouchers, tags, packs,
  rerolls, mid-hand procs, created-card identities. Makes the game
  perfect-information from a seed string (bridge-validated to ante ~5).
- **Pure-sim runner + dataset CLI:** `sim/local_runner.py`,
  `dataset/cli.py` (multiprocessing, resumable JSONL).
- **Two bootstrap teachers:** `basic_strategy_bot` (~17–23% white win, fast)
  and the beam `SolverPolicy` (~8%, slower). Either can seed imitation data.
- **Methodology that works:** process_time A/B (`scripts/datagen_speed.py`),
  ≥96-seed quality gates, parity tests, pure-Python source-of-truth with
  native accelerators.

The cost prerequisite that usually kills these projects (a trustworthy
simulator) is paid. We build the brain on top.

---

## 3. Target architecture

Five components, mapped to a modern engine:

| Component | Engine analog | Status |
|---|---|---|
| **State encoder** — `GameState` → learnable tensors (raw joker/card/shop identities, editions, counters, hand levels, deck comp), versioned | NNUE/AZ input features | **✓ done (Stage 0)** |
| **Value head** — state → P(win) / expected ante; cheap forward pass replaces the rollout leaf | NNUE static eval | not started |
| **Policy head** — state → action priors; prunes search to top-k | AZ policy prior | not started |
| **Selective search** — best-first / A* / PUCT over the *single-agent* tree, value+policy guided, tree-reuse across in-run decisions | AZ MCTS / αβ search | not started |
| **Self-play loop** — net guides search → search yields better-than-net targets → retrain → repeat | AZ training loop | not started |
| **Substrate** — batched net inference across parallel workers; encoder/eval in/near Rust | Leela batched NN eval | not started |

Design rules (non-negotiable, "done right"):
- **Raw inputs, not heuristic aggregates.** The 2026-05-29 value-head probe
  failed because it re-weighted the heuristic's own collapsed signals. The net
  must see joker identities, editions, counters, deck composition, hand levels,
  shop contents — things the heuristic throws away.
- **Pure-Python source of truth; native/torch are accelerators** behind parity
  tests. The encoder is dependency-free (stdlib) so it tests under `unittest`.
- **Everything versioned** (`ENCODING_VERSION`, model checkpoints carry the
  encoder version) so schema drift is detectable.
- **Keep all seeds** (wins *and* losses) — losses train the value head.

---

## 4. Staged build plan

Each stage ends with a measured gate. Do not advance until the gate is met.

### Stage 0 — Foundations for learning  ✓ COMPLETE (2026-05-31)
- **0.1 State encoder** (`ml/encoding.py`, `ENCODING_VERSION=1`) ✓ — structured,
  versioned, UNK-safe encoding with full joker/card/shop identity + editions +
  counters + hand levels + deck composition. 13 tests green.
- **0.2 Data pipeline** (`ml/dataset.py`) ✓ — `capture_run` stores a thin but
  replay-complete action log + outcome; `replay_states` re-simulates it to
  reconstruct per-step states; `examples_from_capture` encodes + labels them.
  `verify_capture_roundtrip` gate passes (re-sim reproduces the run exactly and
  survives JSON persistence). 8 tests green.
- **0.3 Training harness** (`ml/model.py` + `ml/train.py`, torch) ✓ — set-encoder
  `ValueNet` over the encoding + `collate_states`; BCE win-prob trainer with eval
  split and versioned checkpoints. Overfit gate passes (tiny synthetic set →
  loss 0.000, 100% acc; checkpoint round-trips). 6 tests green. torch 2.12 (cpu)
  installed (already declared in the `ml` extra).

### Stage 1 — Learned value (the first faster+stronger win)
- **1.1** Bootstrap dataset from `basic_strategy_bot` (the stronger, faster
  teacher) — wins and losses, labeled by outcome.
- **1.2** Train the value head (set-encoder over jokers + card/deck features).
- **1.3** Drop it in as the beam leaf, replacing `clear_probability`. *Gate
  (≥96-seed A/B):* faster per decision **and** ≥ current trajectory quality.

### Stage 2 — Policy head + selective search
- **2.1** Train the policy head (imitate teacher actions; later, search visits).
- **2.2** Replace the fixed beam with policy-pruned best-first / A* / PUCT over
  the single-agent tree; reuse the tree across in-run decisions.
- **2.3** *Gate:* beats Stage 1 winrate at equal or lower compute.

### Stage 3 — Self-improvement loop (the engine of superhuman)
- **3.1** net guides search → record (state, search-visit policy, outcome value)
  → retrain → repeat.
- **3.2** *Gate:* winrate climbs through the 40–50% white-stake mark and keeps
  rising across loop iterations (not just one-shot imitation).

### Stage 4 — Scale & harden
- **4.1** Batched net inference across the worker fleet; encoder/eval hot path
  in Rust if profiling demands. **4.2** Bigger nets, more self-play iterations,
  more seeds. **4.3** Continuous eval harness with fixed seed pools + regression
  gates. *Gate:* stable >50% white, reproducible.

### Stage 5 — High-stakes & superhuman push
- **5.1** Generalize past white stake. **5.2** Stake-specific fine-tuning,
  challenge modes. **5.3** Benchmark vs strong human play / known seed solutions.
  *Gate:* high-stake win rates exceeding strong human expectations.

---

## 5. Risks & mitigations
- **Cross-platform determinism** (for cloud data-gen): integer-exact RNG + IEEE
  floats → likely fine, but gate every new environment with the test suite + a
  same-seed trajectory diff.
- **Encoder omissions** silently cap the model. *Mitigation:* 0.1 gate requires
  field coverage; bump `ENCODING_VERSION` on any change.
- **Net inference cost** could eat the rollout savings. *Mitigation:* keep the
  net small; batch across workers; measure per-decision, not per-run.
- **Self-play instability / reward hacking.** *Mitigation:* outcome-grounded
  value target (won/ante), fixed-seed eval gate every iteration, keep the
  heuristic as a safety baseline.

## 6. Methodology (carried from the project's hard-won lessons)
process_time A/B is ground truth; ≥96 seeds for quality; cProfile lies (use
`scripts/phase_timing.py`); parity-test every native/learned replacement against
the pure-Python path before trusting it.

## 7. Current step & log
- **2026-05-31:** Plan created. Started **Step 0.1 (state encoder)** —
  `src/balatro_ai/ml/encoding.py` + `tests/test_ml_encoding.py`.
- **2026-05-31:** **Stage 0 COMPLETE** — 0.1 encoder (`ml/encoding.py`), 0.2 data
  pipeline (`ml/dataset.py`), 0.3 value net + trainer (`ml/model.py`,
  `ml/train.py`). 27 ml tests green; overfit gate fits to loss 0.000 / 100% acc;
  torch 2.12.0+cpu installed. **Next: Stage 1** (learned value leaf, bootstrap
  from `basic_strategy_bot`).
