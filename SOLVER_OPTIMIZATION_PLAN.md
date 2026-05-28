# Solver Optimization: Fast + Deep Search

**Status:** Tier 1 #2 (dataset CLI) is the only unambiguous win. The
search_v2 play backend is built and tested but defaults to OFF after
a 4-seed comparison showed it loses ~2.5 antes vs the M4 legacy
backend at d3w2 (avg ante 2.0 vs 4.5 — 4× faster but the speed/quality
trade isn't worth shipping by default). v2 stays available via
`SolverPolicy(play_backend="v2")` for deeper-search experiments.
Memoization (Tier 1 #3) is built but backed out of the hot path
(0.2% hit rate at d3w2). Tier 2 is the path to a real speedup —
Cython port would let v2 afford the wider candidate set + larger
leaf samples that close the quality gap.
**Parent:** [`SOLVER_PLAN.md`](SOLVER_PLAN.md) — the milestone-tracking doc
for the solver itself. This document is the rewrite plan that gets the
solver from "9× slower than basic_strategy with parity-only quality" to
"fast enough that depth=5 is affordable and trajectories are actually
expert-quality."
**Last updated:** 2026-05-26.

---

## 1. Why this exists

After M1–M6, the solver works end-to-end but:

- **Throughput:** median 234s/seed serial. With 8-core multiprocessing
  that's ~30s effective. Acceptable for dataset generation but leaves
  zero headroom for deeper search.
- **Quality:** beam at `depth=3, width=2` (matching `search_bot_v2`).
  Single archetype bias regresses by ~1.2 antes; multi-archetype
  portfolio gives upside on ~30% of seeds (score improvements, no ante
  improvements). The solver is essentially a slower basic_strategy_bot.
- **Where the time goes:** per profile,
  - **Play decisions** (~8s each): 90% in `_opening_first_blind_hunt_action`
    + `_basic_best_discard_action` heuristic preprocessing inside
    `best_hand_action`. The beam itself is cheap; we're paying for live-bot
    safety guards we don't need.
  - **Shop decisions** (~1.5s each): 60% in `shop_leaf_terms` →
    `_shop_pressure` → `_sample_hand_build_score`. The build scoring runs
    per leaf state; identity-based caching always misses because beam
    expansions create fresh GameState objects.
- **Where the architecture is wrong:** the solver inherits
  `basic_strategy_bot`-shaped infrastructure that was tuned for live-bot
  real-time constraints. The solver wants the opposite tradeoff: spend
  minutes per seed to get high-quality trajectories. The current code
  carries the live-bot's per-decision cost without the live-bot's
  per-decision speed.

The way to fix this is to **replace** the search infrastructure with a
solver-specific implementation, then make the inner loop fast enough
that depth=5+ is affordable. Not optimize knobs on the existing code.

---

## 2. Architecture overview

```
src/balatro_ai/
  solver/
    seed_game.py            (M1, keep)
    trajectory.py           (M2, keep)
    policy.py               (M5, EVENTUALLY REPLACE — see below)
    play_search.py          (M4, REPLACE with search_v2/play.py)
    archetypes.py           (M6, keep)
    multi_archetype.py      (M6, keep)

    search_v2/              ← NEW package, the optimization work
      __init__.py
      play.py               ← #1: custom solver play search
      leaf_value.py         ← #1: solver-specific leaf evaluator
      state_signature.py    ← #3: content-keyed state hashing
      memo.py               ← #3: typed memoization caches
      shop.py               ← #1.5: custom shop search (M5 followup)

  dataset/
    cli.py                  ← #2: M7 dataset CLI with multiprocessing
    writer.py               ← #2: streaming JSONL writer, resumable

  rules/
    hand_evaluator.py       ← #4: Cython port targets this file
    hand_evaluator_native.pyx  ← #4: Cython module, parity-tested vs Python

  search/
    hand_search.py          (live-bot path, untouched; solver no longer uses)
    shop_search.py          (same)
```

The existing `search/hand_search.py` and `search/shop_search.py` stay as
the live-bot's code path — we don't break `search_bot_v2`. The solver
gets its own `search_v2/` package, which is the parallel rewrite.

`solver/policy.py` now defaults to `search_v2/` for play decisions. The old
`play_search.py` stays available as the `play_backend="legacy"` fallback for
parity testing and regression comparisons.

---

## 3. Tier 1: Foundation

Goal: get per-seed cost from 234s → ~30-60s SERIAL (so 8-core parallel
hits ~4-8s effective). At that point depth=4 or 5 is affordable.

### #1 — Custom solver play search (3-4 days)

**Where:** `solver/search_v2/play.py`.

**API:**

```python
def solver_beam_play_action(
    state: GameState,
    *,
    depth: int = 3,
    width: int = 4,
    leaf_evaluator: LeafEvaluator,
    candidate_provider: CandidateProvider | None = None,
    memo: SearchMemo | None = None,
) -> Action:
    """Whole-blind beam search returning the first action of the best line.
    
    No preprocessing shortcuts. No live-bot safety guards. Pure beam over
    forward_sim states.
    """
```

**Internals:**

- Beam node: `(state, action_sequence, leaf_value)`.
- Per ply: enumerate candidate actions (via `candidate_provider` —
  default `top_k_by_immediate_score(state, k=width*4)`), simulate each via
  `forward_sim.simulate_play` / `simulate_discard`, keep top `width` by
  `leaf_evaluator(child_state)`.
- Terminate at `state.phase != SELECTING_HAND` (cleared or busted) or
  `depth` plies, whichever first.
- Return the action from the highest-value root branch.

**What's gone vs current `best_hand_action`:**

- `_opening_first_blind_hunt_action` — the live-bot's "is this the first
  hand of the run, hunt for a clear" shortcut. Solver doesn't need it.
- `_basic_best_discard_action` — heuristic that runs `discard_score` 29×
  per call evaluating projected hands. The beam does this implicitly.
- `_action_reason` annotation. Solver doesn't need decision tracing.

**Leaf evaluator:** see #1b below.

**Done when:**
- `solver_beam_play_action(state, depth=3, width=2)` returns an action
  matching what `best_hand_action(state, beam_depth=3, beam_width=2)`
  returns on at least 80% of canonical-seed states (parity ≥ 80%).
- Per-call cost on AAAAAAA opening hand: <500ms (currently ~8s).
- Full trajectory through `generate_trajectory` runs to RUN_OVER with
  ante ≥ baseline median 4.

**Progress 2026-05-26:**
- Added `solver/search_v2/play.py` and `SearchV2PlayPolicy`.
- Switched `SolverPolicy` default play backend to `search_v2`; legacy M4
  play search remains selectable with `play_backend="legacy"`.
- Added a fast deterministic default leaf so the inner beam no longer calls
  rollout/headroom scoring on every child.
- AAAAAAA opening decision measured ~0.41s at `depth=3,width=2` in a local
  smoke test.

### #1b — Solver-specific leaf evaluator (concurrent with #1)

**Where:** `solver/search_v2/leaf_value.py`.

**Why we don't reuse `state_value.clear_probability`:** it's a greedy
1-ply rollout that doesn't see archetype-aware patterns. For a solver
that's planning deeper, the leaf evaluator should predict outcome from
the leaf state, not just whether the current blind clears.

**API:**

```python
class LeafEvaluator(Protocol):
    def evaluate(self, state: GameState) -> float: ...

class ClearProbabilityLeaf(LeafEvaluator):
    """Wraps the existing state_value.clear_probability."""

class FutureBlindSurvivalLeaf(LeafEvaluator):
    """Projects expected score-vs-required across the next 2-3 blinds."""

class ArchetypeAwareLeaf(LeafEvaluator):
    """Wraps a base evaluator + adds archetype-fit bonus."""
```

**Done when:**
- Three implementations land with unit tests.
- `FutureBlindSurvivalLeaf` measurably outperforms `ClearProbabilityLeaf`
  on a 5-seed antes-reached comparison.

**Progress 2026-05-26:**
- Landed `FastHeuristicLeaf`, `ClearProbabilityLeaf`,
  `FutureBlindSurvivalLeaf`, and `ArchetypeAwareLeaf`.
- `FastHeuristicLeaf` is the default for v2 play search because the rollout
  leaves are too expensive inside the beam; the rollout leaves remain
  available for comparison runs.
- Added direct unit coverage for the fast, future-survival, and
  archetype-aware leaf paths.

**Effort:** 1 day, in parallel with #1.

### #2 — Multiprocessing dataset CLI (1 day)

**Where:** `dataset/cli.py` + `dataset/writer.py`.

**API:**

```
python -m balatro_ai.dataset.cli \
  --seeds-file .data/canonical_seeds.txt \
  --out .data/trajectories.jsonl \
  --workers 8 \
  --timeout-seconds 300 \
  --policy multi-archetype \
  --depth 3 --width 2
```

**Implementation:**

- `multiprocessing.Pool(workers)` over seed list.
- Each worker calls `solve_seed_multi_archetype` (or
  `generate_trajectory` for baseline-only) with `--timeout-seconds`
  wall-clock limit per seed.
- Writer streams to JSONL, one row per completed seed. Resumable: skip
  seeds already in the output file.
- Schema (one row per seed):
  ```
  {
    "seed": "AAAAAAA",
    "stake": "white",
    "won": false,
    "final_ante": 6,
    "final_score": 25492,
    "final_money": 12,
    "n_steps": 145,
    "wall_seconds": 234.1,
    "terminated_reason": "RUN_OVER",
    "policy": "multi-archetype",
    "best_archetype": "scaling_joker",
    "attempts": [...],          # multi-archetype only
    "steps": [...]              # if --record-steps
  }
  ```

**Done when:**
- 100-seed prototype run completes in <30 min on 8 cores.
- JSONL round-trips through a `dataset/reader.py` helper without
  errors.
- Re-running with the same `--out` skips already-completed seeds.

### #3 — Content-keyed memoization (2 days)

**Where:** `solver/search_v2/state_signature.py` +
`solver/search_v2/memo.py`.

**Why:** Profile shows `_repeatable_build_score`, `_sample_hand_build_score`,
and `evaluate_played_cards` called hundreds-to-thousands of times per
decision with logically identical inputs across beam branches. Current
`_identity_cached_value` / `_decision_cached` use `id(state)` as the key,
which always misses since branches create fresh GameState objects.

**Approach:**

```python
@dataclass(frozen=True, slots=True)
class CardSignature:
    rank: str
    suit: str
    enhancement: str | None
    seal: str | None
    edition: str | None

@dataclass(frozen=True, slots=True)
class JokerSignature:
    key: str
    edition: str | None
    visible_counter: int  # current_mult / current_chips / etc.

@dataclass(frozen=True, slots=True)
class StateSignature:
    """Hashable, content-keyed digest of state fields the cache cares about."""
    ante: int
    money: int
    hands_remaining: int
    discards_remaining: int
    hand: tuple[CardSignature, ...]
    jokers: tuple[JokerSignature, ...]
    blind_name: str
    required_score: int
    current_score: int
    # plus a digest of modifiers that matter for scoring
```

**Memo:** `functools.lru_cache(maxsize=1_000_000)` on the wrapped
functions, keyed on `StateSignature`. Per-process cache (worker-local
in the multiprocessing pool).

**Done when:**
- `evaluate_played_cards_cached(state, action)` returns identical
  results to uncached version across all 5074 audit transitions.
- Per-decision speedup ≥2× on play decisions, ≥1.5× on shop decisions.

**Risk:** the signature must include every field that affects the
cached function's output. Missing a field gives stale results silently.
Mitigation: parity test on all audit transitions before enabling cache
in production paths.

### Tier 1 acceptance

After all three: median per-seed serial cost **<60s** (target ~30s),
trajectory quality at least at baseline parity. 8-core multiprocessing
puts effective per-seed at ~4-8s, enabling 10k-seed datasets in
~2-3 hours and depth=4-5 search experiments.

### Tier 1 #1 + #1b results (landed 2026-05-26)

Single-seed AAAAAAA measurement at `depth=3, width=2`:

| Configuration | Final ante | Final score | Wall time |
|---|---|---|---|
| M4 PlaySearchPolicy (baseline) | 5 | 4,828 | **291s** |
| search_v2 + FastHeuristicLeaf | 1-2 | ~300 | ~6s |
| search_v2 + ClearProbabilityLeaf(samples=2) | 2 | ~400 | 79s |
| **search_v2 + PlanningValueLeaf (new default)** | **4** | **7,392** | **155s** |

Findings:
1. **PlanningValueLeaf is the production default.** The earlier draft
   defaulted to `FastHeuristicLeaf` (rollout-free, ~50x faster per
   leaf eval) for speed, but a single-seed measurement showed it
   loses ~3 antes at d3w2. The rollout-free leaf can't drive setup
   discards because it sees no progress signal from them. The
   `FastHeuristicLeaf` remains exported for deeper-search experiments
   where the algorithm has enough plies to find the build without
   leaf guidance.
2. **Algorithm shape matters.** Initial draft used greedy 1-ply
   pruning (keep top-`width` children by leaf value, recurse).
   That pruned away setup discards before recursion could see
   they pay off. Fixed by switching to "recurse every candidate,
   cap branching at `width+1` per ply" — matches the M4
   `_beam_plan_value` shape. Speedup vs M4 comes from a cheaper
   leaf (no preprocessing overhead) and no `_opening_first_blind_hunt_action`,
   not from greedy beam pruning.
3. **Single-decision opening-hand timing**: d3w2 + PlanningValueLeaf
   measures at ~480ms vs M4's ~8000ms. The 1.9× full-trajectory
   speedup (155s vs 291s) is smaller than the per-decision speedup
   because the trajectory has many shop decisions too (unchanged
   in this phase).

The acceptance bar from §3 #1 ("ante ≥ baseline median 4") is met.
Score is actually +53% over baseline at the same op-point. Per-call
cost is well under the <500ms bar.

Files landed:
- `src/balatro_ai/solver/search_v2/__init__.py`
- `src/balatro_ai/solver/search_v2/play.py` — `solver_beam_play_action`,
  `SearchV2PlayPolicy`, `TopKByImmediateScore`, `CandidateProvider`
- `src/balatro_ai/solver/search_v2/leaf_value.py` — `LeafEvaluator`,
  `FastHeuristicLeaf`, `ClearProbabilityLeaf`, `PlanningValueLeaf`,
  `FutureBlindSurvivalLeaf`, `ArchetypeAwareLeaf`
- `src/balatro_ai/solver/policy.py` — `SolverPolicy` now has
  `play_backend="v2"` (default) / `"legacy"` switch
- `tests/test_solver_search_v2_play.py` (6 tests)
- `tests/test_solver_search_v2_leaf_value.py` (14 tests)
- `tests/test_solver_policy.py` — extended with `test_default_play_backend_uses_search_v2`
  and `test_legacy_play_backend_still_available`

All 920 project tests pass; 59 solver tests pass in 391s.

### Tier 1 #2 results (landed 2026-05-26)

4-seed parallel run on 4 workers via `python -m balatro_ai.dataset.cli`:

| Seed | Final ante | Final score | Wall time |
|---|---|---|---|
| AAAAAAA | 6 | 26,213 | 273.9s |
| BBBBBBB | 6 | 27,914 | 250.9s |
| CCCCCCC | 5 | 21,876 | 144.7s |
| 1234567 | 6 | 17,799 | 309.5s |

Average ante: **5.75** (above M5 baseline median of ~4-5)
Total wall: **310s** for 4 seeds = ~77s/seed effective on 4 cores.

Findings:
1. **Full SolverPolicy (v2 play + M5 shop) quality is at or above M5
   baseline** on the canonical seeds. The earlier single-process
   ante=3 result for AAAAAAA was an outlier (from a pre-fix algorithm).
   The 4-seed batch matches what we'd expect from the play-only
   measurement plus reasonable shop decisions.
2. **Multiprocessing gives the expected ~Nx speedup**: 77s/seed
   effective on 4 cores vs 234s/seed serial = 3.0× speedup. On
   8 cores we'd project ~38s/seed effective. Acceptance bar of
   `<60s` effective per seed is met.
3. **Resume works**: running the CLI a second time on the same
   `--out` skips all 4 completed seeds and exits cleanly.
4. **Schema round-trips cleanly** (`SeedResult.to_json_dict` →
   JSON → `from_json_dict` produces an equivalent object).

Files landed:
- `src/balatro_ai/dataset/__init__.py`
- `src/balatro_ai/dataset/schema.py` — `SeedResult`, `StepRow`,
  `ArchetypeAttemptRow` dataclasses with JSON round-trip
- `src/balatro_ai/dataset/writer.py` — `JsonlSeedWriter` (fsync-per-row,
  append-mode)
- `src/balatro_ai/dataset/reader.py` — `JsonlSeedReader.completed_seeds()`,
  `read_seed_file()` (one-per-line / comma-separated)
- `src/balatro_ai/dataset/worker.py` — `WorkerConfig`, `solve_seed`
  (pickleable; catches all exceptions and returns a SeedResult
  with `error_type` populated)
- `src/balatro_ai/dataset/cli.py` — `python -m balatro_ai.dataset.cli`
  entry point with `--dry-run`, `--resume` via existing-row skip,
  `--policy v2|legacy|multi-archetype`
- `src/balatro_ai/solver/multi_archetype.py` — extended with
  `play_backend` / `play_depth` / `play_width` pass-through args
- `tests/test_dataset.py` (11 tests)

All 920 project tests pass; 11 dataset tests pass in <1s; 59 solver
tests pass after the multi_archetype API extension.

Known limitations (not blockers for Tier 1):
- Per-seed wall-clock timeout is not enforced at the pool level
  (ProcessPoolExecutor can't kill workers mid-task on Windows).
  Bounded instead by `--max-steps` inside each worker. Revisit if
  a stuck-seed case shows up in production runs.

### Tier 1 #3 results (landed 2026-05-26, then partially backed out)

Content-keyed memoization for `clear_probability` and `planning_value`.

**Code landed** (kept available, not on hot path):
- `src/balatro_ai/solver/search_v2/state_signature.py` — content-keyed
  digest including `current_score` / `required_score` (which the
  existing `_search_state_cache_key` excludes — that key is unsafe
  for caching score-dependent functions).
- `src/balatro_ai/solver/search_v2/memo.py` — `cached_clear_probability`,
  `cached_planning_value`, `solver_search_cache_scope()`.
- `tests/test_solver_search_v2_memo.py` (7 tests including parity).

**Production wiring** (`leaf_value.py` imports + `play.py` scope wrap):
backed out. Initial measurements showed a 1.77× full-trajectory speedup
but follow-up profiling revealed two issues:

1. **Cache hit rate at d3w2 is ~0.2%** — most beam-expansion leaves are
   content-unique within one decision (every rollout step changes
   hand+deck_size). The memo costs more in lookup overhead than it
   saves in avoided rollouts at this op-point. Pays off at d4+, but
   the production op-point is d3w2.
2. **Cache-induced cross-decision drift.** The cache scope is per
   decision, but accidentally sharing partial state across decisions
   (via the way `_state_identity_cached_value` interacts with
   re-derived legal_actions) was producing different actions than
   the uncached path on the same input. Reverting fixed it.

The memoization infrastructure is correct and tested; it just isn't
the right tool for d3w2-shaped search. Tier 2 (IDA*/Cython) is the
structural speedup needed at this op-point.

### Tier 1 #1 candidate-ranking fix (2026-05-26)

While debugging the Tier 1 #3 backout I discovered the dominant
quality issue wasn't memoization — it was `TopKByImmediateScore`.

The original implementation returned `(plays, discards)` concatenated,
which meant plays (numerical scores 100-1000s from the heuristic)
always ranked above discards (heuristic scores 5-30), and discards
got pruned out at every internal ply. The beam never learned that
"discard now, play later" was a useful plan because it couldn't see
past one ply of discards.

Two-stage fix:
1. **Unified ranking** — combined plays+discards on a single rank
   scale matching M4's `_cheap_beam_action_rank`, which adds a
   +15000 "pressure bonus" to discards when no available play can
   clear at current pace.
2. **Min-per-category guarantee** — at the root, ensure at least
   `min_plays` plays AND `min_discards` discards survive the
   per-ply cap. Without this, the pressure bonus pushes ALL
   discards above ALL plays on the opening hand of low-value seeds,
   and the beam misses winning play paths entirely.

### Honest assessment of Tier 1 speed gains

| Metric | Plan projection | Actual | Notes |
|---|---|---|---|
| Per-call opening-hand timing | <500ms (16× vs M4) | ~2000ms (4× vs M4) | FastHeuristicLeaf hits the bar but loses 3 antes; production uses PlanningValueLeaf |
| Memoization speedup | ≥2× per decision | ~0% at d3w2 | Backed out of hot path; kept available for d4+ experiments |
| Full-trajectory wall (single-seed AAAAAAA) | <60s, target ~30s | ~150-250s | Variance from pre-existing shop RNG non-determinism |
| Parallel effective per seed (4 cores) | ~15s extrapolated | ~50s effective | Below target but ~6× faster than M4 serial |
| 4-seed average ante | (implied parity) | ~3.5-4.75 across runs | At or above M4 baseline on average |

**Where the speed gains actually came from:**
- Removing `_opening_first_blind_hunt_action` preprocessing: ~1.5-2× per call.
- Multiprocessing the dataset CLI: linear in worker count.
- Cheaper leaf evaluator default (samples=2 vs M4's planning_value samples=16): ~3-4× per leaf.

**Where they DIDN'T come from:**
- Memoization (kept off the hot path; ~0% hit rate at d3w2).
- Algorithm shape changes (beam stays minimax-with-per-ply-cap; same shape as M4).

### Known cross-run variance — investigated, mostly false alarm

Initial measurements showed wild ante swings (1 vs 6) between
"the same" runs of v2. I suspected `shop_sampler.py`'s ~10
unseeded `Random()` fallbacks, but a controlled experiment
showed otherwise:

- **3 fresh subprocesses, no code edits between**: identical results
  (ante=1, score=524 across all 3).
- **3 sequential runs in one process**: small variance (score
  ±10%) — accumulating state in module-level caches.
- **2 CLI batches back-to-back, no edits**: byte-identical
  output.

So the trajectory pipeline IS deterministic across subprocesses
when no code changes. The "variance" I was chasing was my own
edit churn between measurements — each edit shifted the picked
action for borderline-tied leaves. The `shop_sampler.py` unseeded
fallbacks never actually fire in production because every caller
(local_runner + shop_search) threads its own seeded RNG.

Lesson: when measuring optimization impact, ALWAYS take baseline
+ test measurements without code edits in between. Earlier
results in this doc that showed "v2 ante 4 / 87s" on AAAAAAA
were valid for that exact code state, but several edits
followed before the 4-seed batch was run; the batch reveals
the v2 quality is lower than that early single-seed result
suggested.

### Clean 4-seed measurement (final, 2026-05-26)

After reverting the abortive ranking experiments back to the
naive `(plays, discards)` concat:

| Seed | M4 legacy ante | M4 wall | v2 ante | v2 wall |
|---|---|---|---|---|
| 1234567 | 6 | 514.8s | 1 | 40.0s |
| AAAAAAA | 5 | 293.6s | 4 | 128.9s |
| BBBBBBB | 4 | 313.0s | 2 | 65.8s |
| CCCCCCC | 3 | 119.5s | 1 | 11.3s |
| **avg ante** | **4.50** | — | **2.00** | — |
| **batch wall (4 workers)** | — | **515.2s** | — | **129.8s** |

v2 is 4× faster but 2.5 antes worse on average. SolverPolicy
now defaults to `play_backend="legacy"` so the production
solver ships M4 quality. v2 stays opt-in for deep-search
experiments where the leaf evaluator can be FastHeuristicLeaf
(saving most of the rollout cost).

### What Tier 2 needs to fix

For v2 to become competitive with M4 at production samples:
- Wider root candidate enumeration (M4: 24 candidates, v2: 6).
- Larger leaf samples (M4: 16, v2: 4).
- Probably reintroduce something like the
  `_opening_first_blind_hunt_action` preprocessing.

All three need cheaper per-call cost than today, which is what
Cython delivers. **Tier 2 is the actual path to v2 beating M4
on quality at a fraction of the wall time.** Without it, v2's
"speed advantage" comes from cutting corners that matter.

---

## 4. Tier 2: Deepen the search

Goal: with Tier 1 making per-decision cheap, push search depth to
`depth=4-6` and measure quality gains. If the per-decision cost is
still the bottleneck for deeper search, swap the scoring inner loop
to a compiled implementation.

### #4 — Cython port of `hand_evaluator` hot path (3-5 days)

**Where:** `rules/hand_evaluator_native.pyx` + `setup.py` cython
extension build.

**Scope:** port only the inner loop functions, leave high-level
orchestration in Python.

Inner functions (~800 lines of the 2273-line file):
- `evaluate_played_cards` — entry point
- `_effect_adjustments` — joker effect aggregation
- `_scoring_indices` — which played cards score
- `_identify_hand_type` — poker hand classification
- `_card_chip_value` — per-card chip calculation

Python wrapper keeps the same API. Calls native impl when available,
falls back to Python when not.

**Build:**
- `pyproject.toml` adds cython to build-system requires.
- `setup.py` adds an `Extension(...)` for the native module.
- `pip install -e .` triggers compilation.

**Parity check:**
- New `tests/test_hand_evaluator_native_parity.py` runs both Python and
  native on every audit transition. Must be exact match on score,
  money_delta, hand_type, scoring_indices.

**Expected speedup:** 10-50× on `evaluate_played_cards`. With Tier 1's
memoization, the cumulative effect compounds: cached calls go from
~50μs to ~5μs.

**Risk:** Cython compilation step adds a build dependency. Acceptable
trade. The bigger risk is parity — port has to be exact.

### #5 — Single-player search reformulation (2 days)

**Where:** `solver/search_v2/play.py` — algorithmic upgrade to existing
beam.

**Insight:** With seed-faithful RNG, Balatro is a SINGLE-PLAYER planning
problem, not a game tree. No opponent moves means:
- No expectimax needed (every "chance node" is actually deterministic
  given the seed)
- A*/IDA* with an admissible heuristic prunes vastly more aggressively
  than beam search
- Branch-and-bound with `clear_probability` as the upper bound becomes
  the natural algorithm

**Approach:**

```python
def ida_star_play(state, leaf_evaluator, max_depth, time_budget):
    """Iterative deepening with branch-and-bound pruning.
    
    Each iteration deepens by 1. Prunes branches whose upper-bound
    leaf value can't exceed the current best.
    """
```

**Expected impact:** with a tight heuristic, may achieve depth=8+ in the
same time current beam takes for depth=3. Quality gain is meaningful
because depth=8 covers the whole blind.

**Risk:** harder to get right than beam. Beam stays as a known-good
baseline; IDA* runs alongside for A/B during development.

### Tier 2 acceptance

Median per-seed serial cost **<15s** at `depth=4-5`. Single-archetype
trajectory at depth=5 measurably beats baseline (>0.5 ante average
improvement, or >2× score at same ante).

---

## 5. Tier 3: Algorithmic rewrite (optional, last resort)

### #6 — MCTS with tree reuse (2 weeks)

**Where:** `solver/search_v2/mcts.py`.

**Why:** beam discards work between decisions. MCTS builds one tree per
run, reuses subtrees across sequential decisions. For "spend minutes per
seed" workloads, MCTS is significantly more compute-efficient than
restart-from-scratch beam.

**Implementation pattern:** standard four-step MCTS (selection / expansion
/ simulation / backpropagation) with UCT for selection. Tree node stores
visit counts + average rollout value. Decisions pick highest-visit child
(robust child rule).

**Required infrastructure:**
- Tree node storage (dict keyed on action sequence)
- Rollout policy (basic_strategy_bot for fast playouts)
- UCT exploration constant tuned per archetype

**Expected impact:** depth equivalent to ~100-1000 simulations per
decision, with reuse across decisions roughly halving the per-decision
cost.

**When to do this:** only if Tier 1 + Tier 2 don't get depth high enough
for the quality target.

### #7 — Rust port of `forward_sim` + `hand_evaluator` (3-4 weeks)

The big rewrite. Maximum speed, maximum effort. Only justified if we
need to regenerate the dataset many times and the wall-clock saving
across regenerations exceeds the porting cost. Defer until that
decision is forced.

**STATUS (2026-05-27):** The Rust port has been started. Phases 1
(state types), 2 (hand evaluation + ~80 jokers), 3 (forward-sim
helpers + `simulate_play_simple`), and 4a (batched action scorer)
are landed. See **[`RUST_PORT_PLAN.md`](RUST_PORT_PLAN.md)** for the
active roadmap and status. Phase 2 evaluate alone gives a 75× per-call
speedup; Phase 4a delivers ~5% trajectory speedup; the full Phase 4
(native solver beam) targets the ≥3× trajectory goal.

---

## 6. Build sequence

```
Week 1: Tier 1
  Day 1-2:  #1 custom play search (search_v2/play.py)
            + #1b leaf evaluators (search_v2/leaf_value.py)
            + parity tests against existing best_hand_action
  Day 3:    #2 multiprocessing dataset CLI (dataset/cli.py)
  Day 4-5:  #3 content-keyed memoization (search_v2/state_signature.py)
            + parity test on all audit transitions
            CHECKPOINT: rerun 5-seed measurement; expect 234s → ~60s
                        serial, ~8s effective on 8 cores

Week 2: Tier 2
  Day 6-8:  #4 Cython port of hand_evaluator inner loop
            + parity tests on all 5074 audit transitions
  Day 9-10: #5 IDA* / branch-and-bound play algorithm
            + A/B vs beam at same time budget
            CHECKPOINT: 1000-seed dataset run at depth=4-5,
                        compare antes vs baseline solver

Week 3+: Tier 3 (only if Tier 2 measurement says it's needed)
  Day 11+:  #6 MCTS or #7 Rust port
```

After Week 2: `10k seeds at depth=5 in ~30 min on 8 cores` is the
target end-state. That's the "fast + deep" picture.

---

## 7. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Custom search has bugs that produce worse trajectories than wrapped version | Med | Parity test vs `best_hand_action` on canonical states before swapping in; keep old play_search.py available as fallback |
| Content signature misses a field, cache returns stale results | High | Run full audit-transition parity test BEFORE turning cache on in production |
| Cython parity drift after a game update | Med | Audit parity test in CI; native module is built locally per dev, not distributed as binary |
| IDA* heuristic is non-admissible, finds suboptimal lines | Med | Keep beam alongside as the "safe" search; A/B every change |
| MCTS tuning rabbit hole (exploration constant, rollout policy) | High | Cap MCTS investment at 2 weeks; revert to beam if quality gains haven't shown by then |
| Cython build adds friction for new contributors | Low | Provide pure-Python fallback path that doesn't require the compile step |

---

## 8. What we delete and what we keep

### Keep (no change)

- `solver/seed_game.py` — perfect-information layer is solid
- `solver/trajectory.py` — trajectory recorder is solid
- `solver/archetypes.py` — archetype mechanism is the right shape, just
  needs better integration with new search
- `solver/multi_archetype.py` — portfolio selection logic is right
- `search/hand_search.py`, `search/shop_search.py` — live-bot's code
  path. Untouched. `search_bot_v2` keeps working.
- `rules/hand_evaluator.py` — Python implementation stays as
  fallback + parity check target

### Replace

- `solver/play_search.py` — `PlaySearchPolicy` replaced by
  `solver_beam_play_action` (or `ida_star_play` after Tier 2)
- `solver/policy.py` — `SolverPolicy` rewrites to use `search_v2/`
  primitives. Public API stays the same.

### Add

- `solver/search_v2/` — entire package, the new search infrastructure
- `dataset/` — entire package, the dataset CLI + readers
- `rules/hand_evaluator_native.pyx` — Cython implementation (Tier 2)
- `pyproject.toml` / `setup.py` — Cython build hooks (Tier 2)

---

## 9. Open questions before starting

1. **Build system for Cython.** Project uses `python -m unittest`
   directly, no `pip install -e .` workflow yet. Need to decide:
   skip Cython entirely (lose Tier 2 speedup), or land a minimal
   `pyproject.toml`/`setup.py` to support the extension. Recommend
   the latter — it's overdue anyway.

2. **Memoization scope per worker.** With multiprocessing, each worker
   has its own cache. Per-worker memo is simpler but loses cross-seed
   sharing. Per-seed memo is even more isolated. For Tier 1 #3, start
   per-worker; revisit if needed.

3. **Should `archetypes.py` get a redesign too?** Current additive
   bias was shown to regress on most seeds. If we're rewriting, this
   is the right time to redesign — maybe replace `shop_leaf_terms`
   entirely for archetype runs instead of adding a bonus on top.
   Defer to a separate plan; this document is about search infrastructure
   speed, not archetype mechanism.

4. **Multi-stake support.** Currently white-stake only. The new
   solver should be stake-agnostic from the start, but no measurement
   work for non-white stakes is planned. Keep the API parameterized,
   no immediate implementation.

5. **How aggressive should the timeout be?** A 5-min per-seed timeout
   is safe for current ~234s median + tail. With Tier 1 expecting 60s
   median, a 90-second timeout might be tight enough to keep parallel
   throughput predictable. Calibrate after Tier 1 lands.

---

## 10. Definition of done

The optimization project is complete when:

- **Throughput:** 10k seeds, multi-archetype, depth=5 → <2 hours wall
  on 8 cores. (Currently: same config would be ~50 hours.)
- **Quality:** at depth=5, ≥30% of seeds in a 100-seed test improve
  on the depth=2 baseline by either +1 ante or 2× score.
- **Reproducibility:** the dataset CLI run twice on the same seed file
  produces byte-identical JSONL output (modulo wall-time floats).
- **Maintainability:** all parity tests pass; new contributor can run
  `python -m unittest discover -s tests` and see green; Cython build
  failure falls back to Python without breaking anything else.
- **Documentation:** this plan stays current — each completed item
  gets a status note inline; lessons go in a "Lessons learned" section
  appended at the bottom.

If we hit those four, we have a solver that can plausibly produce the
Phase 8 training dataset at a quality level worth training on, in a
turnaround time that allows iteration.
