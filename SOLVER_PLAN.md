# Offline Solver: Implementation Plan

**Status:** M1–M5.5 done; `SolverPolicy` is built and generating data. The
active work is no longer M6 archetype branching — it pivoted to **raising the
data-gen winrate by fixing systematic value-function bugs** (see `PROGRESS.md`
2026-05-30/31 and memory `project_datagen_speed.md`). M6 archetype branching was
measured low-value (only ~35% of bought jokers fall in any archetype; Full House,
the #1 leveled hand, is in none) and is deprioritized; the wins came instead from
a play-value bug fix (ec9d0b7) and a shop joker-churn fix (f2944d8, ~0%→8%), plus
a first-shop Buffoon-pack fix.
**Parent:** [`PHASE7_OFFLINE_SOLVER_PLAN.md`](PHASE7_OFFLINE_SOLVER_PLAN.md) section 5.
**Last updated:** 2026-05-31.

This document tracks the offline solver build itself — milestone status,
decisions made, things that surprised us, throughput numbers. Updated as
the work progresses.

---

## 1. Goal

Given a run seed, produce a `Trajectory(actions, outcome, ante_reached,
score)` strong enough to be Phase 8 imitation data. Target dataset: 10k–50k
trajectories on a single workstation in days, not weeks.

The solver is a TEACHER. It does not need to play in real time. It can take
minutes per seed. The bot the network produces later is a separate concern.

## 2. Architecture overview

```
solve_seed(seed, time_budget) -> Trajectory:

    # 2.1. Materialize what we can predict from the seed alone.
    game = SeedMaterialization(seed)
        - deck shuffle order
        - boss blind per ante (8)
        - voucher per ante (8)
        - tag pair per ante (Small + Big tag)
        - shop contents per ante (predicted at decision time, not all eager)
        - pack contents (predicted lazily on open)
        - per-card RNG outcomes (lazily on use)

    # 2.2. Branch on build archetypes at the root.
    candidates = []
    for archetype in ARCHETYPES:
        trajectory = search_within_archetype(
            game, archetype, budget=time_budget / len(ARCHETYPES))
        candidates.append(trajectory)

    # 2.3. Return the best trajectory.
    return best_by_depth_then_score(candidates)
```

Search within an archetype:

- **Hand-play decisions:** whole-blind beam search (width 8–16, depth = full blind).
- **Shop decisions:** beam search over the buy/sell/reroll/end_shop tree, with archetype-aware leaf scoring.
- **Pack opening:** enumerate predicted pack contents, pick best subset for archetype.
- **Tag / voucher / blind-skip decisions:** archetype-aware heuristic (no deep tree).

## 3. Milestones

Each milestone updated inline with status, file paths, and lessons. Strike
through items when superseded by later work.

### M1 — Seed materialization (the perfect-information layer) ✅ DONE

**Status:** done 2026-05-25.

Built `src/balatro_ai/solver/seed_game.py`:

- `SeedGame(seed: str, stake: str = "white")` with memoized
  `initial_surface()`, `deck()`, `boss_for_ante(ante)`, `initial_state()`.
- `initial_state()` reuses `_with_blind_selection_surface` from
  `local_runner` so the returned state has the correct `current_blind`,
  `blinds` dict, blind name, and `required_score` populated — matches what
  the bridge would emit immediately after `start_run(seed=...)`.
- `deck_for_seed(seed)` exposed as a standalone helper.

**Acceptance:** 11 tests pass in `tests/test_solver_seed_game.py`:
- All 4 canonical seeds match the captured deck fixtures (full 52-card).
- All 4 canonical seeds match the ante-1 boss in captured shop fixtures.
- Initial state has correct ante/money/hands/discards/deck_size.
- Initial state has populated `current_blind` + `required_score=300`.
- `simulate_select_blind(initial_state, drawn_cards=deck[:8])` advances
  to SELECTING_HAND with a full 8-card hand.
- `simulate_play` advances state, decrements `hands_remaining`, scores
  > 0. End-to-end pipeline works without bridge.

**Lessons:**
- `simulate_select_blind` takes `drawn_cards` as a positional second arg,
  not an `Action`. The select-blind action itself has no card_indices;
  the API model only exposes "what's drawn at start of blind" as
  injected state.
- Reusing `_with_blind_selection_surface` from `local_runner` was the
  right move — reimplementing the bridge-style modifier payload would
  have been hundreds of lines.
- Boss for ante 2+ is NOT trivial — it needs the running boss-use-counts
  dict because the source filters used bosses. Punted to M5/M6 work.

**Estimated effort vs actual:** half day estimated, ~30 min actual
(the existing RNG layer did most of the work).

### M2 — Minimal trajectory generator (the throughput baseline) ✅ DONE

**Status:** done 2026-05-25.

Built `src/balatro_ai/solver/trajectory.py`:

- `Trajectory(seed, stake, steps, final_phase, final_ante, final_score,
  final_money, won, sim_step_count, wall_seconds, terminated_reason)`
  dataclass.
- `StepRecord` per-step summary (phase_before, action_type, money_before/after,
  score, ante, hands, discards) — enough for Phase 8 imitation
  targets without bloating the dataset with full state snapshots.
- `generate_trajectory(seed, policy, *, stake, max_steps, record_steps)`
  drives any `GameState -> Action` callable through `SeedGame.initial_state()`
  + `LocalBalatroSimulator.step()`. Records timing, classifies terminations
  as `RUN_OVER` / `STEP_LIMIT` / `STUCK` / `POLICY_ERROR` / `POLICY_NOOP`.

**Acceptance:** 4 tests pass in `tests/test_solver_trajectory.py`:
- AAAAAAA + basic_strategy_bot → RUN_OVER with >50 steps.
- Step records match step count.
- `record_steps=False` produces an empty steps tuple but still tracks counts.
- Consecutive discards (common bot behavior) don't trip the stuck detector.

**Implementation notes:**
- Reuses `LocalBalatroSimulator.step()` for action dispatch. The simulator
  uses its own Python `Random` for shop/pack sampling — this is NOT
  seed-faithful, but for M2/M3's "measure compute cost" goal it's fine.
  M4+ swaps in seed-faithful shop predictions.
- Initial state from `SeedGame.initial_state()` (M1) — the only piece M2
  contributes to seed-faithfulness is the deck order.
- Stuck detection uses a 10-field state signature
  `(phase, ante, score, money, hands, discards, deck_size, hand_size,
   joker_count, consumable_count)` with a threshold of 10 repeats. First
  pass used a 5-field signature with threshold 3 and falsely flagged
  legitimate consecutive-discard sequences as stuck.

### M3 — Throughput baseline (10-seed prototype) ✅ DONE

**Status:** done 2026-05-25.

Ran `generate_trajectory` on 10 mixed seeds (4 canonical + 6 random) with
`basic_strategy_bot` as the policy.

| Metric | Value |
|---|---|
| Wall median per seed | **25.67s** |
| Wall mean per seed | 26.27s |
| Wall max per seed | 38.45s |
| Antes reached | range 3–8, median ~6 |
| Wins | 1/10 (XKCD0001 reached ante 8) |
| Terminations | 10/10 `RUN_OVER` (no STUCK, no errors) |

**Projected dataset costs (8-core parallelism):**

- 10k seeds: **~8 hours**
- 50k seeds: **~2 days**

**Decision per M3 acceptance criteria:**

- Median <30s/seed ✅ → continue to M4 without optimization detour
- Max <60s/seed ✅ → no per-seed timeout needed yet (but M7 will add one)
- The actual solver (M4+ with search) will be slower per decision. Budget
  has headroom — if M4 + M5 + M6 individually 2–3x the per-decision cost,
  we're still within "1 day for 10k seeds" range. If they 10x it, we'll
  need to optimize.

**Caveat:** these numbers are basic_strategy_bot, which is heuristic-only
(no search). Solver decisions cost more compute but produce better
trajectories. M4 measurement will tell us how much the search overhead
actually is. The 25.67s baseline establishes the "ceiling of acceptable
slowdown" — if M4 takes 200s/seed, we have a problem.

### M4 — Whole-blind beam search ✅ DONE

**Status:** done 2026-05-25.

Wrapped `best_hand_action` / `best_blind_beam_action` from the existing
`search/hand_search.py` as a solver policy rather than reimplementing.
Lives in `src/balatro_ai/solver/play_search.py`.

`PlaySearchPolicy` interface:
- `choose_action(state)` — at SELECTING_HAND with play/discard available,
  runs `best_hand_action` with a beam config; otherwise delegates to a
  fallback callable (default: `basic_strategy_bot`).
- `__call__` aliased to `choose_action` so the policy is directly usable
  as the `policy` parameter to `generate_trajectory`.
- Defaults: `beam_depth=3, beam_width=2` — matches search_bot_v2's
  known-working values. Higher (8/4) blew up per-decision cost ~10× in
  smoke testing without proportional gain; archetype branching in M6 is
  the right place to spend more compute.

**Acceptance:** 3 tests pass in `tests/test_solver_play_search.py`:
- Falls through to fallback for BLIND_SELECT phase.
- `__call__` alias matches `choose_action`.
- Full trajectory through `generate_trajectory` finishes (RUN_OVER or
  STEP_LIMIT) without crashing.

**Throughput (M3-style measurement, beam_depth=3 / beam_width=2):**
- Single seed AAAAAAA: **235 seconds** (vs basic_strategy_bot's ~25s).
- That's **9.4× slowdown**, on the upper end of the plan's "2–3× okay,
  10× would mean optimization needed" tolerance.

**Profile of one search call (1.5s for state with 8-card hand,
436 legal actions):**
- ~1.34s in `_opening_first_blind_hunt_action` →
  `_basic_best_discard_action` → `discard_score` →
  `_projected_score_after_discard` → `_best_play_from_hand`. This is the
  search bot's *opening-hand heuristic preprocessing*, NOT the beam.
- ~0.18s in `_best_play_action` (also preprocessing).
- The beam itself (`_beam_action_value` and descendants) is a small
  fraction.

So most of M4's slowdown comes from the heuristic preprocessing the
existing search infrastructure runs BEFORE the beam — the beam itself
is cheap. Optimization options for later:
- Bypass `_opening_first_blind_hunt_action` for solver invocations
  (we don't necessarily need its "guard rail" behavior).
- Cache the opening-hand heuristic per (hand, joker-set) signature.
- Use a thinner action enumeration during preprocessing.

**Dataset cost projection (8-core parallel at 235s/seed):**
- 10k seeds: ~3.5 days
- 50k seeds: ~17 days (out of budget)

**Decision:** continue to M5 with current numbers. The optimization
options above are all viable if M5+M6 push us further out. The
single biggest unknown is whether shop search adds another 2–3× or
another 10×; M5 measurement will tell us.

**Lessons:**
- `best_hand_action` runs `_opening_first_blind_hunt_action` first,
  which can return a discard directly and skip the beam entirely.
  Looking at one decision and assuming it's "the beam" is misleading.
- `best_blind_beam_action` is the actual beam-search entry point. If
  M5/M6 needs to call beam search explicitly without the preprocessing,
  invoke that one instead.

### M5 — Shop search ✅ DONE (but throughput problem surfaced)

**Status:** done 2026-05-25; flagged optimization milestone before M6.

Built `src/balatro_ai/solver/policy.py` with composed `SolverPolicy`:
- Dispatches by phase: `SELECTING_HAND` → `PlaySearchPolicy` (M4),
  `SHOP` → `best_shop_action` from existing `search/shop_search.py`,
  everything else → fallback (default `BasicStrategyBot`).
- `__call__` aliased to `choose_action` so it drops into
  `generate_trajectory` directly.
- Shop config defaults: `beam_width=8, depth=3, reroll_samples=32`
  (matches existing `ShopSearchConfig` defaults).

**Acceptance:** 3 tests pass in `tests/test_solver_policy.py`:
- BLIND_SELECT delegates to fallback.
- `__call__` matches `choose_action`.
- Full trajectory through `generate_trajectory` finishes without crashing.

**Throughput (single seed AAAAAAA, default config):**
- **768 seconds per seed** (12.8 minutes).
- That's 3.3× M4 (235s) and ~30× M3 baseline (25s).

**Dataset cost projection (8-core parallel at 768s/seed):**
- 10k seeds: **~11 days**
- 50k seeds: **~57 days** (effectively unfeasible)

**Decision: optimize before M6.** M5 alone consumes ~7× the M4 budget
that was already 9.4× the baseline. Continuing to M6 (archetype
branching) on top of this would compound the problem, potentially
pushing 10k seeds past a month. The structural search machinery is
right — the cost is in how often expensive paths run, not in the
algorithm.

**Profile target for the next milestone (M5.5):**
- Re-run the per-call profile from M4 (`_opening_first_blind_hunt_action`
  was 90% of cost there). With shop search now in the mix, the second
  per-call profile point is `best_shop_action`'s leaf-value
  computation, particularly `shop_leaf_terms` and `reroll_samples=32`
  sampling.
- Concrete optimization candidates ranked by expected payoff:
  1. **Bypass `_opening_first_blind_hunt_action` in solver path** —
     the preprocessing heuristic adds little value when the beam is
     already doing whole-blind planning. Add a HandSearchConfig flag
     or branch on it in PlaySearchPolicy.
  2. **Drop `reroll_samples` from 32 to 4–8** — sampling 32 shop
     rolls per reroll evaluation is excessive for a wide-but-not-deep
     beam.
  3. **Cache shop leaf values per (state-signature, archetype)** —
     similar shop states recur often within a single run.
  4. **`functools.lru_cache` on `evaluate_played_cards`** if profile
     confirms it's still the bottleneck after (1) and (2).

### M5.5 — Throughput optimization ✅ DONE

**Status:** done 2026-05-25.

**Result:** median 234s/seed across 5 mixed seeds (max 714s,
mean 355s). Down from M5's 768s single-seed measurement.

| Metric | M5 | M5.5 | Δ |
|---|---|---|---|
| Median /seed | 768s (n=1) | 234s (n=5) | ~3× faster |
| Antes reached | — | [2, 3, 4, 6, 7] | comparable to basic_strategy's [3, 4, 5, 5, 6] |
| 10k @ 8 cores | 11 days | **~3.4 days** | within target |
| 50k @ 8 cores | 57 days | ~17 days | borderline, acceptable |

The 234s exceeds the <200s target slightly but the dataset projection
(3.4 days at 8 cores for 10k) is functional. Diminishing returns on
further tuning — push to M6 instead, where the actual quality gains
land.

Goal: get per-seed cost from 768s to under 200s. That's:
- 10k seeds: <3 days at 8 cores (acceptable)
- 50k seeds: <2 weeks at 8 cores (acceptable for the larger run)

Approach: profile-driven, one optimization at a time, re-measure after
each. Don't compound speculative changes.

**Optimizations attempted:**

1. **Bypass `_opening_first_blind_hunt_action` in PlaySearchPolicy** —
   ❌ rejected. Single PLAY decision went from ~1.5s to ~8s. The
   preprocessing wasn't pure overhead — it's a shortcut path that
   returns a clear-winning discard without invoking the beam. Removing
   it forces the beam to run on every state. Kept the flag
   (`skip_preprocessing=False` default) for parity-testing only.

2. **Shop config tightening** — applied. Three knobs together:
   - `beam_width` 8 → 4 (halve leaves per ply)
   - `depth` 3 → 2 (cover BUY+END or BUY+BUY+END common case)
   - `reroll_samples` 32 → 8 (variance reduction, 8 still adequate)
   Per-shop-call profile expected to drop 3-5×.

3. **State-content caching** — investigated. `_repeatable_build_score`
   and `_shop_pressure` already use `_decision_cached` /
   `_identity_cached_value`, but those key on object identity, not
   content. Each beam expansion creates fresh GameState objects, so
   identity-based caching always misses. Content-based caching would
   need a custom hash on GameState, which is invasive (touches
   tens of cache call sites). **Deferred** unless shop config
   tightening + later play optimizations don't get us under budget.

4. **Drop play search budget** — pending. PlaySearchPolicy currently
   `beam_depth=3, beam_width=2`. Try `beam_depth=2, beam_width=2` if
   shop optimization alone isn't enough. PLAY calls dominate
   trajectory cost (see Lessons), so this is the biggest remaining
   lever before invasive caching work.

5. **Cache `evaluate_played_cards`** — pending, only if profile still
   shows it as the bottleneck after the above. Same identity-vs-content
   caveat applies.

Acceptance: full trajectory under 200s/seed on AAAAAAA, and the
trajectory completes RUN_OVER (no STUCK) with a final ante consistent
with M3/M4 behavior (i.e. we didn't lobotomize the policy).

### M6 — Archetype root branching

**Status:** **DEPRIORITIZED (2026-05-31).** M6a partial, M6b scaffolded, but a
build audit (`scripts/solver_build_audit.py`) showed the archetype model is a
poor fit for what the solver actually builds — only ~35% of bought jokers are in
any `BUILT_IN_ARCHETYPE` key list, and Full House (the #1 leveled hand) is in
none — so archetype-coherence terms barely engage. The winrate work moved to
fixing systematic value-function bugs instead (play-value ec9d0b7, shop-churn
f2944d8 ~0%→8%, first-shop Buffoon pack). The archetype machinery remains in the
codebase, default-off. See `PROGRESS.md` and memory `project_datagen_speed.md`.
Original M6 notes below kept for context.

**M6a (single archetype, soft-bias shop leaf scoring) — built but regressed quality.**

Implemented `Archetype` dataclass and `SolverPolicy(archetype=...)` —
the archetype's `archetype_fit_score(leaf_state)` adds an additive bonus
(per matched joker/consumable in the leaf state) to the shop leaf value.

5-seed measurement vs M5.5 baseline (sorted antes):
| Run | Antes | Avg |
|---|---|---|
| Baseline (M5.5, no archetype) | [2, 3, 4, 6, 7] | 4.4 |
| Flush archetype | [1, 2, 3, 4, 4] | 2.8 |
| Scaling Joker archetype | [2, 4, 4, 4, 4] | 3.6 |

Both archetypes underperformed the baseline. The additive bias as
designed appears to override useful baseline decisions when the
archetype's preferred items aren't present in the seed's shops (which
is most of the time at ante 1). 5 seeds is high variance, but the
trend is consistent enough to call M6a's mechanism a partial failure.

**M6b (multi-archetype branching) — scaffolded.**

Built `src/balatro_ai/solver/multi_archetype.py` with
`solve_seed_multi_archetype(seed)`:
- Always runs the no-archetype baseline as one of the candidates.
- Runs each `BUILT_IN_ARCHETYPES` entry as a separate candidate.
- Returns `MultiArchetypeSolve(seed, attempts, best)` where `best` is
  chosen by `(won, final_ante, final_score)`.

This guarantees solver quality ≥ baseline since baseline is always a
candidate. The upside is when an archetype's items happen to fit a
seed, it can beat baseline; the downside is just compute cost.

Tests: 4 in `tests/test_solver_multi_archetype.py` covering selection
ordering (won > ante > score) and API shape. No full multi-seed
measurement yet — pending the gentle-bonus tuning result.

**Open M6 questions:**
- **Bonus magnitude.** 2.0/match was the M6a default; 0.5/match
  measurement is in flight. If 0.5 brings antes back to baseline
  parity (and a subset of seeds exceed it), the additive mechanism
  is salvageable, just over-tuned.
- **Multi-archetype throughput.** 5× per seed × 234s = ~20 min/seed
  serial, ~3 min/seed at 8 cores. 10k seeds = ~3 weeks. M7 will need
  adaptive budget (skip dead archetypes early, terminate when one wins).
- **Archetype seed condition.** Could require the archetype's first
  match before applying bonus (preventing over-commit at ante 1).
  Defer until we have multi-archetype data.

**Critical bug found during M6a measurement (2026-05-25):**

`generate_trajectory` derived `LocalBalatroSimulator`'s integer seed via
`abs(hash(seed)) & 0x7FFFFFFF`. Python's built-in `hash()` is randomized
per process (PYTHONHASHSEED), so the same seed string produced
DIFFERENT simulator seeds across Python invocations. Each archetype
measurement ran in a separate background process, so the per-archetype
trajectories ran against different shop samples — not against each other.

Within-process variance check: same policy, same seed string, two runs
in one process → identical antes. Cross-process: same policy, same seed
string, two separate `python -c "..."` invocations → antes can differ
by ±5 (e.g., XKCD0001 went 7→1 across processes).

**Fix shipped (`_stable_seed_int(seed)` in `trajectory.py`):** swapped
`hash()` for first 4 bytes of `md5(seed.encode())`. Pinned canonical
seed → int values in `test_solver_trajectory.StableSeedIntTests` so
the regression can't return silently.

**Same-process Flush vs baseline re-measurement (post-fix):**

| Seed | Baseline ante | Flush ante | Δ |
|---|---|---|---|
| AAAAAAA | 5 | 4 | -1 |
| BBBBBBB | 4 | 3 | -1 |
| CCCCCCC | 3 | 3 | 0 |
| 1234567 | 6 | 5 | -1 |
| XKCD0001 | 6 | 3 | -3 |
| Avg | 4.8 | 3.6 | **-1.2** |

Flush still regresses on stable-seed same-process comparison. The
M6a regression is real, not process noise — but only by ~1 ante on
average (smaller than the apparent -1.6 from the unfixed
cross-process measurement). The additive bias as designed genuinely
hurts more than it helps for a single-archetype commit.

This shifts M6b from "nice-to-have" to "necessary": single
archetypes aren't a win, so the value (if any) of the archetype
mechanism comes from the portfolio. M6b's `solve_seed_multi_archetype`
always includes baseline as a candidate, so picking the best across
N attempts guarantees solver quality ≥ baseline. The question is
whether any archetype EVER beats baseline on a given seed.

**M6 acceptance criteria (gating M7 dataset CLI):**
- Multi-archetype solve produces best trajectories at least as good
  as no-archetype baseline across a 20-seed measurement.
- Some seeds (target ≥20%) measurably improve.
- Throughput projection for the chosen archetype-set size fits a
  reasonable dataset window (target <2 weeks for 10k seeds at 8 cores).

The structural change that's supposed to actually move winrate. M4+M5
layer search on top of basic_strategy_bot's heuristics, so decisions
still rationalize the same greedy patterns ("full slots no xmult"
remains the dominant loss mode). Archetype branching is what lets the
solver commit to a build early and route shop decisions toward it.

**Two-phase rollout:**

M6a — Scaffold + single archetype:
- `Archetype` dataclass: name, target hand types, key joker keys,
  shop-scoring delta function, optional blind-play preference.
- `ArchetypeSolverPolicy` wraps `SolverPolicy` but biases shop and
  blind decisions toward one archetype. No branching yet — just
  proves the archetype-aware decision path moves trajectories
  measurably.
- Implement Flush archetype first (simplest: shop-score suited cards
  + Flush-related jokers + Magic/Crystal Ball tarots).

M6b — Multi-archetype branching:
- Run N candidate archetypes as separate per-seed search roots.
- Pick the deepest/winning trajectory.
- Adaptive budget: stop branching to new archetypes if one reaches
  ante 8.
- Implement remaining archetypes: scaling joker, high-card-mult,
  pair retrigger, polychrome accumulation, steel xmult, discard
  economy.

**Open question:** how aggressively does the archetype bias shop
decisions? Hard bias ("only buy items matching this archetype") may
miss obvious upgrades. Soft bias ("add archetype-fit to existing
shop_leaf_value") is gentler. Start soft, measure.

### M7 — Dataset CLI

**Status:** design pending M6b verdict.

`python -m balatro_ai.solver.dataset --seeds 100 --out .data/trajectories.jsonl`

**Design depends on M6b outcome:**

- If multi-archetype solve produces best trajectories that beat
  baseline on a meaningful fraction of seeds → dataset uses
  `solve_seed_multi_archetype` per seed (5× compute but better
  trajectories).
- If multi-archetype doesn't reliably beat baseline → dataset uses
  the plain no-archetype `SolverPolicy` (1× compute, baseline
  quality, but we ship a corpus and pivot to a different
  archetype/search redesign before going bigger).

**Common shape (independent of M6b):**

- Parallel across cores via `multiprocessing.Pool` (8 workers by
  default).
- Per-seed wall-clock timeout (default 30 min) to recover from
  pathological cases; timed-out seeds are recorded with a
  `terminated_reason="TIMEOUT"` so they're filterable.
- JSONL output, one row per seed:
  ```
  {
    "seed": "AAAAAAA",
    "stake": "white",
    "won": false,
    "final_ante": 6,
    "final_score": 25492,
    "wall_seconds": 474.0,
    "terminated_reason": "RUN_OVER",
    "best_archetype": "baseline",          # only if multi-archetype
    "attempt_summary": [...],              # only if multi-archetype
    "steps": [...]                          # if --include-steps
  }
  ```
- `--record-steps` flag (default off) controls per-step recording.
  Phase 8 imitation training needs step records; lightweight
  exploration/sweep runs don't.
- Resumable: skip seeds already present in the output file.

**Acceptance:** generate a 100-seed dataset that round-trips through
the readout helper without errors and projects to <2 weeks for a
10k-seed run on 8 cores (target: <3 days at base scale).

### M8 — 1000-seed validation run

**Status:** not started.

Run M7 on 1000 white-stake seeds. Measure:

- Solver winrate (target ≥30% on white stake)
- Average ante reached on losing seeds (target ≥6)
- Wall-clock total (validates the 10k throughput projection)

### M9 — Phase 8 dataset

**Status:** not started.

10k–50k seeds, white stake first, then expand to other stakes if quality is
adequate. Versioned outputs under `.data/phase8-dataset-v*/`.

---

## 4. Decisions log

- **2026-05-25 (M1):** Reuse `_with_blind_selection_surface` from `local_runner`
  rather than building a fresh blind-surface helper in `solver/`. Cost: tight
  cross-package coupling (private-ish import). Benefit: ~150 lines saved and
  the local-sim test suite stays the single source of truth for surface
  modifier shape. Trade accepted; if local_runner ever refactors that helper,
  the solver gets the change for free.
- **2026-05-25 (M2):** Drive trajectory via `LocalBalatroSimulator.step()`
  for action dispatch, not a fresh dispatch in `solver/`. Cost: shop/pack
  sampling uses Python `Random`, not seed-faithful. Benefit: every action
  type (BUY/SELL/REROLL/OPEN_PACK/CHOOSE_PACK_CARD/USE_CONSUMABLE/...) is
  already handled. Plan: M4+ swaps in seed-faithful surface predictions at
  the dispatch boundary, but the action plumbing stays. Trade accepted.
- **2026-05-25 (M3):** Continue to M4 (whole-blind beam search) without
  optimization detour. Median 25.67s/seed on basic_strategy_bot leaves
  enough headroom for the actual solver to be 2–3× slower per decision
  and still hit the dataset budget.
- **2026-05-25 (M4):** Wrap existing `best_hand_action` rather than
  build a fresh beam search in `solver/`. Cost: tied to the live bot's
  search infrastructure, including its heuristic preprocessing. Benefit:
  ~200+ lines saved, plus the search machinery is already test-covered
  by `search_bot_v2`. Trade accepted.
- **2026-05-25 (M4):** Default beam config `depth=3, width=2` matching
  `search_bot_v2`. Higher (8/4) blew up cost without proportional gain
  in single-step smoke. Will revisit when M6 adds archetype branching —
  more aggressive per-archetype beam may be worth it if the archetypes
  prune the action space.
- **2026-05-25 (M4):** Continue to M5 despite the 9.4× slowdown vs
  basic_strategy. Most of the cost is in heuristic preprocessing, not
  the beam — three concrete optimization levers if M5+M6 push us
  further out (bypass `_opening_first_blind_hunt_action`, cache,
  thinner action enumeration). Defer optimization until we know the
  total search overhead.
- **2026-05-25 (M5):** Wrap `best_shop_action` for SHOP phase, mirror
  M4's wrap-existing approach. Cost: shop sampling uses
  `ShopSampler.from_default_data` (still not seed-faithful inside the
  beam's hypothetical futures); benefit: ~150 lines saved. Accepted —
  the shop sampler's role inside the search beam is for hypothetical
  rerolls, where seed-faithful sampling matters less than throughput.
- **2026-05-25 (M5 → M5.5):** Stop and optimize before M6. M5 hit
  768s/seed = 30× baseline = 11 days for 10k seeds at 8 cores. Adding
  M6's archetype branching on top would compound past the 2-week mark.
  The plan's section 7 explicitly called this out: "prototype on 10
  seeds before committing." The 10× tolerance in section 4 was for
  M4+M5+M6 *combined*; M5 alone is already at 30× and M6 hasn't
  landed. Optimize first.

---

## 5. Surprises / lessons

- **M1:** `simulate_select_blind` takes `drawn_cards` as a positional
  second arg, not an `Action`. The select-blind action carries no
  card_indices; "what's drawn at start of blind" is modeled as injected
  state, not as part of the action. Worth flagging in any solver code
  that constructs select-blind actions.
- **M2 (caught and fixed):** Initial 5-field state signature for stuck
  detection was too coarse. Three consecutive discards have identical
  `(phase, ante, score, money, hands_remaining)` because discards don't
  decrement `hands_remaining`. The stuck detector killed legitimate runs
  4 steps in. The 10-field signature catches every action type that
  actually changes state; threshold raised from 3 to 10 for safety.
- **M3:** The basic_strategy_bot is non-deterministic across runs when
  reset with `seed=0`. AAAAAAA reached ante 6 the first run and ante 5
  the second on identical configuration. Likely the bot's internal
  `_fallback: RandomBot` advances based on call count, not state hash.
  Not blocking M4 work — but means trajectory throughput numbers have
  variance; the 25.67s median is the right anchor, not any single seed.
- **M4:** "Beam search" was 90% misleading as a description of what's
  actually slow. Profiling the policy showed ~90% of per-call cost in
  the existing `_opening_first_blind_hunt_action` heuristic that runs
  BEFORE the beam — the beam itself is cheap. Any optimization push
  should target that preprocessing, not the beam structure.
- **M5:** Shop search added 3.3× compute on top of M4 (235s → 768s).
  The plan's "2–3× per milestone is fine" tolerance was per-milestone,
  but at this rate M6 lands us past 2 weeks for 10k seeds. The
  throughput risk called out in section 7 of the parent plan was
  real — needs an optimization detour, not a "we'll deal with it
  later" deferral.
- **M5.5:** The "bypass preprocessing" optimization was the wrong
  call. The M4 profile showed ~90% of cost in
  `_opening_first_blind_hunt_action`, but I read that as "the
  preprocessing is overhead." It's actually a shortcut: when the
  preprocessing finds a clear discard, the beam doesn't run. Removing
  it forces the beam onto every state and made PLAY decisions ~5×
  slower (1.5s → 8s) in the measured case. Lesson: "expensive" in a
  profile isn't the same as "wasteful." Some expensive paths are
  short-circuiting more-expensive paths.
- **M5.5:** PLAY decision cost dominates trajectory cost. Back-of-
  envelope: 100 plays × ~7-8s + 50 shops × ~1.5s = ~800s. Shop
  optimization alone caps savings at ~10-15%. Real wins need PLAY
  optimization (smaller beam, or content-based caching of
  preprocessing results across calls in the same blind).
- **M5.5:** Identity-based caching (`_identity_cached_value`,
  `_decision_cached` keyed on `id(state)`) doesn't help search
  workloads where every leaf creates a fresh GameState. Content-based
  caching would help but requires invasive changes to GameState
  hashing. Worth doing if the easy wins don't get us to budget.

---

## 6. Open questions

- **Does `forward_sim` actually run end-to-end?** It's been validated
  per-action across many transitions, but never sequenced through a full
  run as the driver. M1's smoke test catches this.
- **What's the cost of `simulate_play` in microseconds?** Estimate was
  200–500μs; M3 measures actual. If it's >2ms, throughput math fails.
- **Lazy vs eager game-tree materialization.** Eager makes search code
  simpler; lazy saves memory and avoids predicting branches we never
  visit. Starting lazy.
- **How many archetypes is too many?** Each one is a separate search per
  seed. 4 is the M6 plan; if individual archetype runs take 30s, that's
  120s per seed already. Adaptive budget (stop branching when one
  archetype reaches ante 8) is the obvious mitigation but adds
  complexity.
