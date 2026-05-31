# Phase 7: Offline Solver for Phase 8 Training Data

**Status:** Active. Core RNG surface validation done; Rust core (Phases 1-4a) accelerates the solver inner loop. The solver itself is BUILT (`solver/policy.py`, milestones in [`SOLVER_PLAN.md`](SOLVER_PLAN.md)) and generating data — section 5 below ("Solver design") is no longer "not started". The active gate is now **raising the solver's data-gen winrate** (~1% → ~8% so far via value-function bug fixes) before a large dataset run; see `PROGRESS.md` (2026-05-30/31) and memory `project_datagen_speed.md`.
**Last updated:** 2026-05-31.

> **Cross-reference:** the Rust port — which speeds up forward_sim and
> evaluate by 1-2 orders of magnitude — lives in
> [`RUST_PORT_PLAN.md`](RUST_PORT_PLAN.md). The dataset-generation
> cost estimates in §6 below assume the Rust core is in place.

This document tracks the **pivot from "make the live bot stronger" to "build
an offline expert solver"** as the path to Phase 8 neural training data. The
high-level project plan in `PLAN.md` describes the original phase 0–15
roadmap; this document is the working plan for the active Phase 7 → Phase 8
slice.

---

## 1. Why this pivot exists

Original framing of Phase 7: improve `basic_strategy_bot` and `search_bot_v2`
until they win consistently, then use their replays as Phase 8 imitation
data. The Phase 8 gate is **40–50% white-stake winrate**.

What actually happened: after weeks of leaf-tuning and search-variant
experiments, every variant ties `basic_strategy_bot` at **~5–7% winrate**.
Loss-pattern analysis from the 1000-seed baseline (`.data/basic-strategy-
baseline-1000-analysis.txt`) shows the bot reliably builds incoherent runs:

- 22.8% deaths from "full slots, no xmult" (fills joker slots without
  scaling jokers)
- 20.5% from "money held while missing power" (hoards cash, doesn't pivot)
- 16.1% from "very high money death" (dies rich)

No amount of leaf tuning on the live bot is going to bridge this gap. The
incremental approach is a trap.

## 2. The new architecture

**Stop trying to make the bot win.** Instead:

- Build an **offline per-seed expert solver** with no real-time budget,
  capable of finding winning (or maximum-depth) trajectories per seed.
- Use those trajectories as the training dataset for Phase 8.
- The bot you ship as the data generator does not need to be the bot you
  ship as the policy network.

Inspiration: AlphaGo's network was weak; MCTS+rollouts were the teacher.

### Design principles for the solver

- **Keep all seeds, filter by trajectory quality within each seed.** Even
  losses at ante 8 with coherent builds train the value head. Filtering
  out hard seeds would teach the network nothing about doomed positions.
- **Branch on build archetype at the root.** Flush, scaling, retrigger,
  high-card-mult, etc. The solver commits to a build early and routes
  shop decisions toward it. This directly addresses the "full slots, no
  xmult" loss pattern.
- **Pure-sim path preferred over bridge-execution.** Bridge throughput is
  ~20 min/seed; sim throughput is seconds/seed. 10k–50k seeds is the
  target dataset size — bridge can't reach that on a single workstation.
- **Validate against the live game.** Anything in the pure-sim pipeline
  needs ground-truth checks via captured bridge fixtures; bugs in the
  sim or RNG would poison the dataset silently.

### Why RNG matching is on the critical path

The pure-sim solver needs to predict, given only a run seed:
- The initial deck shuffle order
- Shop contents per ante
- Boss blind per ante
- Tag, voucher, and pack contents

Without seed-faithful RNG, the solver can only plan over actual observed
state — which requires bridge execution and kills throughput. With it, the
solver can plan over the full game tree from a seed string alone.

---

## 3. Work completed

### 3.1 Sim correctness baseline (DONE)

The deterministic surface of `forward_sim.py` is validated against
241 BalatroBench runs.

| Action type | Exact matches | Pct |
|---|---|
| play_hand | 1456 / 1456 | 100.0% |
| discard | 1253 / 1253 | 100.0% |
| end_shop | 1809 / 1809 | 100.0% |
| reroll | 335 / 335 | 100.0% |
| sell | 217 / 221 | 98.2% |
| **Overall** | **5070 / 5074** | **99.9%** |

**Tooling:** `src/balatro_ai/eval/sim_divergence_audit.py` (the v2 diagnostic).

**Bugs fixed during the audit:**
- **"The Arm" boss blind** didn't decrement stored hand levels (24 cases).
  Fix in `forward_sim._hand_levels_after_play`.
- **"To Do List" $4 trigger** was completely missing (10 cases). Fix in
  `hand_evaluator._to_do_list_target` + integration. Mid-fix discovery:
  pays on EVERY matching play, not just first hand of round.

**Known minor sim issues, deferred:**
- Selling Drunkard mid-shop decrements `discards_remaining` when it
  shouldn't (4 cases).
- Reroll at negative money with Credit Card raises (5 cases) — state
  parsing doesn't extract `bankrupt_at` modifier from joker presence.

Both are sub-0.1% impact, well-characterized, ~30 min each to fix.

### 3.2 RNG matching infrastructure (DONE)

All under `src/balatro_ai/rng/`.

- **`pseudohash.py`** — Balatro's `pseudohash(str)` and `pseudoseed_step`,
  bit-exact port from `functions/misc_functions.lua`. Includes the
  `string.format("%.13f", ...)` + `math.abs` round-trip that prevents
  float drift.
- **`balatro_rng.py`** — `BalatroRNG` class with per-key streams. Default
  `mix_hashed_seed=True` applies `(state + hashed_seed) / 2` per call.
- **`luajit_prng.py`** — LuaJIT TW223 (Tausworthe-223) PRNG and the
  `random_seed(d)` transform (`d = d*pi + e` four times, bit-reinterpret
  to uint64). This is what `math.random` calls inside LÖVE 11.x.
- **`deck.py`** — standard Red-deck construction in verified pre-shuffle
  order (`C,D,H,S × 2,3,4,5,6,7,8,9,A,J,K,Q,T`), recovered from card IDs
  in captured fixtures.
- **`pools.py`** — Tarot (22), Planet (12), and Joker pools by rarity
  (61+64+20+5 = 150), extracted via regex from `game.lua`. Sorted by
  `order` field, matching how Balatro builds `G.P_CENTER_POOLS`.

### 3.3 Validation harness (DONE)

- **`rng/capture.py`** — captures initial deck order to
  `.data/rng-validation/seed_*_red_white.json`.
- **`rng/capture_shop.py`** — drives `basic_strategy_bot` through the
  small blind and captures the first-shop state to
  `.data/rng-validation/shop_seed_*_red_white.json`.
- **`rng/capture_shop_sequence.py`** captures no-purchase multi-shop
  sequences by using the dev `scenario` endpoint to clear blinds, saving
  `.data/rng-validation/shop_sequence_seed_*_red_white.json`.
- **`rng/capture_surfaces.py`** captures opened booster-pack states to
  `.data/rng-validation/pack_seed_*_*.json`; it can either open a visible
  first-shop pack or force one normal pack per kind through the dev
  `scenario` endpoint.
- **`rng/capture_spectral_helpers.py`** captures controlled Familiar/Grim/
  Incantation uses to validate per-created-card Spectral RNG.
- **`rng/validate.py`** — grid-search over candidate `(algorithm,
  shuffle_key, mix_hashed_seed, seed_strategy)` configs against captured
  fixtures.
- **`rng/validate_surfaces.py`** normalizes opened-pack bridge cards and
  compares them to `surfaces.predict_pack_contents`.
- **`rng/validate_shop_sequence.py`** compares carried-RNG shop-card and
  booster-slot predictions against captured multi-shop fixtures.
- **`rng/validate_spectral_helpers.py`** compares enhanced cards created by
  Familiar/Grim/Incantation against the per-card helper predictions.
- **`Mods/rngprobe/`** — Lovely-injector mod that hooks
  `create_card_for_shop` and logs the actual `G.GAME.pseudorandom['cdt1']`
  pre-state, post-state, and `polled_rate` to
  `%APPDATA%\Balatro\rngprobe.log`. Critical instrument for confirming
  Python predictions match the live game digit-for-digit.

### 3.4 Deck shuffle SOLVED

Algorithm: **`luajit_after_pseudoseed`**.

```
pseudoseed('shuffle')   -> float F
math.randomseed(F)      -> seed LuaJIT TW223 via (d*pi + e) bit-reinterpret
for i = 52, 2, -1 do
    j = math.random(i)
    swap deck[i], deck[j]
end
```

**Match rate:** 4/4 captured seeds, full 52-card deck each.

**Critical bug found during this:** my old `pseudohash` walked the string
with the iteration counter as the position multiplier, but Lua's
`for i = #str, 1, -1` keeps `i` as the visited character's ORIGINAL
1-indexed position. The bug was invisible on single-char strings (where
position 1 = iteration 1) and produced totally different output for any
multi-char input.

### 3.5 Shop pool SOLVED (ante-1 basic case)

Algorithm verified end-to-end:

```
For each shop slot:
  1. cdt_float = pseudoseed('cdt' + ante)
     polled = LuaJITPRNG.seeded(cdt_float).next_double() * total_rate
     -> category by cumulative threshold over Steamodded type order
        [Joker, playing_card, Tarot, Planet, Spectral]
  2. If Joker:
       rarity_float = pseudoseed('rarity' + ante + 'sho')
       d = LuaJITPRNG.seeded(rarity_float).next_double()
       rarity = (d > 0.95) ? 3 : (d > 0.7) ? 2 : 1
       pool = JOKER_POOL_BY_RARITY[rarity]   (with locked/used filtered)
       key  = 'Joker' + rarity + 'sho' + ante
  3. Else: pool = TAROT_POOL or planet_pool_for_ante(...)
           key  = type + 'sho' + ante
  4. item = pseudorandom_element(pool, pseudoseed(key))
            (math.randomseed + math.random(N), retry _resample on UNAVAILABLE)
```

**Match rate:** 8/8 (4 seeds × 2 slots) exact category + exact item.

Implemented in `src/balatro_ai/rng/shop.py` as `predict_first_shop(seed)`.
Tests in `tests/test_rng_shop.py`.

### 3.6 Core RNG surface predictors (DONE)

Implemented in `src/balatro_ai/rng/surfaces.py`:

- Initial run surface: boss blind, current voucher, Small/Big skip tags.
- Shop surface: rerollable shop cards, voucher slot, booster-pack slots.
- Boss selection with source min-ante/showdown pools and least-used filtering.
- Voucher and tag pools with source availability/resample behavior.
- General `create_card`-style center prediction for Joker/Tarot/Planet/
  Spectral/Base/Enhanced cards.
- Joker edition, eternal/perishable, and rental polls.
- Booster pack selection and contents for Buffoon, Celestial, Arcana,
  Standard, and Spectral packs.
- Standard-pack playing-card front, edition, and seal RNG.
- Per-card spectral helpers for Sigil, Ouija, Familiar, Grim, and
  Incantation-style random card creation.

Bridge fixture validation now covers the four canonical first-shop seeds for
boss, Small/Big tags, voucher, shop cards, and booster packs. Those tests live
in `tests/test_rng_surfaces.py`. Opened-pack capture and offline validation
tooling now lives in `rng/capture_surfaces.py`, `rng/validate_surfaces.py`,
and `tests/test_rng_pack_surfaces.py`; 24/24 pack fixtures match, including
Omen Globe Arcana spectral rolls, Telescope Celestial planet targeting, and
Glow Up edition-rate Standard packs. No-purchase shop-sequence validation now covers 51/51 captured shops
across the four canonical seeds through the first ante-3 shop: 24 White Stake
shops plus 24 Gold Stake shops with eternal, perishable, and rental sticker
polls plus Magic Trick/Illusion voucher-rate shop fixtures. Spectral helper
validation covers Familiar, Grim, and Incantation created-card RNG for seed
`AAAAAAA`.

Important edge found by the shop-sequence fixtures: source joker pools honor
`enhancement_gate`. Steel Joker, Stone Joker, Lucky Cat, Golden Ticket, and
Glass Joker are unavailable unless the current deck contains the matching
enhancement.

Important sticker edge found by the Gold Stake fixtures: `set_eternal` and
`set_perishable` also honor each joker center's compatibility flags. Runner
and Red Card, for example, can roll through the perishable poll without
receiving the sticker.

Important distinction: this is now source-faithful predictor coverage for the
offline solver's major surfaces. The remaining RNG edge is narrower: Illusion
shop playing-card generation also advances the global `math.random` state that
the first Buffoon pack path uses, and Overstock/slot-count voucher fixtures can
still be added for extra confidence.

### 3.7 Project memory updates (DONE)

- `~/.claude/projects/.../memory/project_sim_correctness_baseline.md`
  documents the verified algorithms, the bugs that took multiple sessions
  to find, and the file paths for next-session re-orientation.
- `~/.claude/projects/.../memory/project_phase7_search_status.md` records
  the architectural pivot so future sessions don't redo the leaf-tuning
  trap.

---

## 4. Work in progress / immediately next

### 4.1 Extend bridge validation for RNG surfaces (~2-4 days)

The predictor coverage exists, but only the initial shop has fixture-backed
validation across boss/tags/voucher/shop/boosters. The next validation slices:

| Surface | Status |
|---|---|
| Boss blind selection | Predictor implemented; initial fixtures pass |
| Voucher per ante | Predictor implemented; initial fixtures pass |
| Tag rolls | Predictor implemented; initial fixtures pass |
| Shop cards ante 2+ | Predictor + White/Gold sequence fixtures pass through first ante-3 shop |
| Booster pack slots | Predictor + White/Gold sequence fixtures pass through first ante-3 shop |
| Pack contents | Predictor + capture/validator pass on 24 fixtures, including Omen Globe/Telescope/Glow Up voucher paths |
| Edition polls | Natural shop/pack editions and Glow Up Standard-pack edition-rate path covered; tag-guaranteed paths still need fixtures |
| Eternal/perishable/rental | Predictor + Gold Stake fixtures pass through first ante-3 shop |
| Voucher-influenced paths | Omen Globe, Telescope, Glow Up, and Magic Trick/Illusion shop-rate fixtures pass; Illusion playing-card global-PRNG carry and Overstock fixtures remain |
| Per-card spectral RNG | Predictor + scenario fixtures pass for Familiar/Grim/Incantation |

For each remaining validation slice: capture fixture via bridge -> compare the
existing predictor -> fix edge cases -> add to
`tests/test_rng_*.py`.

The `predict_*` functions now exist; the remaining work here is fixture
expansion and edge-case correction, not first-pass implementation.

### 4.2 Wrap deferred sim bugs (~1 hour)

- **Drunkard mid-shop sell**: fix `_state_with_current_discard_delta_change`
  to no-op when phase=SHOP.
- **Credit Card negative-money reroll**: derive `bankrupt_at` from joker
  presence at state-parse time, or check Credit Card in `_can_afford_cost`
  directly.

Both bring `forward_sim` from 99.9% to 100% exact on the audited surface.

---

## 5. Solver design (BUILT — see SOLVER_PLAN.md)

The solver is implemented as `solver/policy.py::SolverPolicy` and is generating
data; milestone-by-milestone status is in [`SOLVER_PLAN.md`](SOLVER_PLAN.md)
(M1–M5.5 done). It runs a whole-blind beam play search + a shop beam over a
build-aware value function (the archetype-root-branching idea below was
deprioritized — see SOLVER_PLAN.md M6). The active work is raising its data-gen
winrate (`PROGRESS.md`). The original design sketch is retained below for
context.

### 5.1 Per-seed solver structure

```
solve(seed, max_minutes=10) -> Trajectory:
    # 1. Materialize the full game tree from this seed (deck order, all
    #    blinds, all shops, all packs predicted via the RNG layer).
    game = SeedExpandedGame(seed)

    # 2. Try multiple build archetypes as separate root branches.
    archetype_results = []
    for archetype in CANDIDATE_ARCHETYPES:
        result = search_with_archetype(game, archetype, time_budget=max_minutes/N)
        archetype_results.append(result)

    # 3. Pick the deepest/winningest trajectory across archetypes.
    return best(archetype_results)
```

### 5.2 Archetypes to branch on

Initial set (extend later):
- Flush build (matching suit accumulation)
- Scaling joker (Green Joker / Ride the Bus / Square / Wee)
- High-card-mult (Mime / Raised Fist with held aces)
- Pair retrigger (Sock and Buskin + face Pair-mult jokers)
- Polychrome/holographic accumulation (edition value)
- Steel card xmult (Justice tarot routing)
- Discard economy (Mail-In Rebate / Faceless / Wheel-flipping)

Each archetype has a small score function for shop decisions: "does this
item advance my build?"

### 5.3 Search within an archetype

- **Whole-blind beam** for hand-play decisions (replaces today's 1-ply
  hand search). Depth = full blind, beam width ≈ 8–16, leaf eval =
  clear probability × expected surplus.
- **Shop search** as separate tree: BUY/SELL/REROLL/END_SHOP, with the
  archetype's score function as leaf value.
- **Pack opening** with full enumeration of pack contents (RNG layer
  predicts contents).
- **Boss-blind-aware planning** — boss is known from the RNG layer, so
  the search can prepare specifically (e.g., avoid building hands the
  boss debuffs).

### 5.4 Outputs

Per seed: a `Trajectory(actions, outcome, build_archetype, ante_reached)`
record. For dataset generation:

- **Policy targets:** the action taken at each state.
- **Value targets:** outcome (won/lost), ante reached, score reached.
- All seeds kept (including losses, which teach the value head).

Target dataset size: **10k–50k seeds**, single workstation, ~1–10 days
wall-clock with the pure-sim solver.

---

## 6. Phase 8 transition (NOT STARTED)

Once the solver produces a dataset:

- **Imitation learning** on policy: state → action distribution.
- **Value head** on outcome: state → win probability / expected ante.
- Architecture: small transformer or MLP over a flat state encoding
  (existing state encoding in `src/balatro_ai/api/state.py` is the
  starting point).

The bot the network produces does NOT need to look like the solver. The
solver's job ends at trajectory generation.

---

## 7. Open questions / risks

- **Steamodded vs vanilla RNG paths.** Some shop logic is overridden by
  `SMODS.poll_object_type` when `SMODS.optional_features.object_weights`
  is true. Default is false; no captured fixture has it on. If a future
  user has another mod that flips it on, our predictions break.
  *Mitigation:* document the assumption; add a runtime check in
  `predict_first_shop` that bails clearly if the modifier is set.
- **Joker unlocks per profile.** Some jokers ship `unlocked = false`. The
  bridge captures come from the user's existing profile, where many are
  unlocked. Our `pools.py` currently treats locked jokers as available
  by default; this is correct for the captured fixtures but may not be
  for a fresh-profile run.
  *Mitigation:* the `joker_pool_for_rarity(unlocked=...)` helper accepts
  an explicit unlocked set when a fresh-profile prediction is needed.
- **Ante 2+ state continuity.** Pseudoseed states for `'rarity1sho'`,
  `'Joker1sho1'` are per-ante (the ante suffix changes). But some keys
  reuse across antes (`'soul_*'`, `'etperpoll'+ante`). Need to verify
  that ante-2 predictions still match after ante-1 has advanced various
  per-key states.
- **The 1% of forward_sim that's wrong.** The Drunkard + Credit Card
  bugs are known. Anything else lurking? The diagnostic should be
  re-run any time `forward_sim` is touched, especially after solver work
  exercises new code paths.
- **Solver throughput estimate.** "1–10 days for 10k seeds on a single
  workstation" assumes the solver itself is fast — milliseconds per
  decision via the pure-sim path. If actual search needs heavy MCTS-style
  rollouts (e.g., for shop EV with multiple unknowns), throughput could
  drop 10–100x.
  *Mitigation:* prototype on 10 seeds before committing to a 10k run.
- **Bridge stability for ground-truth captures.** Lovely-injector
  occasionally crashes with "process cannot access the file" on dump
  dirs. Workaround scripted (`PowerShell Remove-Item ...` before
  `uvx balatrobot serve`), but flaky. Affects new RNG validations more
  than steady-state work.

---

## 8. File map (current state)

```
src/balatro_ai/
  rng/                              <- NEW THIS WEEK
    __init__.py
    pseudohash.py                   pseudohash + pseudoseed_step primitives
    balatro_rng.py                  per-key streams (BalatroRNG)
    luajit_prng.py                  LuaJIT TW223 PRNG + random_seed
    xoroshiro.py                    LOVE xoroshiro128+ (unused, kept for ref)
    deck.py                         standard Red deck construction
    pools.py                        Tarot/Planet/Joker pools from game.lua
    shop.py                         predict_first_shop(seed) end-to-end
    surfaces.py                     boss/voucher/tag/shop/pack RNG predictors
    capture.py                      bridge fixture capture (initial deck)
    capture_shop.py                 bridge fixture capture (first shop)
    capture_shop_sequence.py        bridge fixture capture (multi-shop sequence)
    capture_surfaces.py             bridge fixture capture (opened packs)
    capture_spectral_helpers.py     bridge fixture capture (Spectral helpers)
    validate.py                     grid-search validator + report CLI
    validate_shop_sequence.py       multi-shop fixture validator
    validate_spectral_helpers.py    Spectral helper fixture validator
    validate_surfaces.py            opened-pack fixture validator

  eval/
    sim_divergence_audit.py         <- ADDED EARLIER THIS WEEK

tests/
  test_balatro_rng.py               pseudohash, pseudoseed, BalatroRNG
  test_rng_deck_and_validate.py     deck construction + predict_starting_hand
  test_rng_against_bridge.py        full-deck fixtures (skips if absent)
  test_rng_shop.py                  shop predictions vs captured fixtures
  test_rng_shop_sequence.py         multi-shop sequence fixtures
  test_rng_surfaces.py              boss/tag/voucher/shop/booster fixtures
  test_rng_pack_surfaces.py         opened-pack fixture validation helpers
  test_rng_spectral_helpers.py      Spectral helper fixture validation
  test_sim_divergence_audit.py      <- ADDED EARLIER THIS WEEK

.data/
  rng-validation/                   captured ground truth
    seed_*_red_white.json           initial deck order per seed
    shop_seed_*_red_white.json      first-shop state per seed
    shop_sequence_seed_*_red_white.json multi-shop state per seed
    shop_sequence_seed_*_red_gold.json Gold Stake sticker multi-shop states
    pack_seed_*_*.json              opened-pack states per seed/pack
    spectral_seed_*_*.json          Spectral helper use states
  balatro-source/                   extracted Lua + LOVE + LuaJIT source
  sim-divergence-audit-v*.txt       audit run output

C:\Users\Wyatt\AppData\Roaming\Balatro\Mods\rngprobe\
  rngprobe.json                     SMODS manifest
  rngprobe.lua                      Lua-side log helper
  lovely/probe.toml                 patches to log pseudoseed values
```

## 9. Re-orientation steps for the next session

If you're picking this up cold:

1. Read `~/.claude/projects/.../memory/project_sim_correctness_baseline.md`
   and `project_phase7_search_status.md` for the verified algorithm details
   and the architectural framing.
2. `python -m unittest discover -s tests` to confirm the suite passes.
3. `cat .data/sim-divergence-audit-v5.txt` for the sim correctness baseline.
4. `python -m balatro_ai.rng.validate --all` to confirm captured deck fixtures
   still predict.
5. `python -m balatro_ai.rng.validate_surfaces --all` after pack fixtures are
   captured.
6. `python -m balatro_ai.rng.validate_shop_sequence --all` after shop-sequence
   fixtures are captured.
7. `python -m balatro_ai.rng.validate_spectral_helpers --all` after Spectral
   helper fixtures are captured.
8. `python -m unittest discover -s tests -p "test_rng*.py"` to confirm RNG
   predictions and fixture checks still match.

Then the next concrete work item is section 4.1 validation or the first
section-5 solver skeleton.
