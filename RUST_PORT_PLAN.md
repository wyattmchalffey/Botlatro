# Rust Port Plan: `botlatro-core`

**Status:** Phase 1 + Phase 2 no-joker path + Phase 2d (33 jokers)
+ ordering refactor + `_score_action` wire-in complete (2026-05-26).
Toolchain validated on Windows. PoC revealed FFI-overhead trap which
reshaped the port strategy — see §3.

Speedup numbers:
- **Standalone `evaluate_simple` is 65-75× faster** than Python
  `evaluate_played_cards` on vanilla 5-card hands.
- **Full-trajectory speedup is 16-20%** (AAAAAAA: 291s M4 baseline →
  ~231-244s with current Rust coverage). Quality identical
  (ante=5/score=5100). The big jump from 12% → 20% came from removing
  the "any joker has edition → bail" check in Batch 6 — most mid-run
  jokers acquire editions naturally, and that check was hiding them
  all from Rust. The held-card pass (Batch 7) adds some pass-through
  cost on no-held-joker runs (held-cards scan), partially eating the
  Batch 6 gain on AAAAAAA which doesn't use held-card jokers.
- **~60 jokers** + card editions + joker editions + held-card pass
  + scaling jokers via metadata + ctx-dependent jokers (Card Sharp,
  Supernova, Blue Joker, Constellation, Ramen) supported. ~85 jokers
  still bail. Each new joker port = ~10-30 lines of Rust + parity
  test + small trajectory wall reduction.
**Parent:** [`SOLVER_OPTIMIZATION_PLAN.md`](SOLVER_OPTIMIZATION_PLAN.md) — the
broader optimization roadmap. This document is the Rust-specific
plan that supersedes the Cython attempts in that doc (which only
delivered ~1.5-2× and were the wrong tool for the problem).
**Last updated:** 2026-05-26.

---

## 1. Why Rust, not Cython or pure Python

Stockfish evaluates ~100M chess positions per second. Our Python
solver evaluates ~10K Balatro states per second. **That's a 4-5 order
of magnitude gap.** Python micro-optimization (Cython, memoization,
algorithmic tweaks) measured at 1.5-3× in Tier 1 work — useful but
not the missing factor of 10,000.

The gap closes when:
- **Native compiled code** removes Python bytecode interpretation
  (10-100×)
- **Packed state representation** removes allocation + memory
  pressure (10-100×)
- **Algorithmic improvements on top** (alpha-beta, transposition
  tables) prune the search tree by another 10-1000×

Cumulatively: 10^5-10^7×. That's the difference between "depth 3 in
100s" (current) and "depth 12 in 100ms" (chess engine equivalent).

Rust delivers all three layers; Cython only gives a fraction of (1).
Rust + PyO3 + maturin is the toolchain choice (battle-tested by
polars, pydantic v2, ruff).

---

## 2. Current state (2026-05-26)

**Infrastructure:**
- ✅ Rust toolchain installed (cargo 1.95, rustc 1.95)
- ✅ maturin 1.13.3 installed
- ✅ `botlatro-core/` crate scaffolded with PyO3 0.22 + ABI3 for Python 3.11+
- ✅ `pip install ./botlatro-core` builds + installs the native extension

**Phase 2 (hand evaluation) — no-joker path COMPLETE:**
- ✅ `identify_hand_type` (no-Stone/no-Wild fast path) — **27× speedup**
- ✅ `scoring_indices` (per-hand-type scoring card selector) — parity
  green on 500-hand random fuzz.
- ✅ `card_chip_value` (per-card chip contribution) — parity green on
  500-card fuzz including suit-debuff + bonus enhancement.
- ✅ `evaluate_simple` (composed end-to-end: hand_type + scoring +
  chip value + level math) — **75× speedup** on vanilla 5-card hands
  vs Python `evaluate_played_cards`.
- ✅ `Rank::chip_value()` separated from `straight_value()` —
  matched Python's RANK_VALUES semantics. Caught by fuzz on a High
  Card hand returning `[3]` vs Python's `[0]`.
- ✅ Wire-in to `_score_action` complete. `state_value._score_action_uncached`
  tries `balatro_core.evaluate_simple_with_levels` first; falls back
  to Python `evaluate_played_cards` when state has jokers or a
  non-vanilla blind. Parity verified on 4 canonical seeds + ~500
  legal play actions per state. **12% full-trajectory speedup on
  AAAAAAA** (291s M4 baseline → 256s with wire-in). Per-call FFI
  cost amortized via cached `id(state.hand) → list[RustCard]`
  in `_state_value_cache_local` scope.
- ✅ Phase 2d (simple joker effects + Mult enhancement) — **33 joker
  effects ported** across four batches:
  - Batch 1 (14 jokers): Joker, Jolly/Zany/Mad/Crazy/Droll, Sly/Wily/
    Clever/Devious/Crafty, Greedy/Lusty/Wrathful/Gluttonous, Even
    Steven, Odd Todd, Scary Face, Half Joker.
  - Batch 2 (13 jokers): Fibonacci, Scholar, Smiley Face, Walkie
    Talkie, Onyx Agate, Arrowhead, Bull, Banner, Mystic Summit,
    Abstract Joker, Stuntman, Bootstraps, Gros Michel. Added xmult
    channel + HandContext for run-state-dependent effects.
  - **Architectural refactor**: split `simple_joker_effect` into
    `per_card_joker_effect` + `ability_joker_effect`. Mirrors Python's
    two-pass `_effect_adjustments` ordering — per-card xmult (Triboulet)
    applies BEFORE per-ability additive mult (Joker), not after. Caught
    by a fuzz that exposed the order-of-operations bug.
  - Batch 3 (7 jokers): The Duo/Trio/Family/Order/Tribe (ability xmult
    gated on hand shape), Acrobat (x3 on last hand), Flower Pot (x3 if
    scored has all 4 suits). Triboulet restored to native.
  - Batch 4 (3 jokers): Photograph (per-card x2 on FIRST scored face —
    needed `is_first_face` parameter on per-card path), Cavendish
    (flat x3), Seeing Double (x2 if scored has Club + non-Club). Blue
    Joker started but bailed (needs deck_size in HandContext).
  - **Batch 5: card editions** — Foil (+50 chips), Holographic (+10
    mult), Polychrome (x1.5 mult). Applied during per-card pass in the
    same order Python uses (chips → enhancement mult → edition mult →
    edition xmult → per-card jokers).
  - **Batch 6: joker editions** — same Foil/Holo/Polychrome but on the
    joker itself. Order per joker: edition chips → edition mult →
    joker effect → edition xmult. Required extending FFI to pass
    parallel `joker_editions` list, and removed the
    "any joker has edition → bail" check from `_score_action`. This
    blocker was hiding most mid-run jokered evaluations from Rust;
    removing it grew the trajectory speedup from ~12% to ~20%.
  - **Batch 7: held-card pass** — Steel enhancement (x1.5 mult per
    held Steel card), Shoot the Moon (+13 mult per held Q), Baron
    (x1.5 mult per held K), Raised Fist (+2*rank_value on lowest
    held), Blackboard (x3 mult if all held black), Mime (held effects
    retrigger). Required new `held_cards` parameter through FFI +
    held-card pass after ability pass. Plus a fast-skip
    (`held_pass_possible`) to avoid the loop when no held-card joker
    and no Steel is present.
  - **Batch 8: Card Sharp + Supernova + suit-debuff bosses** — Card
    Sharp (x3 mult if hand type repeated) and Supernova (+N mult per
    count of this hand type) added to Rust dispatch with new
    `played_count_this_hand_type` + `hand_type_played_before` ctx
    fields. Wire-in bails to Python until those fields are plumbed
    from `state.modifiers` (the synthetic test case works because
    initial state has counts=0). Suit-debuff bosses (The Club, The
    Goad, The Head, The Window) added to `_RUST_BLIND_SAFE` — they
    use vanilla math with one suit debuffed, which Rust already
    handles. This widens Rust activation to mid-run suit-debuff
    boss states.
  - **Batch 9: scaling jokers via metadata plumbing** — added
    `JokerMetadata { current_plus_mult, current_plus_chips, current_xmult }`
    parallel lists through FFI. Python wire-in calls Python's
    `_joker_current_plus` / `_joker_current_xmult` once at FFI
    conversion. Ports the 18 scaling jokers in one batch:
    - Additive mult (5): Ride the Bus, Green Joker, Spare Trousers,
      Castle (now chip-channel), Erosion
    - Additive chips (2): Square Joker, Runner
    - Xmult (11): Vampire, Hologram, Joker Stencil, Lucky Cat,
      Hit the Road, Steel Joker, Glass Joker, Madness, Throwback,
      Yorick, Campfire
    - Per-trigger bonuses honored where present (Green Joker +1,
      Ride the Bus +1 if no face, Spare Trousers +2 on two-pair,
      Square +4 on 4-card hand, Runner +15 on straight).
  - **Batch 10: activate ctx-dependent jokers** — added `deck_size`,
    `played_hand_types` ctx fields. Activates Blue Joker (2*deck_size
    fallback), Card Sharp (x3 mult if hand type repeated), Supernova
    (+N mult per count). Promoted Constellation and Ramen out of the
    bail list — both work via the existing `_joker_current_xmult`
    helper which handles their custom internal-xmult formula. Added
    identity-cached `_cached_joker_data` to amortize metadata extraction
    across the ~70K evaluate_simple calls per decision. Trade: ~10-15%
    trajectory slowdown vs batch 9 from added FFI args + cache lookups
    on the common no-Card-Sharp path. Net: still ~3-5% faster than
    M4 baseline at preserved quality.
  - **Batch 11: comprehensive joker coverage** — closed the gap on
    almost all single-joker effects. Added:
    - Splash (forces all cards to score via
      `scoring_indices_simple_with_splash`).
    - Retrigger jokers — Hack, Sock and Buskin, Dusk, Seltzer,
      Hanging Chad — handled by wrapping the per-card scoring body
      in a `for _ in 0..triggers` loop driven by
      `scored_card_trigger_count(card, is_first_scored, jokers,
      hands_remaining)`. Per-trigger chips/edition/enhancement/
      per-card-joker effects all fire correctly.
    - Loyalty Card and Driver's License — gated via new
      `JokerMetadata.loyalty_ready` / `drivers_active` bool flags,
      computed in Python wire-in via existing
      `_loyalty_card_ready` / `_drivers_license_active` helpers and
      threaded through FFI as parallel bool lists.
    - Additional scaling-mult jokers: Fortune Teller, Red Card,
      Flash Card, Ceremonial Dagger.
    - Additional scaling-chips joker: Stone Joker.
    - Additional scaling-xmult joker: Caino.
    - ~40 score-neutral jokers (state effects, money effects, RNG
      jokers with empty stochastic_outcomes) — Astronomer, Burglar,
      Cartomancer, DNA, Marble, Faceless, Golden Joker, Golden
      Ticket, Mail-In Rebate, Cloud 9, Rough Gem, Misprint, Business
      Card, 8 Ball, Rocket, Reserved Parking, etc. — all return
      `JokerEffect::default()` so they pass `is_supported_joker`
      without contributing to score.
    - Trajectory: AAAAAAA solver run drops from 244s → **220.6s**
      (10% faster than batch 10) with parity preserved.
    - 91 parity tests green (added Splash, Loyalty Card, Driver's
      License). 66 cargo tests green. 176 total Python-side Rust
      tests green.
  - **Still bailed**: Wee Joker (per-2-rank with retrigger semantics),
    Diet Cola / Burnt Joker / Popcorn / Ice Cream / Obelisk
    (need leading_plus metadata field or custom xmult formula),
    Blueprint / Brainstorm (copy other jokers), Ancient Joker /
    The Idol (target_suit/target_rank metadata), Swashbuckler /
    Baseball Card (multi-joker interactions), Four Fingers /
    Shortcut / Pareidolia / Smeared Joker (identification modifiers
    — handled at hand_type bail).
  - Parity verified on real first-blind states for every supported
    joker + edition combination, plus tricky ordering tests
    (Triboulet+Joker, Triboulet+Photograph, Duo+Tribe stacking,
    Polychrome Jolly Joker on pair, mixed-edition joker sets,
    Hack+Fibonacci retrigger interaction, Sock and Buskin+Scary
    Face per-trigger chips).

  - **Batch 12: scaling + multi-joker interactions** — added:
    - Ice Cream / Popcorn (decaying chips/mult with leading_plus
      fallback when current_plus == 0).
    - Obelisk (xmult scales when no other hand_type dominates;
      needs `played_count_max_other_hand_type` ctx + per-joker
      `obelisk_gain` from metadata).
    - Ancient Joker / The Idol (target_suit / target_rank per-card
      jokers — `has_target_suit` / `has_target_rank` flags +
      Suit/Rank u8 enum values stored on JokerMetadata).
    - Swashbuckler (Python wire-in precomputes sum-of-other-sell
      values into `current_plus_mult`; Rust just reads it).
    - Baseball Card (its x1.5 multiplier on uncommon other jokers
      is applied INSIDE evaluate_simple's ability loop, gated on
      `meta.rarity == 1`).
    - Diet Cola / Burnt Joker (score-neutral additions).
    - JokerMetadata struct grew to include 8 new fields:
      `leading_plus_mult`, `leading_plus_chips`, `sell_value`,
      `rarity`, `has_target_suit`/`target_suit`,
      `has_target_rank`/`target_rank`, `obelisk_gain`.
    - FFI surface grew by 7 new optional parameter lists; Python
      wire-in's `_cached_joker_data` returns a 14-tuple now.
  - **Batch 15: Pareidolia** — face-card identification modifier.
    Added `ctx.pareidolia_active`; every face-detection check
    (`is_face_with_pareidolia`) is now joker-aware. Affects
    Photograph (first scored card becomes the "first face"),
    Smiley Face / Scary Face (fire on every scored card),
    Sock and Buskin (retriggers every scored card), Ride the Bus
    (never fires because every card is a face).
  - **Batch 16: Wee Joker** — handled inline in evaluate_simple
    (its chip contribution needs `total_triggers_on_2s` which is
    only visible in the outer evaluator). Precomputes the retrigger
    sum over scored 2s, then adds `current_plus_chips + 8 * total`
    to chips during the ability loop. Skipped when no Wee Joker is
    held.
  - **Batch 17: Stone card support** — `identify_hand_type_simple`
    now filters out Stone cards (they have no rank/suit for shape
    detection) and uses the remaining `ranked` subset for all
    counts/flush/straight checks. `scoring_indices_simple` merges
    stone-card indices into the regular scoring set so they always
    score (Python's `_with_stone_indices`). `card_chip_value` was
    already Stone-aware (returns 50). Net: Stone cards no longer
    bail the Rust fast path.
  - **Batch 18: Smeared / Four Fingers / Shortcut** —
    `identify_hand_type_with_jokers` + `scoring_indices_with_jokers`
    handle color-merged flushes, 4-card flush/straight, and gap-1
    straights. Picks the right 4 or 5 cards from the played hand
    via subset enumeration (≤5 combinations in the hot path).
    Try-5-then-4 fallback mirrors Python's
    `_five_or_four_fingers_indices`.
  - **Batch 19: Wild card support** — Wild cards count for every
    suit-key in flush detection; `card_chip_value` returns 0 when
    any suit is debuffed (Wild could be the debuffed suit). Routes
    Wild-containing hands through the joker-aware path even when
    no id-jokers are held.
  - **Batch 20: Blueprint / Brainstorm copy-effect** — implemented
    via Python-side resolution at the wire-in. `_cached_joker_data`
    calls `_effective_ability_joker_indices(jokers)` from Python,
    then substitutes the COPIED joker's name + scaling metadata
    into each slot. Editions / rarity / sell_value stay PHYSICAL
    (matches Python's `joker_rarity(physical_joker)` for Baseball
    Card). Rust just sees the resolved joker list and applies
    effects normally. Unresolved copies (no compatible target)
    score zero. Per-card jokers (Triboulet etc.) also fire from
    the copy because the Rust loop iterates the resolved names.
  - Trajectory: AAAAAAA solver run **236.0s** at batch 20 (vs
    220.6s baseline at batch 11). The modest regression is the
    extra FFI + metadata extraction cost from the larger
    JokerMetadata struct + identification-aware paths; pays back
    as essentially every base-game joker now stays in the Rust
    fast path. 117 parity tests + 69 cargo tests + 202 Python-side
    Rust tests all green.
  - **No-bail-list**: Phase 2 joker effect coverage is now
    effectively complete. The only joker-related bails remaining
    are jokers with metadata-driven scaling that's hard to
    precompute (Obelisk's `obelisk_gain` does precompute), or
    jokers we haven't categorized yet. Real solver runs should
    activate the Rust fast path on essentially every play.

Total Rust-binding tests: **202 green**. Rust-internal tests: **69 green** (`cargo test`).

**Phase 3 (forward simulation) — STARTED:**
- ✅ **Phase 3a: scaffolding + `_draw_from_deck`** — created
  `botlatro-core/src/forward_sim/{mod.rs,deck.rs}`. Ported the
  `_draw_from_deck` helper from `forward_sim.py:927`:
  - Exact-known-deck path: drawn cards MUST be in known_deck;
    return Err on missing (mirrors Python `ValueError`).
  - Partial-known-deck path: silently skip missing draws, size
    decrements regardless.
  - Match predicate is canonical 5-field (rank/suit/enhancement/
    edition/seal) — first match wins.
  - PyO3 wrapper exported as `balatro_core.draw_from_deck`.
  - 6 cargo tests + 6 Python-side parity tests (including a
    50-iteration random fuzz comparing Rust vs Python on synthetic
    inputs). All green.
- ✅ **Phase 3b: `_jokers_after_play`** — ported the ~12
  scaling-counter updates into `forward_sim/jokers.rs`. Pure-delta
  jokers (Green, Square, Runner, Spare Trousers, Wee, Vampire,
  Lucky Cat), reset-or-delta (Ride the Bus, Loyalty Card, Obelisk),
  decay-with-removal (Ice Cream, Seltzer). Returns 5 parallel
  Option-lists + remove flags; Python applies non-None values to
  joker metadata. Wee Joker's scored-2-trigger sum is computed
  inline using the existing `scored_card_trigger_count`.
- ✅ **Phase 3c: `_jokers_after_discard`** — ported Ramen (xmult
  decay, removal at <=1), Green Joker (mult-1 floor 0), Hit the
  Road (xmult + 0.5 per Jack), Castle (chips + 3 per target-suit
  discard), Yorick (per-discard countdown with reset + xmult+1).
- ✅ **Phase 3d: phase transition** — new
  `forward_sim/phase.rs::next_phase(required, next_score,
  next_hands, has_mr_bones)`. Returns one of KeepPlaying /
  RoundEval / MrBonesSave / RunOver. Mr Bones gate: score >= 25%
  of required AND a non-disabled Mr. Bones joker.
- ✅ **Phase 3e: end-of-round economy (held-card money)** —
  `forward_sim/economy.rs::held_end_of_round_money_delta`.
  `3 × (1 + mime_count) × gold-card-count`. Remaining economy
  helpers (gift card, blue seal, cash-out interest, joker round-end
  housekeeping) are larger and deferred to Phase 3f if needed.
- ✅ **Phase 3f: discard money + hand-level upgrades** — ported
  `_discard_money_delta` (Trading Card +$3 first-discard,
  Faceless +$5 on 3-face, Mail-In Rebate +$5/match) and
  `hand_level_after_play` (Space Joker +1 / The Arm -1 with
  min-1 clamp).
- ✅ **Phase 3g: wire-in attempt + decision** — wired Rust
  `draw_indices_to_remove` into Python's `_draw_from_deck` (with
  indices-to-remove pattern to preserve Card.metadata, which
  RustCard drops). Microbenchmark showed 1.29× per-call speedup.
  **But trajectory measurement was a net regression**: 3 runs
  consistently landed at 242–244s vs 236s baseline. The FFI cost
  of converting state.known_deck (~40 cards) into a fresh RustCard
  list every call outweighs the algorithm win. Reverted the
  wire-in with a comment explaining why. The Rust function stays
  available for use by a future native simulate_play (Phase 4)
  that can amortize conversion across the whole transition.
- ✅ **Phase 3k: simulate_play_native (simple-case fast path)** —
  Top-level `simulate_play_simple` that composes scoring + joker
  scaling updates + deck draw + phase transition + held-end money
  into ONE FFI call. Bails to Python on:
  - Special blinds (The Hook, The Ox, The Manacle, The Wheel,
    The Psychic, etc. — only Small / Big / Boss + suit-debuff
    bosses supported)
  - "Complex" jokers (Midas Mask, Vampire, Hiker, DNA, Sixth Sense,
    Hallucination, Gift Card, Mr. Bones, Crimson Heart, Trousers,
    Splash, Glass Joker, Madness)
  - Stochastic outcomes (always-empty in solver but still bails to
    be safe)
  - Blue seals on held cards
  - Glass enhancement on any played/held card
  - Any unsupported joker (per `is_supported_joker`)
  - End-of-round transitions (ROUND_EVAL / RUN_OVER fall back
    to Python so end-of-round housekeeping is handled correctly)
  Wire-in: `_try_rust_simulate_play(state, action, drawn)` at top
  of `simulate_play`; returns next-state via dataclass.replace
  with applied deltas, else falls through. **Hit rate 25.7%**
  on AAAAAAA trajectory (2958 fast-path of 11516 calls).
  **Per-call benchmark**: 107µs Rust vs 121µs Python (1.13×).
  **Trajectory wall-time**: 237s vs 236s baseline — savings (~40ms
  per trajectory) lost in noise. Parity preserved (130 steps,
  identical termination).

**Phase 3 status: COMPLETE as scaffolding.** All five forward_sim
helpers (draw, jokers-after-play, jokers-after-discard,
next_phase, held_end_money) ported + a top-level
`simulate_play_simple` that orchestrates them. The Python wire-in
demonstrably works (25.7% hit rate, parity-preserved) but the
trajectory speedup is within noise (~0.4%).

**Architectural conclusion**: the ≥3× speedup goal in Phase 3's
acceptance gate is unreachable from Python-driven simulate_play.
The per-call FFI conversion (~40 cards in known_deck + ~5-10
jokers + their metadata) eats all the algorithm win. To get the
projected speedup, the SEARCH itself needs to live in Rust —
state stays Rust-native, conversion happens once per search, and
internal helpers compose without FFI. That's Phase 4 territory.

Total Rust-binding tests: **226 green**. Rust-internal tests: **97 green** (`cargo test`).

**Phase 4 (native solver search) — STARTED:**
- ✅ **Phase 4a: batched action scorer + legacy beam wire-in** —
  `botlatro-core/src/search/scorer.rs::score_play_actions_batch`
  takes shared inputs (hand, jokers, hand_levels) ONCE and scores
  N candidate plays in one FFI call. Microbench shows **16.58×**
  per-state speedup vs the Python per-action loop.
  Wired into legacy beam's `_cheap_beam_play_scores` (hand_search.py).
  Restricted to `_RUST_BLIND_SAFE` blinds — boss blinds with
  scoring effects (Flint, Arm, Tooth, Psychic, Eye, Mouth, Plant)
  fall back to Python per-action. Boss-adjusted blinds (Eye / Mouth)
  re-score Python-side to pick up `_boss_adjusted_score` correctness.
  **Trajectory** (3-run avg): **223.9s vs 236s baseline → ~5% speedup**,
  parity preserved (130 steps, RUN_OVER reason identical across all
  runs). The per-state microbench (16.58×) doesn't translate to a
  proportional trajectory win because `_cheap_beam_play_scores`
  isn't the dominant cost — leaf evaluation (rollout-based
  `planning_value`) is, and it was already optimized via the
  Phase 2 evaluate_simple wire-in. This is still the first real
  Phase 4 win where FFI amortization shows up at trajectory scale.
- ✅ **Phase 4b: rollout batched scorer wire-in** — wired the
  same `score_play_actions_batch` into the two hot loops inside
  greedy rollouts: `_best_greedy_play_action_uncached` and
  `_best_immediate_score_uncached` in state_value.py. These are
  the dominant cost in `planning_value` (called once per rollout
  step × N samples per leaf × many leaves per beam). Per-step
  cost drops from 218 Python `_score_action` calls to one Rust
  batched call.
  **Trajectory: 134.8s vs 236s baseline → 43% speedup**, parity
  preserved (130 steps, RUN_OVER). This is the biggest win of
  the entire port so far — the rollout loop runs thousands of
  times per decision, and pushing 218×~thousands action scorings
  into batched Rust calls eliminates a huge chunk of Python
  per-call overhead.
- ✅ **Phase 4c: `best_play_action_native`** — combined
  enumerate-all-plays + score-each + argmax into ONE FFI call.
  Saves the Python action-enumeration tuple allocations AND the
  Python argmax loop on top of Phase 4b's scoring batch.
  `botlatro-core/src/search/scorer.rs::py_best_play_action`
  iterates `combinations(1..=5)` of hand indices internally,
  calls evaluate_simple per candidate, returns `(best_indices,
  best_score)`. Wired into `_best_greedy_play_action_uncached`
  with fallback to Phase 4b's batched scorer when this bails.
  **Trajectory: 129.9s vs 134.8s after 4b → small ~3.6%
  additional speedup**, parity preserved. The win is small
  because the enumeration loop was already cheap relative to
  the scoring; this just removes the last per-action Python
  bookkeeping.
- ✅ **Phase 4e: shop-search `_sample_hand_build_score` wire-in** —
  cProfile of a full trajectory revealed shop_search was ~50% of
  trajectory time, dominated by `_sample_hand_build_score` calling
  Python `evaluate_played_cards` ~500K times for build valuation.
  Added `_try_rust_sample_score` in
  `basic_strategy/build_scoring.py` that calls
  `balatro_core.evaluate_simple_with_levels` directly when the
  blind is safe + jokers are supported.
  **Trajectory: 118.4s vs 129.9s after 4c → 9% additional speedup.**
- ✅ **Phase 4f: joker-data extraction cache** — `_sample_build_score`
  calls `_sample_hand_build_score` ~20-30 times per shop decision
  with the SAME jokers tuple, each previously re-extracting the
  14-list of joker fields. Added a tuple-ref-keyed cache (with
  id() collision protection via the tuple-ref check that prevents
  garbage-collected ids from returning stale data — initial bare-id
  version caused a 130→165 step trajectory divergence). Bounded to
  256 entries.
  **Trajectory: 108.5s vs 118s after 4e → 8% additional speedup.**
  Cumulative speedup vs original 236s baseline: **2.18× (54%
  reduction)**.
- ✅ **Phase 4g: centralized rust_bridge + score_projection +
  play_scoring wire-ins** — factored the shared Rust scaffolding
  (joker-data cache + `rust_evaluate_score` + `rust_evaluate_score_
  and_hand_type`) into `src/balatro_ai/search/rust_bridge.py` to
  avoid module-local duplication. Wired into:
  - `basic_strategy/score_projection.py::_score_selected_cards`
    (boss-adjusted: returns score + hand_type, Python applies the
    Eye / Mouth adjustment when needed)
  - `basic_strategy/play_scoring.py::_score_play_action` (same
    pattern; covers any direct callers not already covered by the
    batched scorer)
  - Refactored `build_scoring.py::_try_rust_sample_score` to use
    the shared bridge instead of its own copy.
  **Trajectory: 109.7s — essentially unchanged** (within noise of
  108s baseline). The wire-ins are correct + parity-preserving but
  the call sites aren't on the hot path (`_cheap_beam_play_scores`
  + the rollout already cover the volume). The win here is
  **maintenance**: one canonical rust_bridge module for all future
  wire-ins.
- ✅ **Phase 4d.1+: widened rollout blind-safe set** —
  instrumenting `_try_rust_clear_probability` showed 100% of bails
  were boss blinds (Wheel 330, Manacle 175, Psychic 105, Hook 76),
  ZERO joker bails — clear_probability_native handles the joker
  surface well. Added a wider `_RUST_ROLLOUT_BLIND_SAFE` set
  (specifically for the rollout estimator) that includes Manacle
  (just hand-size, score math unchanged), Wheel (1-in-7 face-down
  per card — rollout slightly overestimates by ignoring), and
  Psychic (must-play-5 — rollout's argmax usually picks 5-card
  hands anyway). Hook still bails because mid-play discards
  meaningfully shift scoring. Hit rate jumped 78.6% → ~98%.
  **Trajectory: 94.4s vs 100.5s after 4d.1 → 6% additional
  speedup**, parity preserved (130 steps, RUN_OVER).
- ✅ **Phase 4d.1: native `clear_probability_native`** — the first
  Phase 4d architectural piece. The greedy-rollout loop now runs
  ENTIRELY in Rust:
  `botlatro-core/src/search/rollout.rs::clear_probability_native`
  takes the state inputs once, runs N rollouts internally, returns
  the clear fraction.
  Each rollout step: enumerate combinations + score each + argmax
  (inline best_play_action), sample drawn cards (xoshiro256** RNG
  seeded from the caller's seed), apply simulate_play deltas
  (scoring + joker scaling + deck mutation + hand update).
  Internal RolloutState carries the mutating fields between steps,
  so the entire `while !terminal: simulate; check_terminal` loop
  stays in Rust — no FFI between steps.
  Bails on jokers that aren't in `is_supported_joker`, jokers in
  `SIMPLE_BAIL_JOKERS` (Vampire, Midas Mask, etc.), Glass cards,
  and blue seals. Internal xoshiro RNG differs from Python's
  `random.Random` so drawn-card sequences vary, but
  clear_probability is an ESTIMATOR — the bot's decisions remain
  quality-equivalent. AAAAAAA trajectory preserves 130 steps and
  RUN_OVER reason, so parity holds in practice.
  **Trajectory: 100.5s vs 108s after 4g → 7% additional speedup,
  parity preserved (130 steps).** Cumulative speedup vs baseline:
  **2.35× (57.4% reduction)**.
- ⚠️ **Phase 4d.2 attempted TWICE, REVERTED both times.**
  Built `botlatro-core/src/search/beam.rs::beam_play_value_native`
  — full recursive beam in Rust with internal top-K + simulate +
  recurse + leaf-rollout machinery (~800 lines).
  - **Attempt 1 (minimal)**: plays only, no discards in recursion,
    4 rollout samples. 52.9s wall-time, but trajectory crashed at
    33 steps (vs 130 baseline). Bot too aggressive — skipped
    setup discards.
  - **Attempt 2 (proper)**: ported discard simulation + discard
    ranking heuristic (`discard_candidate_rank` matching Python's
    `_cheap_beam_discard_rank` + `_discard_candidate_score`), used
    Python's play-rank formula (clear bonus 100K, pace bonus
    10K/needed) in the mixed candidate sort, bumped rollout
    samples to 16. Still failed parity: 51-52s wall-time, 33-34
    steps. The bot's value estimates still diverge enough to
    drive quality-eroding decisions.
  - Both reverted. `beam.rs` stays in the tree as scaffolding
    for a future attempt that does careful side-by-side
    instrumentation to find the exact divergence point.
  - **Likely remaining issues**: apply_play's joker metadata +
    modifier updates may differ subtly from Python's full
    simulate_play; xoshiro RNG draws differ from Python's
    random.Random, causing rollout outcomes the beam treats as
    ground truth at each branch to differ; the mixed
    play/discard ranking might not perfectly match Python's
    `_beam_future_actions` logic.
  - **Conclusion**: native beam recursion is the right
    architectural step but requires a careful research-grade
    port matching every quirk of Python's `_beam_action_value`
    + `_beam_plan_value` + `_state_after_beam_action`. That's
    multi-session work, not a one-session push.

- ✅ **Phase 4h: wire `_evaluate_play_action` to Rust** —
  the biggest remaining Python evaluator caller (193K direct
  calls per trajectory). Synthesizes a minimal `HandEvaluation`
  with `score_override` + `hand_type` populated; `scoring_indices`
  is empty (only blind_setup uses it, in a non-quality-critical
  annotation path). Trajectory 91s → 86s.
- ✅ **Phase 4i: widen `RUST_BLIND_SAFE`** — instrumenting
  `rust_evaluate_score_and_hand_type` showed 94% bail rate
  driven by boss blinds (Wheel 137K, Manacle 84K, Hook 72K,
  Psychic 71K) the bridge wouldn't dispatch on. Audited each
  for true evaluator impact:
  - **The Wheel**: 1-in-7 face-down per card is forward_sim, not
    evaluator. Python `evaluate_played_cards` doesn't apply it.
    Safe to add.
  - **The Manacle**: -1 hand size affects action enumeration, not
    scoring. Safe to add.
  - **The Hook / The Psychic**: DO affect evaluator scoring
    (Hook discards before scoring; Psychic zeros !=5-card plays).
    Stay bailed.
  - Trajectory: 86s → 78.4s, parity preserved (130 steps).

- ✅ **Phase 4i+: extend RUST_BLIND_SAFE to The Hook** — audit
  of Python evaluator showed Hook is forward_sim-only (the
  hook-discarded held cards are removed before `evaluate_played_cards`
  is even called). Added to safe set. Trajectory 78s → 73s.
- ✅ **Phase 4j: activate `decision_cache_scope`** in
  `SolverPolicy.choose_action`. The basic_strategy bot already
  did this, but the solver path didn't — meaning every
  `_identity_cached_value` call fell through to its factory,
  causing `_freeze_for_cache` to run 11M times per trajectory.
  One-line `with decision_cache_scope():` wrap dedupes everything.
  **Trajectory 73s → 49.4s — 33% additional speedup, parity
  preserved (130 steps).** This wasn't a Rust port — just
  recognizing that an existing Python cache wasn't activated
  on the solver path.

**Phase 4 cumulative status (final)**: starting baseline 236s →
**49.4s (79% speedup, 4.78×)** with parity preserved. **The
original Phase 4 acceptance gate (≥3× speedup) is MET 1.6×
over.** Got here through surgical wire-ins + one critical cache
activation, without the architectural native-beam rewrite
(which still failed parity even on the second attempt). The
Rust scaffolding for the native beam stays in tree (`beam.rs`)
for future work toward chess-engine-style deep search.

**Profile snapshot after 4f** (cProfile, cumulative seconds):
- `best_blind_beam_action` (play search): 204s (the play beam
  recursion + leaf rollouts; rollouts now mostly Rust)
- `shop_search` (`best_shop_action` → `_expand_action`): 104s
  (halved from 200s before 4e/4f)
- `evaluate_played_cards` (Python): 141s (down from 233s; ~half
  the calls now go through Rust)
- `_score_action`: 75s (Rust fast path active)
- `planning_value` (rollouts): 91s

The remaining big lever is moving the beam recursion + leaf
evaluation themselves into Rust, which is Phase 4d's scope.

**Phase 1 (state representation) — COMPLETE:**
- ✅ `RustCard` — packed 6-8 byte struct with typed Rank/Suit/Enhancement/Edition/Seal enums
- ✅ `RustJoker` — name+edition+sell_value (JokerId enum deferred to Phase 2)
- ✅ `RustGameState` — full GameState mirror with hand/known_deck/jokers/consumables/vouchers + scalar run state + opaque modifiers/hand_levels dicts
- ✅ `from_python` / `to_python` round-trip preserves all solver-relevant fields
- ✅ 40 parity tests green across the 4 canonical seeds (12 PoC + 15 Card + 7 Joker + 6 GameState)
- ✅ 10 Rust-side unit tests green (`cargo test`)
- ✅ Card size invariant: ≤ 8 bytes (vs Python dataclass ~200 bytes)

**Known divergences (documented):**
- Card metadata not preserved across round-trip (dropped on Rust side)
- Joker metadata + derived effect not preserved
- Card rank "T" normalizes to "10" on round-trip (both functionally equivalent per RANK_VALUES)
- `Py<PyDict>` for modifiers/hand_levels means `RustGameState` is not `Clone` (a manual Clone needing GIL token is the right impl when we need it)

---

## 3. Critical FFI overhead finding (PoC, 2026-05-26)

The PoC ported `_is_stone_card` (the simplest possible helper, single
attribute lookup + string normalization). Timing on 100,000 calls:

- Python `_is_stone_card`: **8.3ms**
- Rust `is_stone_card` (via PyO3): **29.8ms**
- **Rust is 3.6× SLOWER** than Python for this function.

Cause: the FFI boundary crossing (Python object → PyO3 attribute
extraction → Rust function call → return) costs more than the work
itself. The Python version is `@lru_cache(maxsize=32)` so most calls
are a single dict lookup.

**Implications for port strategy:**

❌ **DON'T port small helpers individually.** The FFI cost dominates
their tiny per-call work. `_is_stone_card`, `_card_chip_value`,
`_enhancement_chips`, etc. all fall in this category.

✅ **DO port entire subsystems as single FFI calls.** The FFI cost
amortizes when one Rust call does hundreds-thousands of operations
internally. The right port targets are:
- `evaluate_played_cards` (whole-hand scoring, ~100 ops internally)
- `_best_greedy_play_action` (enumerates 218 plays, calls evaluate × 218)
- `_clear_probability` (runs N rollouts, calls best_greedy × ~7 per rollout)
- `forward_sim.simulate_play` / `simulate_discard` (full transition)

Even better: `solver_beam_play_action` — one FFI call, one Action
return, all the work in Rust. **That's the goal architecture.**

**Implication for state representation:** to make the per-FFI-call
batch big enough, the entire GameState must transit the boundary
ONCE per call, then live in Rust for the duration. So we need either:
- A `RustGameState` that wraps a Rust struct (PyO3 class)
- Or a serialization step at the boundary (slower; not preferred)

The `RustGameState` approach is the right one. Python constructs it
from a GameState once at the search entry point; all subsequent
solver work operates on it natively.

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Python: balatro_ai.*                                             │
│ ─────────────────────                                            │
│ - bots/, eval/, dataset/, solver/policy.py, solver/trajectory.py │
│ - api/state.py (GameState dataclass)                             │
│ - rules/hand_evaluator.py (Python fallback)                      │
│                                                                  │
│ Entry points to Rust core:                                       │
│   from balatro_core import (                                     │
│       GameStateNative,                                           │
│       evaluate_played_cards_native,  # large-batch FFI           │
│       solver_beam_play_action_native,                            │
│   )                                                              │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ PyO3 FFI (boundary)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ Rust: balatro_core (botlatro-core/)                              │
│ ──────────────────────────────────                               │
│ Module organization:                                             │
│   src/                                                           │
│     lib.rs              <- PyO3 module + function registrations  │
│     state/                                                       │
│       game_state.rs     <- Packed GameState, ~256 bytes          │
│       card.rs           <- Card = 1 byte                         │
│       joker.rs          <- Joker = 2-4 bytes + effect table      │
│     hand_eval/                                                   │
│       mod.rs            <- evaluate_played_cards entry           │
│       hand_type.rs      <- _identify_hand_type port              │
│       scoring.rs        <- _scoring_indices + chip calc          │
│       effects.rs        <- _effect_adjustments (joker dispatch)  │
│     forward_sim/                                                 │
│       mod.rs            <- simulate_play / simulate_discard      │
│       deck.rs           <- DeckModel, draw sampling              │
│     rng/                                                         │
│       pseudohash.rs     <- Balatro's pseudohash + pseudoseed     │
│       luajit_tw223.rs   <- LuaJIT PRNG port                      │
│     search/                                                      │
│       beam.rs           <- beam search in native                 │
│       bnb.rs            <- branch-and-bound                      │
│   tests/                                                         │
│     parity/             <- run alongside Python on audit corpus  │
└──────────────────────────────────────────────────────────────────┘
```

The Python side keeps the high-level orchestration (bots, dataset,
NN training pipeline). The Rust side owns the hot path: state
representation, evaluation, simulation, search.

---

## 5. Porting sequence

Each phase has a **measurement gate**: ship + measure on the
4-seed canonical batch (AAAAAAA, BBBBBBB, CCCCCCC, 1234567)
before starting the next phase. If a phase doesn't deliver the
projected speedup, investigate before adding more code.

### Phase 1: State + Card representation (Week 1)

**Goal:** packed `Card` + `Joker` + `GameState` in Rust, with PyO3
constructors from the Python equivalents. No evaluation logic yet —
just data layout.

**Files:** `state/card.rs`, `state/joker.rs`, `state/game_state.rs`

**Card layout:**
```rust
#[repr(u8)]
pub enum Rank {
    Two = 0, Three = 1, ..., Ace = 12,  // 4 bits
}
#[repr(u8)]
pub enum Suit { Clubs = 0, Diamonds = 1, Hearts = 2, Spades = 3 }  // 2 bits
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
    pub enhancement: Enhancement,  // u8 enum, 8 variants
    pub edition: Edition,           // u8 enum, 8 variants
    pub seal: Seal,                 // u8 enum, 5 variants
    pub debuffed: bool,
}  // ~6 bytes, alignable to 8
```

**Joker layout:**
```rust
pub struct Joker {
    pub key: JokerId,         // u16, enum of ~150 joker keys
    pub edition: Edition,
    pub counter: i32,         // accumulated mult/chip/etc.
    pub debuffed: bool,
}  // ~8 bytes
```

**GameState layout:**
```rust
pub struct GameStateNative {
    pub hand: SmallVec<[Card; 16]>,
    pub deck: SmallVec<[Card; 64]>,
    pub jokers: SmallVec<[Joker; 8]>,
    pub consumables: SmallVec<[ConsumableId; 4]>,
    pub vouchers: u32,           // bitmask of owned vouchers
    pub hand_levels: [u8; 12],   // per-HandType level
    pub modifiers: ModifierFlags, // bitmask + ~16 known counters
    pub ante: u8,
    pub blind_id: BlindId,        // enum
    pub required_score: i32,
    pub current_score: i32,
    pub hands_remaining: u8,
    pub discards_remaining: u8,
    pub money: i32,
    pub phase: Phase,
}  // ~250 bytes
```

**PyO3 surface:**
```python
state = GameStateNative.from_python(game_state)  # build from Python GameState
print(state.score, state.hand_size)              # cheap accessors
new_state = state.with_played(cards_indices)     # immutable transitions
```

**Acceptance gate:** GameStateNative.from_python(gs) preserves all
fields; round-trip back to Python produces equal GameState. No
speedup measurement yet — this is plumbing.

### Phase 2: Hand evaluation (Week 2)

**Goal:** `evaluate_played_cards_native(state, action_indices)`
returns the same score tuple as Python on every audit transition.

**Files:** `hand_eval/mod.rs`, `hand_eval/hand_type.rs`,
`hand_eval/scoring.rs`, `hand_eval/effects.rs`

**Scope:** start with the no-joker scoring path (HandType
identification + per-card chips + level-based mult). Then add the
top-20 most common jokers (Mult Joker, Half Joker, etc. — measured
from M4 trajectory frequency). Less common jokers fall back to
Python via a sentinel value.

**Acceptance gate:** parity on 5074 audit transitions for the
top-20 joker subset. Measurable speedup ≥10× on
`evaluate_played_cards` standalone benchmark.

### Phase 3: Forward sim + deck model (Week 3)

**Goal:** `simulate_play_native(state, action_indices, drawn_cards)`
returns the post-state, identical to Python's `simulate_play`.

**Files:** `forward_sim/mod.rs`, `forward_sim/deck.rs`

**Scope:** play/discard transitions only (not shop yet). The
joker-side effects (Crimson Heart disable, Bloodstone triggers,
etc.) come along for the ride from Phase 2's effect table.

**Acceptance gate:** trajectory through `generate_trajectory` works
end-to-end with the Rust forward_sim (Python solver, Rust core).
≥3× speedup on full trajectory wall-time.

### Phase 4: Solver search in native (Week 4-5)

**Goal:** `solver_beam_play_action_native(state, depth, width)`
runs the entire search in Rust. Python's only job is decision
collection.

**Files:** `search/beam.rs`, `search/bnb.rs`

**Scope:** start with the M4-equivalent beam (depth-limited
minimax with per-ply candidate cap). Once at parity, add B&B with
proper admissible upper bounds (now affordable because state eval
is fast).

**Acceptance gate:**
- Quality: ante ≥ legacy M4 baseline (4.5 avg on 4-seed batch)
- Speed: ≤30s per seed serial at d3w2 (vs M4's 130s)
- Combined: ≥4× full-trajectory speedup over legacy

### Phase 5: Optimization + dataset run (Week 6)

**Goal:** profile-driven Rust optimization (SIMD where applicable,
allocator tuning, transposition table for solver reuse) + first
1000-seed dataset run with the new core.

**Acceptance gate:**
- 1000 seeds, multi-archetype, depth=5 → ≤4 hours on 8 cores
- Quality: ≥30% of seeds improve on M4 baseline (+1 ante or 2× score)

---

## 6. Risks + mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| FFI overhead dominates for fine-grained calls | HIGH (already seen) | Port at coarse granularity (whole evaluate / whole simulate); never port single helpers |
| Joker effect coverage drift (150+ jokers, complex effects) | HIGH | Port top-20 first, fall back to Python for the rest. Parity test catches any divergence. Add more over time. |
| GameStateNative ↔ GameState conversion overhead | MED | Construct ONCE per search; pass by reference inside Rust |
| Windows toolchain issues (linker, ABI) | LOW (validated) | PoC already builds + imports + runs |
| Maturin develop vs pip install discrepancy | LOW (resolved) | `pip install ./botlatro-core` works reliably; `maturin develop` had install-location issues, avoid |
| Rust + Python skill gap | MED | PyO3 has good docs; start simple and grow |
| Build time creep as crate grows | LOW | Cargo's incremental compilation handles this well |

---

## 7. What we keep, what we delete

### Keep (no change)
- All Python bots, dataset CLI, eval scripts, trajectory recorder
- `api/state.py` — Python GameState stays as the public API; Rust
  uses it only at the FFI boundary
- `rules/hand_evaluator.py` — stays as fallback + parity-test
  ground truth. Rust calls into it for unported joker effects.
- `solver/` policies — they orchestrate at the Python level; the
  Rust call is just one of their tools

### Replace
- `solver/search_v2/play.py` — becomes a thin wrapper around
  `balatro_core.solver_beam_play_action_native`
- `search/forward_sim.py` — becomes a thin wrapper around
  `balatro_core.simulate_play_native` / `simulate_discard_native`

### Add
- `botlatro-core/` — entire crate (this doc's subject)
- `tests/test_rust_*_parity.py` — per-port parity test modules
- `src/balatro_ai/core/` — Python-side helpers for Rust interop
  (Card↔RustCard converters, etc.) if needed

### Sunset
- Cython `setup.py` extension entry for `hand_evaluator_native.pyx`
  — superseded by Rust. The Cython infrastructure stays
  (`pyproject.toml` build-system entries) in case we want a
  fallback, but no new Cython work happens.

---

## 8. Definition of done

The port is complete when:

- **Speed:** 1000 seeds, multi-archetype, depth=5 → ≤4 hours on
  8 cores (currently: ~50 hours estimated with legacy).
- **Quality:** ≥30% of seeds in a 100-seed test improve on the
  legacy M4 baseline by either +1 ante or 2× score.
- **Reproducibility:** the dataset CLI run twice on the same
  seed file produces byte-identical JSONL (modulo wall-time floats).
- **Maintainability:** parity tests green; new contributor can
  `pip install -e ./botlatro-core` and see all tests green.
- **Documentation:** this plan stays current — each completed
  phase gets a status note; lessons go in a §9 "Lessons learned"
  appendix.

If we hit those four, we have a solver that can plausibly produce
the Phase 8 training dataset at AlphaZero-style quality + speed.

---

## 9. Lessons learned (appended as phases land)

### 2026-05-26 — PoC FFI overhead

Ported `_is_stone_card` as the smallest possible PoC. Parity green
on 12 test cases, but Rust was 3.6× SLOWER than the lru-cached
Python version. The FFI boundary cost dominated the tiny per-call
work. **Reshaped the port strategy** (§3) to target large
batched FFI calls instead of fine-grained helpers — this is the
single most important architectural choice for the port. The
`_is_stone_card` Rust function stays in the codebase as a building
block for the larger Rust-internal evaluation pipeline (where it
won't cross the FFI boundary — it'll just be a `#[inline]` Rust
function called from `evaluate_played_cards_native`), but it's
NOT exposed at the Python boundary.

### 2026-05-28 — id()-memo nondeterminism + the "parity preserved" claims were unreliable

Every prior "parity preserved (130 steps on AAAAAAA)" note in this
document was measured under a **nondeterminism bug**: the v2 beam's
leaf-value memo in `solver/search_v2/play.py` was keyed by
`(id(state), depth)` and stored a bare float — no tuple-ref guard.
Transient child `GameState`s get GC'd mid-search, CPython recycles
their addresses, and new states collided with stale memo entries,
reading another state's value. Result: the solver returned different
trajectories run-to-run (130 steps was one random sample), and
**dataset generation was silently corrupted** whenever a worker
processed more than one seed (heap state leaked across trajectories).

Fixed with the same tuple-ref guard used everywhere else in
`state_value.py` / `rust_bridge.py` (store `(state, value)`, check
`cached[0] is state`). Post-fix: solver is deterministic; **Rust-on
vs Rust-off match step-for-step (40/40 steps, 0 score mismatches on
AAAAAAA)** — confirming the Rust port itself is correct and the
prior "divergences" were this bug, not Rust.

**Validation findings (post-fix, 20-seed deterministic batch):**
- v2 ≈ legacy ≈ ante ~3.1 / ~5% win. The documented "legacy 4.5 vs
  v2 2.0" was a 4-seed measurement taken under the bug.
- **Depth is a dead end:** d3≈d4≈d5≈ante 3.1. Per-decision cost
  ~2.3× from d3→d5, then flat — the beam's one-blind horizon caps
  effective depth at ~5 (branches hit blind-clear/bust before the
  depth cap). "Chess-like depth 12" is undefined in this
  architecture; true lookahead would need cross-blind/shop search.
- **Leaf reweighting is a dead end:** scaling the headroom term is
  monotonic, so it never changes the argmax. The value function is
  clear-probability-dominated and economy/build-blind (post-blind
  money moves the leaf ~0; cross-blind joker scaling isn't valued;
  tarot/consumable use isn't even a search action). This — not
  search depth — is the ante-~3 ceiling, and it's why the project
  pivoted to a learned value (Phase 8).

**Remaining Rust-port opportunities (profile of one trajectory):**
- **Shop search is still ~50% of trajectory time and almost entirely
  Python** (`reroll_ev`, `basic_strategy_shop_item_value`,
  `shop_leaf_terms`, `shop_sampler._record_available` / `_normalized`).
  Largest unported hot path.
- **~46% of hand evaluations still fall back to Python**
  (`evaluate_played_cards` called 136K times vs Rust
  `evaluate_simple_with_levels` 158K) — reducing that bail rate is
  the second-biggest lever. Note: more Rust speed helps Phase-8
  data-gen throughput, not winrate (winrate is value-function-bound).

### 2026-05-29 — opt-in Rust best-play fast path; exact parity blocked by stateful jokers

Profiled the **live `basic_strategy_bot`** (separate from the solver) and
found its play-search spends ~87% of a game in pure-Python
`hand_evaluator.best_play_from_hand`, which brute-enumerates every 1..5-card
subset (~333K `evaluate_played_cards` calls/game, ~38 s/game). The build/shop
valuation already uses Rust; this hot loop did not.

Added `search/rust_bridge.rust_best_play_scores` — batch-scores every subset
via `balatro_core.score_play_actions_batch` in one FFI call, then Python
builds the full `HandEvaluation` only for the winner (ties broken in Python on
the exact `(score, chips, mult)` key). Gated behind `BALATRO_RUST_BESTPLAY`.
**Measured 2.1× (26.6 → 11.0 s/game).**

**Default OFF — exact parity is not achievable here.** A full-vector parity
check (`scripts/bestplay_parity_check.py`, `BALATRO_BESTPLAY_PARITY=1`) over 8
games showed ~9% of calls bail and ~1.5% of *decisions* differ. The
divergences concentrate on **stateful jokers** — Ride the Bus, Bull, Banner,
Blue Joker, The Family — where the Rust *simple* evaluator and Python
`evaluate_played_cards` legitimately disagree (e.g. Rust models Ride the Bus's
face-card mult reset; the Python projection path does not, and the live bot
compensates for that in `play_scoring`'s explicit `_ride_the_bus_*` logic).
The culprits co-occur, so a clean bail-list isn't tractable, and the shift
moved a 100-seed winrate 14→11. Decision: keep the canonical bot bit-for-bit
pure-Python, expose the fast path as an opt-in for speed-tolerant bulk work
(e.g. data-gen). Reinforces the standing note: Rust play-search speed is a
throughput lever, not a winrate lever.

### 2026-05-30 — Phase 4d native-beam: complete divergence spec + RNG enabler done

Built `scripts/native_beam_divergence.py` (side-by-side: drives the Python-beam
trajectory, asks BOTH beams per state). Result on AAAAAAA: the existing
`beam.rs` is **~4× faster (11.5s vs 45s) but 72/91 = 79% of play decisions
diverge**, dying at ante 1 — reproducing both prior attempts.

**Root cause is a cascade of byte-sensitive transitions, NOT one bug.** The
existing `beam.rs` is a *simplified, structurally-different* port. To reach
byte-identical (the only thing that ever held parity — "quality-equivalent"
xoshiro failed twice), match Python EXACTLY, in this order (each gated by the
divergence harness; the leaf drives the argmax so #1-3 and #4-5 must land
together):

1. **Inter-node RNG** — Python: `Random(config.seed + seed_offset + action_index*1000003)`
   then `random.sample(range(len(pool)), draw_count)`. ✅ **DONE+VERIFIED**:
   `botlatro-core/src/search/py_random.rs` is a bit-exact MT19937 + `random.sample`
   port (231 (seed,n,k) cases, both pool & set paths). The two prior attempts used
   xoshiro → diverged at every branch.
2. **Draw pool order** — Python samples from `DeckModel._expanded_pool()` which
   is `sorted(counts, key=_key_sort)` then repeated representatives. The Rust beam
   gets `state.known_deck` order. FIX: have `_try_rust_beam_plan_value` pass
   `list(DeckModel.from_state(state)._expanded_pool())` as known_deck; the beam
   maintains it (removing by sampled index keeps it sorted).
3. **seed_offset threading** — plan→action uses `seed_offset + (i+1)*131071`;
   action→plan (per draw) uses `seed_offset + (j+1)*65537`; leaf uses
   `config.seed + seed_offset`. Thread `seed_offset:u64` + `base_seed:u64`
   through `beam_plan_value`; add a `seed_offset` pyfunction param.
4. **Hand order after draw** — Python `simulate_play/discard` does
   `next_hand = _sort_hand_cards(held + drawn)`. `_sort_hand_cards` sorts by
   (rank A..2, suit S,H,C,D, enhancement, seal, edition, debuffed). The beam does
   unsorted `held.chain(drawn)`. FIX: sort with that exact key after each draw.
5. **Leaf value** — Python `planning_value` =
   `clear*(1+headroom*0.25) + (1-clear)*progress*0.15` (cleared: `1+min(0.75,
   headroom*0.25)`). The beam returns `clear` only. FIX: port the full formula +
   `headroom_value`/`progress`.
6. **Leaf rollout** — the beam has its OWN play-only 16-sample rollout; Python's
   `clear_probability` is the now-discard-aware 1-sample `rollout.rs`
   `greedy_rollout_clears` (seed `config.seed + seed_offset`). FIX: refactor
   `greedy_rollout_clears` to be callable from `beam.rs` and use it.
7. **forward_sim parity** — `apply_play` must equal `simulate_play` and
   `apply_discard` must apply `jokers_after_discard` (currently a no-op, line ~382).

Workflow gotcha (cost me a cycle): `maturin develop` builds
`target/release/balatro_core.dll` but does NOT copy it into
`site-packages/balatro_core/balatro_core.pyd` here — `cp` manually after every build.
