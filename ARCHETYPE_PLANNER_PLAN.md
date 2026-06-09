# Archetype / Synergy Planner — Design Sketch

Goal: capture the **build-construction headroom** (out-test: ~74% of ante-8 losses had an
affordable, offered joker that would have cleared — the bot reaches the wall one engine-piece
short). Per-decision signals can't capture this (shop selection is near-optimal; neural value
flat; clear-capacity hurts as a leaf). The proven lever in comparable deckbuilders (Slay the
Spire, MTG) is **commit to a build archetype and value acquisitions by fit to that plan** —
hierarchical decomposition that supplies the long-horizon structure a learned value can't.

## What already exists (reuse, do not rebuild)

- `solver/archetypes.py` — 4 `BUILT_IN_ARCHETYPES` (flush, scaling_joker, high_card_mult,
  pair_retrigger). Each carries the **synergy knowledge**: `target_hand_types`, `key_joker_keys`,
  `key_consumable_keys`, and `archetype_fit_score(state)` = (# owned matching items) ×
  `shop_bonus_per_match`.
- `solver/policy.py` `ArchetypeAwareLeaf` — wraps the shop leaf and adds `archetype_fit_score`;
  activated when `SolverPolicy(archetype=...)` is set. **The bias mechanism already works.**
- `solver/multi_archetype.py` — runs baseline + each archetype per seed and keeps the best
  (`_trajectory_score`: won > final_ante > final_score). This is a **seed-known ORACLE** for
  archetype choice (and a labeled-data generator: which archetype wins each seed).
- `bots/basic_strategy/build_profile.py::_build_archetype(state, preferred)` — already classifies
  the run's *current* archetype from owned items.
- `bots/basic_strategy/data.py` — joker→archetype-hand maps (FLUSH_ARCHETYPE_HANDS, etc.);
  `hand_preferences.py` — preferred-hand commitment with archetype support counts.
- `run_plan.py` / `shop_planner.py` — existing build-planning scaffolding.

## The gap

The **deployed live bot never commits** (`solver_shop_basic_play_bot` → `SolverPolicy(archetype=None)`;
no `archetype` in `registry.py`). The offline `multi_archetype` only commits by trying *all* and
keeping the best — unavailable live. And per M6a, **a wrong single commit HURTS** (soft-bias toward
items that don't fit). So the missing piece is a **good live archetype SELECTOR**: pick the right
archetype to commit to during one run, conservatively (allow "none"), with hysteresis.

## Architecture

```
ArchetypePlannerBot(wraps SolverPolicy shop + BasicStrategy play)
  on each shop decision (and blind-select):
    1. SELECT archetype = argmax_a fit_potential(state, a)   [or None if all weak]
       - committed once chosen (hysteresis); re-evaluate only if a clearly dominates
    2. set policy.archetype = selected   (drives ArchetypeAwareLeaf in the shop beam)
    3. delegate to SolverPolicy.choose_action (shop leaf now biased toward the plan)
```

`fit_potential(state, archetype)` — richer than the current owned-key count:
- **owned key jokers/consumables** (existing `archetype_fit_score`) — committed progress,
- **target-hand alignment** — hand_levels + played-hand history + deck composition favoring
  `target_hand_types` (e.g. flush ⇐ suit concentration; pair_retrigger ⇐ paired ranks),
- **economy headroom** to actually afford the plan,
- **anti-flip-flop**: bonus to the already-committed archetype.

Commit policy (mitigates the "wrong commit hurts" risk):
- Commit only when the leading archetype's potential clears a margin over `None` (baseline).
- Commit **early** (antes 1–3) when the engine is forming — that's the whole point (winners
  commit/scale; losers spread thin).
- Keep `None` as a first-class option (never commit to a bad fit).

## Phased plan

**Phase 0 — Leverage gate + label generation (cheap, decisive; do FIRST).**
Run `multi_archetype` (baseline + 4 archetypes) on N held-out seeds, **using the deployed play
backend (BasicStrategy play), not SolverPolicy's v2 play**, so it's comparable to the real 19.5%
bot. Compare **best-of-archetypes (oracle) winrate vs baseline**.
- Oracle ≫ baseline → correct commitment has real headroom → build the selector; its job is to
  approximate the oracle choice. Also yields per-seed labels (which archetype won) for Phase 3.
- Oracle ≈ baseline → the archetype *leaf* is too weak even with perfect choice → strengthen
  `archetype_fit_score` (Phase 2 mechanics) before any selector.

**Phase 1 — Heuristic live selector.** Implement `fit_potential` + commit policy + the
`ArchetypePlannerBot` wrapper. A/B winrate vs `solver_shop_basic_play_bot` on held-out seeds.
Target: close a meaningful fraction of the (oracle − baseline) gap from Phase 0.

**Phase 2 — Strengthen the plan bias.** Tune `shop_bonus_per_match` (currently 2.0, soft);
add target-hand alignment to the fit-score; **mine winning runs** (build_profile) to expand
`key_joker_keys` / add archetypes (data-driven synergy, not just hand-authored).

**Phase 3 — Neural selector (the RIGHT neural niche).** Learn archetype SELECTION as a
**low-dimensional 4-way classification**: early-run state → which archetype wins, trained on
Phase-0 `multi_archetype` labels. This is the tractable neural target (unlike the flat per-card
value): few classes, sharp label, abundant via the oracle. Neural picks the plan; search executes it.

## Evaluation

Winrate bench on contiguous held-out seeds vs the `19.5%` baseline (and stacked with the confirmed
`ECON_W=1.5` economy lever). Mean ante as the dense secondary signal. Diagnostic: which archetype
the selector commits to, and how often it matches the Phase-0 oracle.

## Risks / mitigations

- **Wrong commit hurts (M6a):** conservative commit margin, `None` always allowed, hysteresis.
- **Fit-score too weak (only counts owned keys):** Phase 2 enriches with target-hand alignment.
- **Archetypes incomplete (4 hand-authored):** Phase 2 mines winning runs to expand.
- **Play-backend confound:** always A/B with BasicStrategy play (the deployed bot), never v2.
```
