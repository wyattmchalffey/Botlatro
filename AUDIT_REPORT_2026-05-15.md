# Botlatro Audit — 2026-05-15

Overnight review of the project outside of `basic_strategy_bot.py` (parameter sweep was running, not touched). Goal: surface issues that are blocking the short-term target ("consistently beat White Stake") and long-term goal ("superhuman play"). Findings are grouped by area, ordered roughly by expected impact on win-rate. File-path citations use the file_path:line form so you can jump straight in.

> **Status update — 2026-05-15 post-verification.** After verifying every Tier 1-2 "real bug" claim against the actual code and against the Balatro Lua dump at `C:\Users\Wyatt\AppData\Roaming\Balatro\Mods\lovely\dump\`, four items were genuine bugs and have been fixed; four were over-flagged by the audit agents and the code was already correct. See **"Post-verification status"** at the bottom for the canonical list before acting on anything in the body of this report.

Coverage of this audit:
- **Search architecture** — done in depth. (Body unchanged; **NOT independently verified post-hoc, treat individual claims with same skepticism as Tier 1-2 — many likely real, some likely over-flagged.**)
- **Eval / benchmark harness** — done in depth, then verified. 3/5 highlighted issues were real and have been fixed (B2 RunTimeout, B3 Wilcoxon, B4 Holm). 2 were over-flagged (B1, B6 was real but not addressed yet).
- **Hand evaluator + joker_compat** — done in depth, then verified. C1 and C2 were over-flagged — the code was already correct. Remaining items (C3 Misprint, C4 ordering, C5 float overflow, C6 purple seal, C7 Cavendish, C8 unhandled) still untested.
- **Peripheral modules (state, env, api, probability, replay logger, registry, config)** — done.
- **Local simulator + forward_sim** — partial originally; **the Soul/Black Hole bug (B13) is real and was verified against `common_events.lua:2401-2410`. Now fixed.** An initial new finding ("shop joker dedupe is missing run-wide") was retracted on re-read of the Lua source — see Addendum 3 below for the correction.
- **Test coverage** — partial.

---

## Section A — Highest-impact issues for short-term win-rate

These are the items most likely, individually, to move the search bot past the rule bot, or to make the rule bot's eval signal trustworthy. Roughly ranked by expected effect size.

### A1. The search bot's "guard" routinely reverses search decisions
[src/balatro_ai/bots/search_bot.py:128-130](src/balatro_ai/bots/search_bot.py:128) calls `_basic_blind_action_should_guard` ([:271-279](src/balatro_ai/bots/search_bot.py:271)). The function returns True whenever `{basic_action_type, search_action_type} ⊆ {PLAY_HAND, DISCARD}` — i.e. whenever the search wants to play and basic wants to discard (or vice versa), **the basic bot's pick wins.** Combined with starved sampling (A2), this means the search rarely gets to change the most important decision basic makes. v2 partially routes around this via the `metadata.search == "hand_beam"` check ([search_bot_v2.py:291-294](src/balatro_ai/bots/search_bot_v2.py:291)) but the standard expectimax path still loses.

### A2. Search bots override the search modules' good defaults with starved budgets
- [src/balatro_ai/bots/search_bot.py:51](src/balatro_ai/bots/search_bot.py:51): `DiscardSearchConfig(draw_samples=1, leaf_samples=1, max_actions=8)` — the module default is `32/16/48`.
- [search_bot.py:54-58](src/balatro_ai/bots/search_bot.py:54): `HandSearchConfig(draw_samples=1, leaf_samples=1)` — already the module default, but the action ceilings `max_play_actions=16, max_discard_actions=8` mean only 16/8 of ~218 legal hands are considered.
- [search_bot.py:61](src/balatro_ai/bots/search_bot.py:61): `PackSearchConfig(leaf_samples=1, stochastic_samples=4)` (module default 16/8).
- [search_bot.py:63-68](src/balatro_ai/bots/search_bot.py:63): same starvation for `ConsumableSearchConfig`.

v2 raises sampling to 2/2 ([search_bot_v2.py:51-55](src/balatro_ai/bots/search_bot_v2.py:51)) — still essentially deterministic noise. With `samples=1`, `clear_probability ∈ {0, 1}`. Two distinct actions look identical at the leaf.

### A3. Two leaf-value functions with incomparable scales mix freely
- [src/balatro_ai/search/state_value.py:165-185](src/balatro_ai/search/state_value.py:165) — `state_value = clear*0.8 + clear*future*0.2`, range [0, 1]. Components are all clamped at 1.0, so "clear + huge surplus" looks identical to "bare clear."
- [src/balatro_ai/search/shop_search.py:362-438](src/balatro_ai/search/shop_search.py:362) — `shop_leaf_terms` sums ~8 weighted terms minus penalties, range roughly [-260, +400].
- [src/balatro_ai/search/consumable_search.py:252-257](src/balatro_ai/search/consumable_search.py:252): the default consumable value-fn routes by phase — SHOP → `shop_leaf_value`, BOOSTER_OPENED → blend, else `state_value*100`. So the same consumable is valued in shop-leaf units one turn and `state_value*100` units the next, with no scale conversion. The thresholds `min_shop_delta=2.0` and `min_blind_delta=6.0` apply to incomparable numbers.

`planning_value` ([state_value.py:188-213](src/balatro_ai/search/state_value.py:188)) uses `headroom_value` (unclamped up to ~5) and is the *better* leaf, but `hand_search`'s default value_fn at [hand_search.py:455-456](src/balatro_ai/search/hand_search.py:455) returns `state_value(...) * 100.0` — the flat one. Only the beam path ([hand_search.py:243](src/balatro_ai/search/hand_search.py:243)) uses `planning_value`, and beam is disabled by default.

### A4. Beam search is gated off and the transposition memo is dead
- [src/balatro_ai/search/hand_search.py:27-28](src/balatro_ai/search/hand_search.py:27): `HandSearchConfig.beam_depth = 0, beam_width = 0`. v1 never overrides; only v2 sets `beam_depth=3, beam_width=2` with `beam_draw_samples=1, beam_leaf_samples=1` ([search_bot_v2.py:60-63](src/balatro_ai/bots/search_bot_v2.py:60)).
- [hand_search.py:131](src/balatro_ai/search/hand_search.py:131): `memo: dict[tuple[object, ...], float] | None = None` is passed as `None` to `_beam_action_value`, and the transposition lookup `if memo is not None and memo_key in memo` is therefore always False. The transposition cache is dead code, so beam recomputes identical subtrees and developers correctly intuit that they need to set `beam_width=2` to keep latency in check.

### A5. `discard_search`'s leaf is much better than `hand_search`'s and is dead code
[src/balatro_ai/search/discard_search.py:198](src/balatro_ai/search/discard_search.py:198) uses `one_hand*0.70 + clear*0.25 + clear*future*0.05`. `best_discard_action` is defined but never called by the bots — they route everything through `hand_search.best_hand_action`, which imports only `_candidate_discard_actions` and `_discard_candidate_score` from `discard_search`. The whole `discard_action_value` / `_default_value_fn` / `_one_hand_clear_value` pipeline ([discard_search.py:57-211](src/balatro_ai/search/discard_search.py:57)) is orphaned. The hand bot is stuck with `state_value*100`.

### A6. Shop sell-penalty is 6× too generous
[src/balatro_ai/search/shop_search.py:296-297](src/balatro_ai/search/shop_search.py:296):
```python
return max(0.0, float(sold.sell_value or 0) - (_owned_joker_value(...) * 0.15))
```
`sell_value` is typically 2-10; `_owned_joker_value` returns a build-strength figure of 20-100+. Multiplying by 0.15 makes the deduction small relative to the strategic loss. Combined with `_sell_is_search_candidate` ([:1361-1365](src/balatro_ai/search/shop_search.py:1361)) only filtering `sell_value > 0`, the shop beam will eagerly liquidate good jokers for marginal buys. This alone could explain a meaningful slice of mid-ante losses where the bot suddenly has fewer jokers than it should.

### A7. Stochastic action comparisons share no common random numbers
[pack_search.py:266-275 + 538-542](src/balatro_ai/search/pack_search.py:266), [consumable_search.py:481-485](src/balatro_ai/search/consumable_search.py:481), [hand_search.py:519](src/balatro_ai/search/hand_search.py:519) all XOR the action's `stable_key` into the per-sample seed. So competing actions are evaluated against entirely different random streams. With `stochastic_samples=4`, the noise per pairwise comparison swamps the signal — the "best" pack/consumable is essentially random among similar candidates. Standard fix: share the seed for the random component and let only deterministic content vary per action (common random numbers / antithetic variates).

### A8. `_state_value`-style cache identity keys can collide
[state_value.py:62-73](src/balatro_ai/search/state_value.py:62) caches by `id(obj)`. `id()` is reused by CPython when objects are GC'd. The forward sim builds many Card objects per decision; an `id` collision returns a cached value for the wrong card. Hard to reproduce, easy to land — switch to content-keyed caching for Card.

### A9. `state_value._score_action` has a no-op cache
[state_value.py:393-438](src/balatro_ai/search/state_value.py:393): `_score_action` simply forwards to `_score_action_uncached`. The cache scope set up at [:25-44](src/balatro_ai/search/state_value.py:25) is never used. Per-action evaluation cost is paid every call, which forces sample budgets to stay at 1 to keep latency reasonable.

### A10. `_play_actions_for_hand_size` enumerates ~218 plays per rollout step
[state_value.py:380-390](src/balatro_ai/search/state_value.py:380): For an 8-card hand, `combinations(range(8), k)` for k=1..5 yields 1+8+28+56+70+56 = 219 actions. Called from `_best_immediate_score`, `_best_greedy_play_action`, and `_score_action` inside the rollout loop. No early termination once a clearing play is found.

### A11. Opening-hunt return path can yield a non-legal action
[hand_search.py:49-51](src/balatro_ai/search/hand_search.py:49) returns `_annotated_action(opening_hunt, search_value=128.0, ...)`. The candidate-matching at [:617](src/balatro_ai/search/hand_search.py:617) (the helper that maps the basic-bot-built action to a legal action) falls through when card_indices ordering differs, and returns the basic-bot-constructed action verbatim, which the engine may reject.

### A12. Pack-sell path only considers selling when the pick is illegal
[pack_search.py:171-173](src/balatro_ai/search/pack_search.py:171) gates the sell-then-pick branch on the pick currently being illegal. Selling a low-value joker to free a slot for a *better* swap (already legal because another joker would take the slot) is never proposed.

### A13. Shop "skip" pack-action legality is not checked against `state.legal_actions`
[pack_search.py:192-202](src/balatro_ai/search/pack_search.py:192) returns True unconditionally for skip. Some bridge states (e.g. forced-pick Buffoon with empty slots) reject skip. Latent illegal-action risk.

### A14. Shop search's reroll `min(sampled, blended)` discards the blend
[shop_search.py:285-286](src/balatro_ai/search/shop_search.py:285): `blended_value = basic*0.55 + sampled*0.45; return min(sampled, blended_value)`. The `min` almost always selects `sampled_value` (whenever `basic_value > 0`), making the blend ornamental — and giving an anti-reroll bias.

### A15. Reroll-block fallback denies rerolls on any internal error
[shop_search.py:341-348](src/balatro_ai/search/shop_search.py:341): `except (ImportError, TypeError, ValueError, AttributeError): return state.ante >= 2`. A broad except inside reroll-gating hides real bugs and silently kills late-game rerolls when anything in `_shop_pressure` glitches.

---

## Section B — Eval harness biases (you cannot trust an A/B until these are fixed)

These directly affect how confident you can be that "param X improved win rate."

### B1. Ante-9 win-flip can convert recovered/stale states into spurious wins
**~~Real bug~~ — FALSE POSITIVE on re-verification.** This is intentional logic per PROGRESS.md 2026-05-02: nonterminal `ante=9` cleanup states are White Stake wins; terminal `ante=9, run_over=True, won=False` losses stay losses. The agent's "stale state" concern was theoretical without a reproducer. No fix needed.

[run_seed.py:324-330](src/balatro_ai/eval/run_seed.py:324) (and the mirror in [api/state.py:503-505](src/balatro_ai/api/state.py:503)):
```python
if state.ante >= 9:
    won = state.won or not state.run_over
    return replace(state, ante=8, run_over=True, won=won, legal_actions=())
```
Combined with the silent recovery loop in [run_seed.py:67-77](src/balatro_ai/eval/run_seed.py:67) (sleep + re-query on stale state) and the swallowed `INVALID_STATE`/"Card index out of range" errors at [:85-100](src/balatro_ai/eval/run_seed.py:85), a stale snapshot showing ante=9, run_over=False can be flipped to a win. Replace `not run_over` with an explicit "saw blind clear" gate.

### B2. `RunTimeout` is treated as a retryable bridge error
**Real bug. FIXED 2026-05-15.** [runner.py:232-240](src/balatro_ai/eval/runner.py:232) now excludes `error:RunTimeout` from `_is_retryable_seed_failure`. RunTimeout is wall-clock-driven and retrying just re-rolls the budget.

Original finding: `_is_retryable_seed_failure` matches any `death_reason.startswith("error:")`, including the `error:RunTimeout` produced by `--run-timeout-seconds`. Slow-but-legitimate runs get retried with a fresh wall-clock budget. The bias direction: **timeouts favor the faster bot**. If your search bot is slower on hard seeds, you're attributing real losses to "the bot took too long."

### B3. Wilcoxon p-values lack tie correction
**Real bug. FIXED 2026-05-15.** [compare.py:301-323](src/balatro_ai/eval/compare.py:301) now subtracts `sum(t³−t)/48` over tied absolute-difference groups from the variance.

**Correction to original framing**: I originally wrote "p-values are anti-conservative — more significant-looking than they are." That was inverted. Hand-verification: with ties at |d|=1 and 7 positive / 3 negative diffs, uncorrected variance is 96.25 and gives p=0.2845; corrected variance is 75.625 and gives p=0.2273. Tie correction REDUCES variance → increases |z| → DECREASES p. So without correction the test was **over-conservative** (Type II inflated), hiding real effects. The fix surfaces them. Direction of user impact: helps you find real improvements that were getting buried.

Original finding: standard formula needs `sum(t^3 - t)/48` variance adjustment for tied ranks. Ante deltas are tie-heavy (most pairs differ by 0/±1).

### B4. `config_sweep` makes ~10 paired McNemar tests with no multiple-comparisons correction
**Real bug. FIXED 2026-05-15.** [config_sweep.py:285-310](src/balatro_ai/eval/config_sweep.py:285) now provides `_holm_adjusted_p_values()`; the sweep table and CSV expose a `Holm p` column. Verified against the canonical Holm example.

Original finding: With six+ sweep values you have a ~25-30% chance of at least one spurious p<0.05 even when nothing differs. This directly biases parameter tuning toward false positives.

### B5. Average ante deflates when one arm errors more
[metrics.py:78-94](src/balatro_ai/eval/metrics.py:78) computes `mean(ante_reached)`. Error seeds have `ante_reached=0` ([runner.py:314](src/balatro_ai/eval/runner.py:314)). If bot B has 2% more bridge errors than bot A, B's average ante is silently lower even with identical playthrough quality.

### B6. `PYTHONHASHSEED` is set for `local_benchmark` and `config_sweep` but not for `benchmark` or `compare`
[local_benchmark.py:119](src/balatro_ai/eval/local_benchmark.py:119), [config_sweep.py:387](src/balatro_ai/eval/config_sweep.py:387) re-exec with `PYTHONHASHSEED=0`. The bridge-side `benchmark.py` / `compare.py` do not. Any bot whose decisions touch unsorted set/dict iteration has an extra nondeterminism vector in bridge benchmarks.

### B7. Replay summary mode silently disables `replay_diff` / `replay_analyzer`
[run_seed.py:106-122](src/balatro_ai/eval/run_seed.py:106) skips per-step `log_step` when `replay_mode=="summary"`. `compare.py` defaults to summary. `replay_analyzer.py:204-211` parses `state` debug strings for `max_ante` and falls back to 0 if absent — so summary-only runs report "everyone died at ante 0" in analysis. The two tools claim to work together but break when both are using their defaults.

### B8. Seed pool is regenerated by label, not pinned to a file
[seed_sets.py:29-44](src/balatro_ai/eval/seed_sets.py:29) hashes the label string. Two consequences: (a) changing the label-format adds invisibly silent diff, and (b) `make_seed_set("white:default", 100)` is **not** a prefix of `make_seed_set("white:default", 1000)` — so the "100" sweep and the "1000" sweep cover overlapping but not nested pools. Pin the 100/1000/10000 pools to checked-in JSON files.

### B9. Retry overwrites the original replay, destroying audit evidence
[runner.py:224](src/balatro_ai/eval/runner.py:224) `_delete_replay_for_seed` unconditionally drops the first attempt before retry. You can't tell how often retries flipped a result.

### B10. Endpoint re-queued even when park fails
[runner.py:128-130](src/balatro_ai/eval/runner.py:128) puts the endpoint back in the queue regardless of `_park_endpoint` return value. Silently degrades workers.

### B11. `compare.py` runs Bot A's full sweep, then Bot B's
[compare.py:583-598](src/balatro_ai/eval/compare.py:583). The bridge state at the moment B starts is whatever A left it as. If a bridge worker dies between bots, the pairing is no longer same-conditions.

### B12. Local-sim and bridge winrates are not interchangeable
[local_runner.py:1-7](src/balatro_ai/sim/local_runner.py:1) admits this explicitly. The local sim uses a `random.Random(seed)` ([:287](src/balatro_ai/sim/local_runner.py:287)), not Balatro's seed-derived RNG. Shop sampling, boss selection, joker procs all draw from the wrong distribution. `config_sweep` only runs local sim — anything that wins on local-sim must be re-validated on bridge before being treated as real progress. The 99.9% replay-validator pass rate is misleading: it works because the validator *injects* observed outcomes, not because the RNG matches.

---

## Section C — Scoring engine gaps

The hand evaluator is the foundation. It's been heavily worked, but coverage holes remain.

### C1. `Oops! All 6s` does nothing
**~~Real bug~~ — FALSE POSITIVE on re-verification.** The agent grepped only `hand_evaluator.py`. Oops! All 6s is implemented via state modifier indirection: [forward_sim.py:2640-2642](src/balatro_ai/search/forward_sim.py:2640) sets `probability_multiplier *= 2` when acquired (halves when sold), and every `_roll_odds(N, probability_multiplier=PM)` call across [local_runner.py:584](src/balatro_ai/sim/local_runner.py:584), [shop_search.py:1554](src/balatro_ai/search/shop_search.py:1554), [consumable_search.py:396](src/balatro_ai/search/consumable_search.py:396) reads it. Works correctly. No fix needed.

### C2. Pareidolia + Triboulet / Shoot the Moon / Baron
**~~Real bug~~ — FALSE POSITIVE on re-verification.** In real Balatro, Pareidolia affects `is_face()` checks only. Triboulet, Baron, and Shoot the Moon each use direct rank checks (`base.id == 12 or 13`, `base.id == 13`, `base.id == 12` respectively in source). Pareidolia does NOT change a card's rank, so these three correctly fire only on actual Q/K. The Botlatro code at [hand_evaluator.py:1102, 1383, 1387](src/balatro_ai/rules/hand_evaluator.py:1102) matches source behavior. No fix needed.

### C3. `Misprint` defaults to 0 mult, not the mean
[:1160](src/balatro_ai/rules/hand_evaluator.py:1160). Caller-driven `stochastic_outcomes["misprint_mult"]` is 0 if missing. Any non-replay eval treats Misprint as worthless. For the deterministic audit this looks fine (0 misses) but for *EV reasoning* it's wrong by ~11.5 mult per played hand.

### C4. Joker-ordering interleaving when a hand-level joker sits left of `Photograph`
[:1110-1287](src/balatro_ai/rules/hand_evaluator.py:1110) is a two-loop structure: per-scored-card jokers, then on-played-hand jokers. Real Balatro fires by physical-slot order. A leftward `Joker Stencil`/`Madness` should resolve its XMult before Photograph's per-card X2, but here Photograph fires first. Edge case but affects rare Stencil builds.

### C5. Float overflow chain has no clamp
[:1290-1291](src/balatro_ai/rules/hand_evaluator.py:1290) `_score_floor` uses `math.floor`. `effect_xmult` is Python float; chained X100+ at high hand levels produces `inf`, then `floor(inf)` → error. Balatro clamps at ~1.7e308.

### C6. Purple seal entirely unimplemented
Tarot creation EV from purple seals is lost. Only mentioned as a `SealKind` enum in [state.py:33](src/balatro_ai/api/state.py:33). Forward sim has no path either.

### C7. Cavendish flat X3 missing the death-roll side effect
[:1274-1275](src/balatro_ai/rules/hand_evaluator.py:1274). Score is right but the joker should remove itself stochastically — caller will overestimate Cavendish's long-term EV.

### C8. Unhandled jokers with state-shape effects
`Sixth Sense` (sole-Six replaced with Spectral pre-score), `DNA` (card duplication), `Burnt Joker`, `Mail-In Rebate` rank parsing, `Chicot` (boss disable) — listed in rarity dict, no scoring branch. Most are correctly zero-impact at score time but a few mutate the played hand or change which debuffs apply.

### C9. Untested hand types: Flush House, Flush Five, edge cases under Four Fingers
No test asserts that 4 same-rank + 1 stone with Four Fingers correctly resolves to Four of a Kind. No Flush House or Flush Five fixtures.

### C10. No fuzz / property tests, no `hypothesis`
Scoring is a pure function over a moderate input space. Zero use of `hypothesis` or any property testing. A single afternoon's worth of random-input fuzzing would likely surface several latent edge cases.

---

## Section D — Local simulator & forward sim notes

The deep audit got rate-limited. From spot-checking:

### D1. Local sim RNG is plain `random.Random(seed)` — not Balatro's RNG
[local_runner.py:287](src/balatro_ai/sim/local_runner.py:287): `self._rng = Random(self.seed)`. Every shop sample, joker proc, boss selection, deck shuffle draws from a python.random stream that has zero relationship to Balatro's seed-derived RNG. So:
- Two runs with the same seed across local-sim ↔ bridge will see different shops, different bosses, different boss pools, different joker rolls.
- The "23/200 strict 200" local-sim winrate cannot be expected to predict the bridge winrate of any specific tuning change.
- The replay validator's 99.9% match rate doesn't refute this; the validator injects observed outcomes rather than rolling them itself.

This is documented at [local_runner.py:1-7](src/balatro_ai/sim/local_runner.py:1) but the docstring's "not the source of truth for official benchmarks" is easy to forget once you've spent a week iterating only on local-sim numbers. The `.data/` directory shows you've been doing exactly that.

### D2. Forward sim never invents outcomes
[forward_sim.py:1-7](src/balatro_ai/search/forward_sim.py:1) — by design. Callers must inject stochastic outcomes. This means search code rolling its own forward sim sees a *zeroed* stochastic outcome by default unless the caller threads `StochasticPlayOutcomes` through. Grep `StochasticPlayOutcomes` to confirm — most search call-sites do not pass it.

### D3. `state_value._greedy_rollout` likely doesn't pass StochasticPlayOutcomes
A quick scan of [state_value.py](src/balatro_ai/search/state_value.py) shows no `StochasticPlayOutcomes` import. If the rollout never sees randomness, then `clear_probability` with `samples=N` is doing N identical rollouts — i.e. variance is artificial sampling noise, not the real stochastic spread. Worth confirming before raising sample counts in A2.

### D4. Skip tags and the recent voucher/boss work are huge surface area; smoke tests are thin
The PROGRESS.md log around 2026-05-02 to 05-08 added skip tags, voucher effects, source boss pools, Crimson Heart, Cerulean Bell, etc. That's hundreds of lines of new logic. The bridge-backed joker smoke validates 185 scenarios, but skip tags / voucher effects / boss-blind-on-blind-select transitions aren't all replay-validated in the same way. Any one of them could silently produce a different game tree.

---

## Section E — Test coverage

I couldn't get the deep dive, but ran the structural check.

### E1. Untested source modules
The following `src/balatro_ai/**/*.py` files have **no corresponding `tests/test_*.py`**:
- `bots/base.py`, `bots/registry.py`, `bots/search_bot_v2.py`, `bots/config.py`
- `eval/benchmark.py`, `eval/local_benchmark.py`, `eval/config_sweep.py`
- `env/balatro_env.py`, `env/rewards.py`
- `rules/hand_evaluator.py` ← surprising; tests exist as `test_hand_eval.py` but no `test_hand_evaluator.py`
- `sim/replay_validator.py`
- `search/consumable_search.py` ← surprising; tested indirectly via `test_search_bot.py`, but no dedicated unit tests
- `api/actions.py`, `api/client.py`, `api/state.py`
- `gui/benchmark_app.py`, `gui/hand_probability_app.py`

The `hand_evaluator.py` case is the most worrying — 2127 lines of math with no direct unit-test file. Coverage exists via `test_hand_eval.py` (which exercises evaluator surface) plus fixtures, but a test file named after the module would force discipline.

### E2. No property / fuzz testing
No `hypothesis` import anywhere. Scoring is a pure function; flipping a random subset of card enhancements/jokers and asserting "evaluator agrees with itself under shuffling of joker order" or "stone-card-only hand scores X" would catch ordering bugs cheaply.

### E3. `basic_strategy_bot.py` is 9615 lines; `test_basic_strategy_bot.py` is 4942 lines
A ~0.5:1 test:source ratio for a rule bot of this size is not enough to lock behavior under refactors. Many of those 9615 lines are heuristic constants that have been hand-tuned over months; without behavioral lock-in tests, your overnight sweep can find a "better" tuning that quietly regresses an untested edge case.

### E4. Bridge-dependent tests
`test_balatrobot_schema.py` and `test_runner.py` reach for `http://127`-ish endpoints. They'll silently skip / error when the bridge isn't running. Confirm what your CI/local-test workflow does — silent skip is the failure mode that lets a real regression slip in.

### E5. No skipped / xfail markers, no `pytest.skip`
Nothing is explicitly marked skip/xfail in tests. So either the test suite genuinely covers everything it claims to (good), or there are missing tests that nobody has written down as TODOs (more likely).

### E6. No coverage tooling
[pyproject.toml](pyproject.toml) lists only `pytest>=8` in dev deps. No `coverage`, no `pytest-cov`, no `hypothesis`, no `ruff`, no `mypy`. No CI config visible. Adding `coverage run -m pytest && coverage report` once would tell you where the cliffs are.

---

## Section F — Peripheral modules I personally reviewed

These I read fully rather than delegating.

### F1. `api/state.py` — Card.from_mapping silently produces "None" rank
[state.py:362-380](src/balatro_ai/api/state.py:362): when `data.get("rank")` is missing AND `value.get("rank")` is missing, `str(None) = "None"`. The resulting Card carries `rank="None"`. Hand evaluator probably tolerates it (RANK_VALUES will KeyError) but the failure mode is silent until score-time. Either error early or default to a sentinel.

### F2. `api/state.py` — `_with_derived_legal_actions` uses 5-card cap for play
[state.py:797](src/balatro_ai/api/state.py:797): `max_cards = min(5, len(state.hand))`. Correct for normal play (max 5 played cards). But Four Fingers / Shortcut don't change the played-count maximum, so this is fine. Confirming there's no missing case.

### F3. `api/state.py` — Ante-9 normalization is duplicated
The same `if state.ante >= 9` block exists in `state.from_mapping` ([:503-505](src/balatro_ai/api/state.py:503)) and `run_seed._with_standard_win_boundary` ([:324-330](src/balatro_ai/eval/run_seed.py:324)). Both flip `won = state.won or not state.run_over`. The duplication is small, the semantic risk is large (B1).

### F4. `data/replay_logger.py` — no post-state per step
[replay_logger.py:18-43](src/balatro_ai/data/replay_logger.py:18) writes pre-action state + chosen_action + reward. The next row's pre-state is the post-state of the previous action, **except** for the final action, whose post-state lives only in `final_state`. Replay diff handles this implicitly, but it makes "what did the last decision land us in?" hard to extract from any row except the summary.

### F5. `probability/hand_type_odds.py` — looks correct
Wide audit of `_has_exact_flush` / `_has_exact_straight` / `_has_exact_full_house` / `_has_exact_five_of_a_kind` against Balatro's "stronger type wins" rules looks right. The `lru_cache`s on `_suit_has_exact_flush` and `_suit_selection_vectors` are sound. No issues found here.

### F6. `env/balatro_env.py` and `env/rewards.py` — minimal scaffolding
[balatro_env.py](src/balatro_ai/env/balatro_env.py) is a 43-line Gym-like wrapper. [rewards.py](src/balatro_ai/env/rewards.py) gives `+10` for win, `-1` for loss, plus shaped chip/money/ante deltas. For Phase 10 RL this reward shape is sane, but the current weighting (`max(0, score_delta)/1000 + 0.02 * money_delta + ante_delta`) makes hands-for-pure-score the dominant signal at low antes, which will train an early-blind specialist. Worth re-shaping before Phase 10.

### F7. `env/observations.py` — minimal feature set
[observations.py:7-22](src/balatro_ai/env/observations.py:7) ships 14 scalar features. No card-level encoding, no joker encoding, no shop encoding. PLAN.md Phase 2 promises "fixed-size numeric tensor, card features, joker features, economy features, blind features, deck-composition features, legal action mask." Almost none of that exists. Phase 8 imitation learning needs this *before* it starts, not at the time of starting.

### F8. `bots/registry.py` — search bot variant matrix is implicit
[registry.py:21-25](src/balatro_ai/bots/registry.py:21):
- `search_bot` / `search_bot_v0` → `SearchBot(enable_shop_search=False)`
- `search_bot_v1` / `shop_search_bot` → `SearchBot(enable_shop_search=True)`
- `search_bot_v2` → `SearchBotV2()` (a separate class that duplicates 90% of v1)
The `SearchBot` vs `SearchBotV2` split looks like a fork-don't-modify pattern; both classes hold ~300 lines of nearly identical bookkeeping. Refactoring v2 onto a `SearchBotConfig` flag would let you A/B variants without duplicating bug fixes.

---

## Section G — Project-shape concerns

Cross-cutting issues that aren't bugs but will hurt you in Phase 8+.

### G1. `basic_strategy_bot.py` is 9615 lines
The rule bot is now bigger than the rest of `balatro_ai/` combined. With heavy local-state, threadlocal scope, regex effect parsing, and dozens of hand-tuned constants, it's the single point of dependency for every search bot, every test, and every benchmark. Concrete risks:
- **One bot, no behavioral lock-in tests** (E3). Your overnight tuner can win on aggregate while silently regressing an edge case.
- **Profile data once captured permutation cost of 720** (PROGRESS.md, fixed at 5-joker cap). There are very likely more O(n!) traps in there; one decision taking 15s blows out wall-clock at scale.
- **Memory of 2026-05-06** (`project_search_architecture.md`) called this a 5717-line bot; the file is now +3900 lines. Most of that growth was lossless feature work, but it raises the bar for anyone trying to rewrite or distill the bot into a neural policy.

### G2. Local sim has become the default eval loop without bridge parity
The `.data/` directory shows dozens of overnight local-sim runs and only a handful of bridge runs. Local sim is fast (D1) but has no shared RNG with Balatro. You're optimizing a proxy. The locked baseline at `baselines/basic_strategy_2026_05_02.json` is bridge-based at 7.4% but the current "23/200 strict 200" reference is local-sim at 11.5%. These are not directly comparable. Pin a *fresh* bridge baseline alongside each local-sim baseline.

### G3. `.data/` has accumulated ~100+ exploratory JSONL files
`git status` shows ~80 untracked `.data/codex-*.jsonl` files. Most are exploratory traces. Worth either pruning or moving truly-archival ones into a tracked manifest. Currently it's not obvious which `.data` files are still load-bearing vs which can be deleted.

### G4. No CI / no automated regression catch
No `.github/`, no `tox.ini`, no `noxfile.py`. The `python -m unittest discover -s tests` workflow is fine for hand iteration but cannot catch regressions until you remember to run it. Phase 8 onward, where neural training will silently produce regressions, this becomes a hard requirement.

### G5. Phase 8 prerequisites that aren't in place yet
PLAN.md gates Phase 8 on 40-50% white-stake winrate. Even if you hit 50% next month, Phase 8 needs:
- A proper observation tensor (F7 — currently 14 scalars, none card-aware).
- Replay logs detailed enough to imitate (B7 — summary mode strips them).
- A trustworthy A/B that can tell you whether the neural policy improved (Section B — most of these biases hit Phase 8 evaluations too).
- A scoring engine that's correct for the jokers a policy will pick (C1, C2, C5).

Address these in the order Section A → B → C → D so the foundations are correct when Phase 8 starts.

---

## Recommended fix order (highest-leverage first)

Because most of these are independent, you can take them in any order, but if I had to pick six to try first:

1. **A1** — invert / remove `_basic_blind_action_should_guard`'s play↔discard veto. Cheapest possible test of "is the search actually finding better hand actions when given the chance?"
2. **A6** — fix the shop sell-penalty so `_owned_joker_value` is in the same units as `sell_value`. Likely large win-rate impact alone.
3. **B1 + B2** — fix the win-flip and the RunTimeout retry bucket *before* trusting any new A/B from the overnight sweep.
4. **A2 + A4** — raise sample budgets to 8/16 (or 16/32 for discard) and enable the beam with the dead memo wired up. Without this, no other search fix can be measured.
5. **A7** — share the stochastic seed across competing actions in pack/consumable/hand. Variance reduction for cheap.
6. **C1** — thread `Oops! All 6s` as a probability multiplier through the evaluator's stochastic outcomes. Real bug, narrow scope.

If A1/A2/A4/A6/A7 land, the search bot should at minimum beat basic_strategy on a same-seed A/B by a clear margin. If it doesn't, the leaf value (A3) is the next thing to attack — pick one consistent leaf, calibrate, retest.

---

## What this audit did not cover (and why)

- **GUI code paths**: not reviewed.
- **Tools / preflight**: not reviewed.

---

# Addendum 1 — Local simulator + forward_sim deep dive (relaunched after rate-limit reset)

This section replaces the earlier partial Section D. Same severity tags: **H** = skews evaluation, **M** = silent drift, **L** = cosmetic. All file:line refs are absolute paths.

## D1 confirmed — generic Python `Random`, not Balatro's seed-derived RNG (H)
[local_runner.py:13](src/balatro_ai/sim/local_runner.py:13), [:287](src/balatro_ai/sim/local_runner.py:287), [:294](src/balatro_ai/sim/local_runner.py:294) — one `Random(self.seed)` drives:
- deck shuffle ([:298](src/balatro_ai/sim/local_runner.py:298), [:1821](src/balatro_ai/sim/local_runner.py:1821), [:1829](src/balatro_ai/sim/local_runner.py:1829))
- boss selection ([:1188-1193](src/balatro_ai/sim/local_runner.py:1188))
- skip tag sampling ([:392](src/balatro_ai/sim/local_runner.py:392))
- shop/joker/pack/voucher sampling (30+ call-sites)
- Misprint roll ([:724](src/balatro_ai/sim/local_runner.py:724))
- Bloodstone / Business / Reserved Parking / Space / Lucky / Glass / Gros Michel / Cavendish ([:591-605](src/balatro_ai/sim/local_runner.py:591))
- Wheel of Fortune ([:1093](src/balatro_ai/sim/local_runner.py:1093))
- Cerulean Bell forced index ([:1080](src/balatro_ai/sim/local_runner.py:1080)), Crimson Heart ([:732-740](src/balatro_ai/sim/local_runner.py:732)), Amber Acorn ([:915](src/balatro_ai/sim/local_runner.py:915))

`shop_sampler.py:5` docstring already concedes "does not try to replay Lua's seeded pseudorandom stream." A `seed=12345` local run does NOT reproduce a bridge `seed=12345` run for ANY of these.

## D2 confirmed — search rollouts inject NO stochastic outcomes (H, major)
[state_value.py:233](src/balatro_ai/search/state_value.py:233): `simulate_play(current, action, drawn_cards=drawn_cards)` — no `stochastic_outcomes` kwarg.
[hand_search.py:449](src/balatro_ai/search/hand_search.py:449): same.
[forward_sim.py:203](src/balatro_ai/search/forward_sim.py:203) `_play_outcome_mapping(None)` returns `{}`; `_outcome_int` defaults to 0.

Net effect during search rollouts:
- Misprint contributes **0 mult** (mean should be 11.5)
- Bloodstone / Business Card / Reserved Parking / Space Joker / Lucky Card all roll **0 triggers**
- Glass shatters **never happen** (so glass survives forever in the rollout — overvalues glass)
- Hook never discards from held cards (Hook EV undercounted)
- Gros Michel never dies, never spawns Cavendish

So the search systematically **undervalues stochastic mult jokers** and **overvalues Gros Michel** and glass at long horizons. This is a likely root cause of Phase 7 search not beating heuristics. Fix shape: pass mean-EV outcomes (`triggers/odds * probability_multiplier`) per rollout, using `local_runner._stochastic_play_outcomes` ([:559](src/balatro_ai/sim/local_runner.py:559)) as reference.

## B1 — Wheel of Fortune edition probabilities are ~50× too generous (H)
[local_runner.py:1092-1098](src/balatro_ai/sim/local_runner.py:1092):
```python
poll = self._rng.random()
if poll > 1 - (0.006 * 25):    # > 0.85  -> POLYCHROME (15% of all)
    return "POLYCHROME"
if poll > 1 - (0.02 * 25):     # > 0.50  -> HOLOGRAPHIC (35% of all)
    return "HOLOGRAPHIC"
return "FOIL"                   # 50% of all
```
Source: when Wheel of Fortune procs (1-in-4), edition rolls are roughly 0.3% poly / 1% holo / 2% foil / 96.7% nothing. The arithmetic above produces 15% / 35% / 50%. **Wheel-of-Fortune polychrome is ~50× more likely in local sim than in real game.** Same function is reused for Aura ([:1116](src/balatro_ai/sim/local_runner.py:1116)) — Aura should be ~uniform over three editions; this distribution is wrong for it too. Bots tuned against this will dramatically over-value Wheel of Fortune.

## B2 — `_misprint_mult_for_play` rolls one mult per HAND, not per Misprint (M)
[local_runner.py:721-724](src/balatro_ai/sim/local_runner.py:721) returns a single int regardless of how many Misprints are equipped. Source: each Misprint independently rolls 0..23. Two-Misprint stacks have mean ~23 mult; local sim gives 11.5.

## B4 — Lucky Card mult/money RNG verified correct
[local_runner.py:591-595](src/balatro_ai/sim/local_runner.py:591). 1-in-5 for +20 mult, 1-in-15 for $20 — source-correct. Each retrigger independently rolls — also source-correct. (No bug; flagging because it's worth knowing.)

## B5 — Glass cards auto-shatter every play if `extra` metadata is missing (H)
**~~Real bug~~ — FALSE POSITIVE on re-verification.** The audit agent misread `_glass_shatter_odds`. The function has `if value > 0: return value` — a 0/invalid `extra` causes the loop to continue, and the function returns `4` as the default (correct Balatro odds, ~25% per-play shatter). [local_runner.py:2626-2632](src/balatro_ai/sim/local_runner.py:2626) is correct as written. No fix needed.

## B8 — Top-level except converts simulator bugs into losses (H for rankings)
[local_runner.py:1364-1369](src/balatro_ai/sim/local_runner.py:1364):
```python
except Exception as exc:
    error = f"local_error:{type(exc).__name__}: {exc}"
    break
```
Any `KeyError`/`IndexError`/`ValueError` in joker logic becomes a "loss" with `death_reason="local_error:..."`. Bot rankings sum winrate without separating these. Should either be fail-loud during eval, or surfaced separately in summaries.

## B9 — Only white stake is correctly modeled (H)
[local_runner.py:1586-1590](src/balatro_ai/sim/local_runner.py:1586) `_parse_stake` returns `Stake.UNKNOWN` on bad input — but more importantly, greps for "eternal", "perishable", "rental", "boss_blind_score_x" return nothing in local_runner. Red/Green/Black/Blue/Purple/Orange/Gold stakes are parsed but never modulate gameplay. Bots evaluated on higher stakes in local sim are essentially still on white. Phase 13 ("High-stakes specialization") cannot start with this simulator.

## B13 — Soul / Black Hole excluded entirely from packs (H)
**Real bug. FIXED 2026-05-15.** Verified against Balatro source [common_events.lua:2401-2410](C:/Users/Wyatt/AppData/Roaming/Balatro/Mods/lovely/dump/functions/common_events.lua) — Soul rolls 0.3% per slot in Tarot/Spectral/Tarot_Planet pools, Black Hole 0.3% in Planet/Spectral pools. Spectral packs roll both; Black Hole wins ties. Soul/Black Hole only spawn if not already used unless Showman is owned. The rolls are also gated on `soulable=true`, which source passes only for pack creation (not for tarot-effect creates like The Fool, Wraith, Hallucination, etc.). [shop_sampler.py:382-440](src/balatro_ai/search/shop_sampler.py:382) now implements `_maybe_legendary_spectral_payload` in `_sample_pack_card` matching this.

100k-sample distribution check vs source 0.3% target:
- Arcana: Soul 0.306%, BH 0.000% ✓
- Celestial: Soul 0.000%, BH 0.297% ✓
- Spectral: Soul 0.294%, BH 0.310% ✓
- Standard / Buffoon: zero legendaries ✓ (correctly gated out)

**Known minor gaps in the fix (not addressed)**:
1. **Omen Globe Arcana→Spectral override.** When Omen Globe is owned and its 20% trigger fires, the slot is converted to Spectral in source AND then runs Soul/BH rolls. My code returns early on the Omen Globe path before the legendary check — so that slot won't roll legendaries. Source-correct fix would route the converted slot through the legendary check.
2. **Consumable-slot dedupe.** Source `G.GAME.used_jokers['c_soul']` is true while any Soul card exists, including in the consumable slot. My dedupe checks only `state.jokers` (legendaries spawned by Soul) and `state.modifiers["used_jokers"]` (never set). If you hold an unused Soul and open another Arcana pack, my code could roll a second Soul. Niche.

Both gaps are low-frequency. The headline behavior — packs go from 0% Soul/BH to ~0.3% per slot at source-correct rates — is correct.

Original finding: `_pool_records` filters out `Black Hole` and `The Soul` from Tarot/Planet/Spectral pools. Source: both appear at 0.3% rate per slot in Arcana / Spectral / Celestial packs. Excluding them meant The Soul (legendary joker spawn) was unreachable; late-game legendary EV undercounted; pack value across the board was wrong.

## B14 — Replay validator overrides hand/deck/jokers before comparing (H for validator confidence)
[replay_validator.py:461-497](src/balatro_ai/sim/replay_validator.py:461) `_with_observed_validation_modifiers` replaces simulated `deck_size, hand, known_deck, jokers, hand_levels` with observed values **before** comparing the post-state. So the validator only checks money/score/phase deltas, not deck or hand evolution. A simulator bug producing wrong cards in hand won't be caught — masked by the override.

Combined with **B15**: [replay_validator.py:277, 299, 324-328](src/balatro_ai/sim/replay_validator.py:277) — on any divergence, `sim_state = observed_pre/post`. Compound bugs that drift over multiple transitions are invisible. The 99.9% headline number is single-transition match rate, not run-trajectory fidelity.

## Skip-tag and voucher specific findings

- **B6 — Skip Tag payout off by one (M)**: [local_runner.py:796](src/balatro_ai/sim/local_runner.py:796) `5 * skips` reads `modifiers["skips"]` *after* `_skip_blind` already incremented it. Source pays $5 × *prior* skips; local pays one too many.
- **Handy Tag undercounts (H)**: [:798](src/balatro_ai/sim/local_runner.py:798) falls back to per-blind `hands_played` rather than run-total. Source pays $1 × total hands across run.
- **Garbage Tag wrong basis (M)**: [:800](src/balatro_ai/sim/local_runner.py:800) reads current `discards_remaining`; source uses run-total unused discards.
- **Coupon Tag misses vouchers (M)**: [:954-957](src/balatro_ai/sim/local_runner.py:954) zeros buy cost on `shop_cards` + `booster_packs` but not on the voucher slot.
- **Orbital Tag wrong distribution (M)**: [:806](src/balatro_ai/sim/local_runner.py:806) rolls hand-type at consumption time via `rng.choice(_base_hand_levels())`; source picks at mint time and is biased differently.
- **`shop_edition_tag_consumed` is dead (L)**: [forward_sim.py:415](src/balatro_ai/search/forward_sim.py:415) reads `state.modifiers.get("shop_edition_tag_consumed")` — grep shows nothing ever writes that key. Dead gate.

## Boss-blind specific findings

- **The Pillar (M)**: `card.metadata` "played this ante" flag at risk of being dropped through `_round_eval_deck_after_play` / known_deck reconstruction ([forward_sim.py:1045](src/balatro_ai/search/forward_sim.py:1045)).
- **Crimson Heart (M)**: [local_runner.py:732-740](src/balatro_ai/sim/local_runner.py:732) picks disabled joker via `rng.choice` over non-disabled; source rotates each hand differently — biased distribution.
- **Cerulean Bell (M)**: [:1075-1082](src/balatro_ai/sim/local_runner.py:1075) re-rolls forced index on every `_play_or_discard` rather than once per blind.
- **The Arm (M, likely unimplemented)**: not visible in local_runner; should lower hand level on play. Search the evaluator before assuming it's covered.
- **The Tooth (M)**: $1 penalty per played card — not visible in local_runner; needs verification.
- **Mr. Bones destroys only first (L)**: [forward_sim.py:3727](src/balatro_ai/search/forward_sim.py:3727) destroys only the first Mr. Bones; source destroys all of them.
- **Mail-In Rebate rank rerolls per blind (M)**: [forward_sim.py:1276](src/balatro_ai/search/forward_sim.py:1276) picks a new target_rank every blind select; source fixes the rank when the joker is acquired and keeps it for the run.

## Determinism — verified
[local_runner.py:391-392](src/balatro_ai/sim/local_runner.py:391) and other critical paths iterate tuples and ordered dicts, not sets. Same-seed-same-process runs reproduce. Cross-process also OK (Python's `Random(int)` is portable). The non-determinism here is *between* local-sim and bridge, not within local-sim.

## Items the audit could not verify from these files
- Whether `RunResult.death_reason` containing `local_error:*` is excluded from winrate denominators in eval summaries.
- Observatory stacking-multiplier behavior.
- Retcon respect for `boss_rerolls_unlimited` in `_boss_reroll_available`.
- The Arm / The Tooth implementations (likely in hand_evaluator).
- The Eye / The Mouth per-blind hand-type tracking correctness.

## Local-sim addendum: recommended fix order
1. **D2** — wire `StochasticPlayOutcomes` (mean EV) into search rollouts. Biggest expected payoff.
2. **B5 / glass shatter** — clamp `_glass_shatter_odds` to a default of 4 instead of 0.
3. **B1 / Wheel of Fortune** — fix edition probability thresholds (`poll < 0.003 → POLY; < 0.013 → HOLO; < 0.033 → FOIL`).
4. **B13** — restore Soul and Black Hole at source-correct weights.
5. **B14** — stop overriding hand/known_deck/jokers in the validator so it actually validates them.
6. **B8** — surface `local_error:*` death-reasons separately from blind losses in winrate aggregation.
7. **B9** — implement at minimum the boss-blind score scaler and eternal/perishable joker mechanics for red+ stakes, or refuse to run higher stakes in local sim.

---

# Addendum 2 — Test quality (relaunched after rate-limit reset)

This section replaces and expands Section E. **Bottom line: the test suite is unusually behavior-rich for a Python project.** The dominant pattern is "construct a fully-specified `GameState`, invoke the function, assert exact numbers or exact action shapes." Smoke testing is the exception, not the rule.

## E-A. Behavior vs smoke — the suite is genuinely deep
Across 13 audited test files, tests almost universally assert exact integers, tuples, or `metadata` keys. Sampling `assertIsNotNone` / bare `assertTrue` usage in test_basic_strategy_bot.py (4942 lines) shows only ~24 occurrences, and every one I sampled is followed by a stronger assertion on the same object. [test_basic_strategy_bot.py:464](tests/test_basic_strategy_bot.py:464) `test_straight_draw_eval_rewards_open_ended_outs_over_gutshot` is representative — `assertIsNotNone` + `assertGreater(out_count, ...)` + `assertGreater(completion_probability, ...)`. [test_shop_search.py:132-135](tests/test_shop_search.py:132) asserts action type AND amount AND a search-tag metadata key.

You can refactor with confidence that the tests are checking real behavior.

## E-B. Scoring edge case coverage in `test_hand_eval.py` — dense
Covered with exact-score assertions:
- Flush Five ([test_hand_eval.py:39](tests/test_hand_eval.py:39))
- Stone Card under suit-debuff ([:198](tests/test_hand_eval.py:198))
- Vampire-stripped Hanging Chad retrigger ([:276](tests/test_hand_eval.py:276))
- Red-seal retrigger of Steel + Baron at `1.5**6` ([:586](tests/test_hand_eval.py:586))
- Sock and Buskin copied through Blueprint ([:596](tests/test_hand_eval.py:596))
- Holo + Polychrome edition stacking ([:376](tests/test_hand_eval.py:376), [:856](tests/test_hand_eval.py:856))
- Ramen / Loyalty / License text-fallback variants ([:1207-1287](tests/test_hand_eval.py:1207))
- Triboulet 5-card KQQQQ at xmult=32, score=11520 ([:1061](tests/test_hand_eval.py:1061))

Fixtures (`tests/fixtures/score_edges/`): Flush House at 2660, Flush Five at 3440, Five-of-a-Kind 2100, large XMult chains.

## E-C. Real gaps in scoring tests (where regressions could slip)
- **Purple seal**: zero hits across all files. Purple-seal planet-on-discard would silently regress.
- **Negative editions**: appear only as a name string ([test_basic_strategy_bot.py:77](tests/test_basic_strategy_bot.py:77)); no scoring assertion.
- **Deep retrigger stacks**: Sock + Sock + Photograph at depth 3 covered ([test_hand_eval.py:596](tests/test_hand_eval.py:596)), but no test for Hack + Seltzer + Sock stacked together.
- **Edition × seal × retrigger cross product**: a few pairs tested; most are not.
- **Card enhancement edges fixture has only 4 cases** for a game with ~10 enhancement types.

## E-D. Hardcoded score numbers will rot with legitimate changes
Tests like [test_hand_eval.py:881](tests/test_hand_eval.py:881) (`score=27056`), [:1113](tests/test_hand_eval.py:1113) (`score=14219`), [:1061](tests/test_hand_eval.py:1061) (`score=11520`) assert exact final integers with no tolerance and (mostly) no comment showing the arithmetic. A legitimate scoring-ordering fix that legitimately changes a number by one chip causes test failure without context. Add either tolerance bands or a one-line `# 32 * 360 = 11520` derivation.

## E-E. Bot-decision test depth — high
Random sample of 10 tests across test_basic_strategy_bot.py (the 4942-line monster):

| Line | Class |
|---|---|
| 26 plays_when_best_hand_beats_remaining_score | **deep** — asserts exact `card_indices=(0,1,2)` |
| 50 rearranges_jokers_when_order_improves_score | **deep** — asserts REARRANGE + exact indices |
| 504 straight_hunt_uses_known_top_draw_outs | **deep** — exact DISCARD + reason substring |
| 1784 two_pair_support_joker_does_not_force_two_pair_plan | **deep** — asserts `_preferred_hand_type` returns FULL_HOUSE |
| 1817 two_pair_planet_is_discounted_without_dedicated_scaling | **edge** — relative `_planet_card_value` delta |
| 2956 late_shop_prioritizes_missing_xmult_role | **deep** — exact BUY + audit `missing_roles` |
| 3909 projected_discard_score_reuses_equivalent_hand_content | **edge** — caching invariance via `patch.object(wraps=)` |
| 3941 projected_discard_score_decrements_banner_discard_count | **deep** — exact projected score = 244 |
| 4011 the_eye_avoids_repeating_played_hand_type | **deep** — multi-step state, exact `card_indices` |
| 4931 consumable_room_uses_slot_limit_modifier | **shallow** — boolean only |

Roughly **7 deep / 2 edge / 1 shallow** in the sample. Concentration is in shop logic (2056 lines of shop tests at 1182-3700). **Hand-selection branches under boss blinds beyond The Eye / The Mouth / Card Sharp are sparsely tested** — that's the biggest gap in bot tests.

## E-F. Local-sim joker procs — ~8 tests vs hundreds of jokers
[test_local_runner.py:602](tests/test_local_runner.py:602) Misprint, [:620](tests/test_local_runner.py:620) Blueprint-copied Space Joker, [:637](tests/test_local_runner.py:637) Oops! All 6s with Wheel boss, [:717](tests/test_local_runner.py:717) Wheel of Fortune edition polls, [:731](tests/test_local_runner.py:731) Crimson Heart rotation. About 8 joker-proc tests total. The much deeper coverage is in `test_forward_sim.py` (2593 lines): Ox / Pillar / Green / Egg / Popcorn / Lucky-Cat / Obelisk / Burnt / Mail-In-Rebate / Trading-Card / Ramen / Constellation / Satellite / Rocket / To-Do-List / Gift-Card across [:127-1100](tests/test_forward_sim.py:127). Local runner is mostly **transition correctness**; joker math is tested in `test_forward_sim.py`.

## E-G. Hidden bridge dependencies — none in audited files
Of the files audited, none require a running bridge. Bridge-shaped dicts are constructed inline. (Files NOT audited: `test_bridge_joker_smoke.py`, `test_balatrobot_schema.py`, `test_run_seed.py` — likely do require a bridge.)

## E-H. Fixture quality — comprehensive in shape, thin in volume
9 fixture files, 992 lines, ~50 cases.
- `rare_hand_types.json` — 7 cases, hits exactly the rare hands you'd want (Flush House / Flush Five / Five-of-a-Kind / Four Fingers / Shortcut / Splash / Stone).
- `retrigger_and_xmult_edges.json` — 7 cases, deepest is Triboulet xmult=32.
- `card_enhancement_edges.json` — only 4 cases for ~10 enhancement types.
- `known_gaps.json` — 4 cases explicitly not tested. Honest hygiene.

**Sparse**: zero fixture cases for purple seal, negative joker edition, or holographic-without-polychrome.

## E-I. Test isolation & flake risk — both genuinely low
- Every search test passes explicit `seed=` ([test_hand_search.py:31, 62, 86, 112](tests/test_hand_search.py:31); [test_state_value.py:58-272](tests/test_state_value.py:58)).
- `FixedRandom` deterministic stub ([test_local_runner.py:64-69, 728](tests/test_local_runner.py:64)).
- No `time.` / `datetime` / `sleep(` references.
- Module-level caches scoped via `with strategy.decision_cache_scope():` ([test_basic_strategy_bot.py:3931, 4001](tests/test_basic_strategy_bot.py:3931)) and `with state_values.state_value_cache_scope():` ([test_state_value.py:159, 195](tests/test_state_value.py:159)).
- `test_runner.py` mutates `runner._run_seed` / `_endpoint_is_healthy` / etc. but every test uses `try/finally` to restore originals ([test_runner.py:54-60, 209-211, 267-270, 305-306](tests/test_runner.py:54)). Correct but fragile if a future test forgets the pattern.

## E-J. Real regressions that could still slip through
1. **Purple seal scoring** — no test exists anywhere.
2. **Deep retrigger × negative edition × seal crossproduct** — undertested.
3. **Hardcoded scores in test_hand_eval.py** — fragile to legitimate scoring-order improvements.
4. **`test_score_edge_fixtures.py:13](tests/test_score_edge_fixtures.py:13)** only asserts `len(results) >= 10`. Adding broken fixtures that error before reaching 10 is undetected if other fixtures error gracefully.
5. **basic_strategy_bot at 4942 test lines for 9615 source lines** — ratio is fine for the *covered* areas; the gap is hand-selection under bosses other than The Eye / Mouth / Card Sharp.

## E-K. Updated test recommendations
- Add purple-seal fixtures (consumable creation EV is currently invisible).
- Add a fuzz test (Python `hypothesis` package, only dev dep) on `evaluate_played_cards`: random card + joker + boss combinations should never error and should agree with itself under joker-reordering invariants where source allows.
- Add tolerance bands or arithmetic comments next to hardcoded score numbers in `test_hand_eval.py`.
- Replace `len(results) >= 10` with `len(results) == EXPECTED_COUNT` in `test_score_edge_fixtures.py`.
- Add bot tests for each boss-blind beyond Eye / Mouth / Card Sharp — particularly The Plant / Verdant Leaf / The Pillar / The Ox / Cerulean Bell.

---

# Addendum 3 — Shop joker dedupe (new finding, 2026-05-15)

**Status: RETRACTED.** Initially flagged as a real bug; on re-read of the Lua source and user pushback, the original claim was wrong. Notes below are kept for context.

## Original (incorrect) claim
I claimed Balatro's dedupe is run-wide ("every joker ever seen this run is gone forever unless Showman"). The user disputed: rerolling a shop can re-show a joker you just saw.

## What the source actually does
The reroll flow at [button_callbacks.lua:2924-2939](C:/Users/Wyatt/AppData/Roaming/Balatro/Mods/lovely/dump/functions/button_callbacks.lua) destroys all current shop cards **before** sampling new ones:

```lua
for i = #G.shop_jokers.cards,1, -1 do
    local c = G.shop_jokers:remove_card(G.shop_jokers.cards[i])
    c:remove()
    c = nil
end
-- then create_card_for_shop for each empty slot
```

And the destruction path at [card.lua:5164-5168](C:/Users/Wyatt/AppData/Roaming/Balatro/Mods/lovely/dump/card.lua) clears `used_jokers[key]` when no other copy exists:

```lua
if not G.OVERLAY_MENU then
    if not next(SMODS.find_card(self.config.center.key, true)) then
        G.GAME.used_jokers[self.config.center.key] = nil
    end
end
```

So `used_jokers[key]` is set while a card exists, and cleared when the last copy is destroyed. The dedupe rule is **"this joker currently exists somewhere in the game world"**, not **"this joker has ever appeared"**. After a reroll, the old shop cards are destroyed, their keys are cleared, and the same joker can re-appear in the new shop.

## What Botlatro actually does (correct)
[shop_sampler.py:849-855](src/balatro_ai/search/shop_sampler.py:849) `_used_joker_identifiers` collects currently-owned `state.jokers`. That correctly matches "currently exists":
- Intra-shop dedupe: `sample_shop` grows the set during sampling. ✓ matches Balatro
- Owned-joker dedupe: `state.jokers` is read fresh on each call. ✓ matches Balatro
- Reroll: `sample_shop` is called again with `_used_joker_identifiers` recomputed → clean slate (no destroyed shop cards in `state.jokers`). ✓ matches Balatro

So Botlatro's headline behavior is correct. The only remaining minor gaps:
1. Botlatro doesn't cross-check jokers in currently-OPEN packs against shop sampling (edge case — they're not usually open simultaneously).
2. My Soul/Black Hole patch dedupes against `state.jokers` only, not `state.consumables`. If you hold an unused Soul and open another Arcana pack, Botlatro might roll a second Soul where Balatro wouldn't. Niche.

## What the audit body originally claimed (wrong, for the record)
[card.lua:474](C:/Users/Wyatt/AppData/Roaming/Balatro/Mods/lovely/dump/card.lua) inside `Card:set_ability`:
```lua
if not G.OVERLAY_MENU then
    if self.config.center.key then
        G.GAME.used_jokers[self.config.center.key] = true
    end
end
```
This runs every time a card is created with an ability — meaning **every joker that has ever appeared in this run** (shop slots, pack contents, reroll re-rolls, joker-spawned jokers, blind reward jokers) is recorded in `G.GAME.used_jokers`. The dedupe checks at [common_events.lua:2273, 2402, 2408](C:/Users/Wyatt/AppData/Roaming/Balatro/Mods/lovely/dump/functions/common_events.lua) then exclude any joker whose key is already in that set, unless Showman is owned.

So Balatro's dedupe is **run-wide**: a joker you skipped in shop ante 1 won't reappear at ante 5 unless Showman.

## What Botlatro does
[shop_sampler.py:849-855](src/balatro_ai/search/shop_sampler.py:849) `_used_joker_identifiers` collects:
- All currently-owned `state.jokers` names + their metadata `key`.
- Plus whatever is in `state.modifiers.get("used_jokers")`.

The dedupe filter at [shop_sampler.py:217, 223, 370, 376](src/balatro_ai/search/shop_sampler.py:217) uses this set. Within ONE shop or pack open, the set grows as cards are emitted (so the same joker won't appear twice in a single 2-slot shop or a 5-card buffoon pack).

But `state.modifiers["used_jokers"]` is **never written by Botlatro**:
- Grep returns zero writers in `local_runner.py`, `forward_sim.py`, `api/state.py`.
- The bridge state parser doesn't extract this field either.

So Botlatro's dedupe is **only intra-shop / intra-pack**, plus "what you currently own." Anything you previously owned and sold, or saw in a shop and skipped, or saw in a pack and passed on, is **forgotten** the next time the sampler runs.

## Impact
Differences from Balatro behavior:
- **Reroll fishing is too easy in local sim.** The same joker can appear over and over across rerolls of the same shop visit (across different `sample_shop` calls), and across different shop visits within the same run. In real Balatro, once a joker has been seen, it's gone for the run.
- **Pack joker variety is too high in local sim.** A buffoon pack at ante 5 in real Balatro has a much narrower pool than at ante 1 (because most jokers have already been seen). Botlatro shows the same pool every time.
- **The rule bot is mis-trained** to value rerolls higher than it should and to under-value "buy this joker now because you may never see it again."
- **Search bot reroll EV** is also inflated — the `reroll_ev` calculation in shop_search will overestimate.

## How to fix
Two pieces:

1. **Write `state.modifiers["used_jokers"]` whenever a joker is created.** Locations:
   - [local_runner.py](src/balatro_ai/sim/local_runner.py) shop sampling (after `sample_shop` returns).
   - Pack-card sampling (after `sample_pack_contents`).
   - Any code path that creates a joker via Hallucination / Riff-raff / Showman buff / Wraith / Soul / etc.
   - Bridge-side state parsing should ALSO try to extract `G.GAME.used_jokers` from the bridge JSON (if it's exposed) so local-sim and bridge see the same dedupe.

2. **`used_jokers` should persist across cash-out / select_blind / reroll transitions** — it's a run-wide modifier. Most `replace(state, modifiers={...})` calls in `forward_sim` preserve existing modifiers, so as long as the writes use `set | new_keys` style updates, persistence should be automatic.

The fix is small (~30 lines), but it touches the local_runner + forward_sim + possibly bridge state parsing. Worth doing before re-running any overnight sweep that involves reroll-heavy tuning.

## Severity
Medium-to-high. Not on the "blocks white-stake winrate" critical path (the rule bot doesn't fish-roll explicitly), but it inflates reroll EV in the search bot and gives the rule bot a fictional "I'll see this joker again next shop" assumption that doesn't hold on bridge. Mostly a **fidelity bug** that distorts cross-run reasoning.

---

# Post-verification status (2026-05-15)

After 1:1 verification of every Tier 1-2 item against the actual code (and the Balatro Lua dump for B13 and the new dedupe finding), here is the canonical current state.

## Fixed (verified post-fix)
| # | Issue | Location | Verification |
|---|---|---|---|
| B2 | RunTimeout in retry bucket biases A/Bs toward fast bots | [runner.py:232](src/balatro_ai/eval/runner.py:232) | Logic: RunTimeout is wall-clock-driven, retry just re-rolls budget. Fix excludes it; non-timeout bridge errors still retry. Tested. Trade-off accepted: loses recovery on rare legitimate bridge stalls; eliminates A/B bias toward fast bots. |
| B3 | Wilcoxon p-values were OVER-conservative under tie-heavy ante deltas | [compare.py:301](src/balatro_ai/eval/compare.py:301) | Hand-computed: uncorrected variance 96.25 → p=0.2845; corrected variance 75.625 → p=0.2273 (n=10, all ties at |d|=1, 7+/3-). Tie correction reduces variance, increases \|z\|, decreases p. ⚠️ Original audit framing called this "anti-conservative" — that was inverted; the bug was hiding real effects, not over-claiming them. The fix still right, the framing was wrong. |
| B4 | config_sweep had no multiple-comparisons correction | [config_sweep.py:285](src/balatro_ai/eval/config_sweep.py:285) | Holm verified against 5 textbook cases including canonical (0.01, 0.04, 0.03, 0.005) → (0.03, 0.06, 0.06, 0.02), all-equal, monotone-clamp at 1, single-value, empty. |
| B13 | Soul/Black Hole unreachable in local-sim packs | [shop_sampler.py:382](src/balatro_ai/search/shop_sampler.py:382) | Lua source `common_events.lua:2401-2410` confirms 0.3% per slot, gated on `soulable=true` which source only passes for pack contents. 100k-sample distribution check: Arcana 0.306% Soul / 0% BH, Celestial 0% / 0.297%, Spectral 0.294% / 0.310% — all within sampling error of 0.3% target. Standard/Buffoon: 0 leaks. Two known minor gaps (Omen Globe Spectral override; consumable-slot dedupe) — both niche, not headline. |

## False positives (no action needed)
| # | Original claim | Why it's not a bug |
|---|---|---|
| B1 | Ante-9 win-flip can convert stale states to wins | Intentional per PROGRESS.md 2026-05-02 — nonterminal `ante=9` is a white-stake win |
| B5 | Glass auto-shatters when metadata is missing | `_glass_shatter_odds` defaults to `4` not `0`; the `if value > 0` filter skips zero-extra entries |
| C1 | Oops! All 6s does nothing | Implemented via `probability_multiplier` modifier set in [forward_sim.py:2640](src/balatro_ai/search/forward_sim.py:2640); all `_roll_odds` call sites honor it |
| C2 | Pareidolia + Triboulet / Baron / Shoot the Moon | Balatro source uses `base.id` rank checks, not `is_face` checks, for these three; current code matches |
| WoF | Wheel of Fortune edition probabilities are 50× too generous | Function is conditional on the 1-in-4 proc, not unconditional. Source distribution-conditional-on-proc not separately verified |
| D5 | Shop joker dedupe missing run-wide tracking | Balatro's dedupe is "joker currently exists in game world," not "joker ever seen." Reroll destroys shop cards (clearing dedupe) before re-sampling. Botlatro's `_used_joker_identifiers` reading `state.jokers` correctly matches this. |

## New finding from re-verification
None — the "shop dedupe is wrong" hypothesis (D5) was retracted after the user pointed out that rerolls can re-show recently-seen jokers in real Balatro. Source confirmed: dedupe is "currently exists" not "ever seen." Botlatro's current behavior matches. See retracted Addendum 3.

## Still open (not yet verified or fixed)
From the original report body. **These have NOT been re-verified, so apply the same skepticism that turned up 4 false positives above.** If you act on any of these, spot-check the code first.

- **A1-A15 search architecture findings** — entire Section A is unverified. Likely many are real, some are over-flagged.
- **B6** `PYTHONHASHSEED` not set in `benchmark.py` / `compare.py` — likely real, easy fix.
- **B7** Summary replay mode silently disables replay_diff — likely real.
- **B8** Seed pool not nested across sizes — real.
- **B9** Retry deletes original replay without audit trail — real.
- **B10** Endpoint re-queued when park fails — likely real.
- **B11** `compare.py` runs bot A then bot B sequentially — real (design tradeoff).
- **B12** Local-sim vs bridge are not interchangeable — partially structural, partially fixable.
- **C3** Misprint defaults to 0 mult — likely real if it matters.
- **C4** Joker-ordering interleaving for Stencil-left-of-Photograph — edge case, low impact.
- **C5** Float overflow has no clamp — real but rarely hit at white stake.
- **C6** Purple seal unimplemented — real.
- **C7** Cavendish missing death-roll side effect — real.
- **C8** Sixth Sense / DNA / Mail-In Rebate timing — unverified.
- **D1** Local sim RNG is `random.Random`, not Balatro's seed-derived RNG — real and structural (per `shop_sampler.py:5` docstring).
- **D2** Search rollouts inject no stochastic outcomes — real (verified during re-audit).
- **D3** local_runner uses python.random not Balatro's seed RNG — same as D1.
- **D4** Skip-tag bugs (Skip / Handy / Garbage / Coupon / Orbital off-by-one or wrong basis) — likely real, individually small.
- **B14** Replay validator overrides hand/known_deck/jokers before comparing — likely real.
- **B15** Validator resyncs on every divergence — likely real.
- **F1** `Card.from_mapping` can produce `rank="None"` silently — real, edge case.
- **F7** Observation tensor is only 14 scalars — real, blocks Phase 8.

## Recommended next moves
1. **Re-run the overnight parameter sweep** under the new eval (`Holm p` column, no RunTimeout retry, tie-corrected Wilcoxon).
2. **Pin a fresh bridge baseline** at current `basic_strategy_bot` so local-sim and bridge winrates can be compared apples-to-apples.
3. ~~Fix Addendum 3 (shop joker dedupe)~~ — retracted, no fix needed.
4. **Spot-verify items in Section A and the "still open" list before acting on them.** The agent over-flag rate so far is ~50%.

The audit report body is preserved unchanged below the header for historical reference, but the canonical truth lives in this final section.
