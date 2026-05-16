# Basic Strategy Bot Layout

`basic_strategy_bot.py` owns the decision flow for `BasicStrategyBot`. The
`basic_strategy/` package holds extracted helpers. Private helper names are
still imported back into `basic_strategy_bot.py` so existing tests and trace
tools that import `strategy._foo` keep working.

## Decision Flow

`BasicStrategyBot.choose_action()` installs the active `BotConfig` and opens a
decision cache for one action choice. The uncached flow is:

1. Sync per-shop and per-blind memory.
2. Select blinds and cash out when those actions are available.
3. Evaluate shop actions, including replacement sells, rerolls, packs,
   vouchers, consumables, and end-shop choices.
4. Resolve booster pack choices.
5. Use held consumables when they have positive value.
6. In blinds, consider joker rearrangement, best play, tactical discards, and
   final play/discard reasons.

## Module Responsibilities

`actions.py` contains generic `Action` helpers: selecting the first action of a
type, annotating actions with reasons/audit payloads, target-index rewrites, and
memory keys.

`ante_one_hunt.py` owns the ante-1 small-blind discard hunt. It decides when the
bot should spend early discards looking for a one-hand clear and handles the
near-clear upgrade edge case where a weak winning five-card hand is discarded to
draw higher-value flush or straight cards.

`banner_policy.py` contains the Banner-specific discard veto. It compares the
projected benefit of discarding against the future chip tax from lowering
Banner's remaining-discard count and can preserve the current play when a
discard would make the blind less likely to clear.

`blind_state.py` contains tiny blind-state predicates shared by blind helpers,
such as detecting whether the current blind is a boss blind.

`blind_tactics.py` is the top-level blind play/discard router. It computes the
current best-play score and orders tactical branches such as winning economy
hunts, first-blind hunts, joker setup, preferred-hand hunts, Banner vetoes,
panic/safety/chase discards, and final play annotations.

`blind_setup.py` contains early-blind setup policies that deliberately spend a
play or discard for joker value while preserving a safe clear path, including
DNA/Sixth Sense single-card setup plays, Mystic Summit discard activation, and
other joker-triggered strategic discards.

`build_profile.py` computes the run's current build profile from owned jokers,
money, and preferred hand signals. It owns joker role scores, late durability
discounts for perishable/decaying/finite jokers, archetype labeling, owned-role
replacement value, and late reroll role-hunt urgency.

`build_scoring.py` estimates build strength from representative sample hands,
the visible hand, and joker order-independent score projections. It owns
sample-hand sets by preferred archetype, Card Sharp repeat projection, Joker
Stencil slot-adjusted buy/sell joker sets, and helper deltas used by shop
pressure, joker valuation, planets, and packs.

`blind_reasons.py` formats trace/reason strings for blind play and discard
actions. It should stay descriptive only: no action selection and no strategy
tuning beyond reporting the scores, draw details, and triggered joker labels
computed elsewhere.

`cache.py` owns decision-scoped caching. It should not contain strategy rules;
it only memoizes expensive state, card, joker, and score-key calculations during
one `choose_action()` call.

`cards.py` adapts raw shop card dictionaries and simulator card/joker objects.
It contains card labels/costs, card categories, joker and consumable slot
limits, rank/suit parsing, edition bonuses, and conversion from shop payloads
to `Joker` objects.

`data.py` contains static tables: sample hands, hand-family maps, joker role and
ordering tables, tarot/spectral/voucher values, boss sets, and score baselines.
Changing this file changes strategy tuning but should not add control flow.

`draw_math.py` contains pure probability and rank/straight helpers used by draw
planning. It should stay independent of bot policy: no shop pressure, no money
logic, and no action selection.

`economy_hunt.py` scores small economy side-goals during blind play, including
held gold cards, blue/gold seals, Delayed Gratification cash-out value, and
drawn-card value for winning discard lines.

`discard_state.py` projects state after discards for known-deck and heuristic
discard decisions. It owns draw-count rules, The Serpent draw-three handling,
Trading Card/Burnt Joker discard side effects, round-discard counters, and
joker state updates triggered by discarded cards.

`discard_policy.py` chooses generic discard actions during blinds. It owns
panic, safety, chase, last-hand hunt, discard detail limits, discard
playstyle bonuses, hand-pace math, and the top-level `_best_discard_action`
selector used by tactical blind flow.

`draw_evaluation.py` evaluates draw shapes for preferred-hand hunting. It scores
straight, flush, rank, and full-house completion lines, estimates outs and
probabilities, formats draw-reason details, and maps preferred hands to their
acceptable hand families.

`hand_models.py` contains small dataclasses used by play and draw planning, such
as `_PlayCandidate`, `_BlindContext`, and draw-evaluation records.

`hand_preferences.py` decides the bot's preferred hand family from jokers, hand
levels, advanced-hand viability, and dedicated pair/two-pair signals. Shop,
pack, play, and discard logic use this as a shared description of the build the
run appears to be moving toward.

`hand_value.py` scores cards and kept-card groups for discard and redraw
decisions. It owns keep-score ordering, straight-draw potential, long-term card
value, and joker-specific card retention bonuses such as Baron, Wee Joker, and
Mail-In Rebate.

`held_consumables.py` chooses whether to use already-held consumables in the
shop flow and values planets, Black Hole, target-required tarot cards, storage
generators, Judgement, and spectral cards using the same card valuation helpers
as visible shop and pack choices.

`jokers.py` contains joker metadata and effect helpers: disabled/sticker checks,
current chip/mult/xmult parsing and mutation, metadata lookup, role
classification, static role scores, and lightweight state helpers such as
Castle target suit and Mail-In Rebate rank. It should stay below action policy;
shop-specific value judgments about whether a joker is good remain in
`basic_strategy_bot.py` until the joker valuation area is split as a unit.

`joker_ordering.py` searches joker permutations when order can affect score,
including Blueprint/Brainstorm copy positions and chips/mult/xmult ordering. It
depends on `play_scoring.py` for candidate scoring and returns only a
`REARRANGE` action when the score gain clears the small configured threshold.

`pack_targets.py` decides whether a booster card can be picked and which hand
cards target-required tarot effects should select. It handles suit-conversion
tarots, rare-hand support exceptions, and joker-slot overfill checks for pack
choices.

`pack_choice.py` owns booster-opened action selection: picking the best visible
pack card, preserving targeted card indices, producing stale-pack `NO_OP`
fallbacks, and valuing Red Card pack skips without owning broader pack pricing.

`play_scoring.py` turns legal play actions into `_PlayCandidate` records, scores
them against current boss restrictions, applies hand-sequencing bonuses, and
chooses the best immediate play. Tactical blind policy still lives in
`basic_strategy_bot.py`; this module only evaluates and orders play actions.

`preferred_hunt.py` owns preferred-hand hunt policy in blinds: deciding when to
discard or burn a redraw hand to chase the build's preferred hand family, how
many discard actions to inspect, what cards to protect, and whether projected
draw lines are safe enough under the current blind.

`profile.py` contains shop-pressure and build-profile containers plus payload
formatters for traces. It describes what the bot has and what roles it is
missing; `build_profile.py` computes the concrete role scores.

`rare_hands.py` identifies and scores support for rare-hand plans such as Four
of a Kind, Five of a Kind, Flush Five, and Flush House. It also contains the
rare-hand tarot support checks and visible rank-target helpers used by joker,
planet, tarot, and card valuation.

`score_projection.py` estimates scores after hypothetical discard and draw
lines. It owns projected discard scoring, best-score-from-cards caching,
optimistic completion hands, cache keys for scoring states, and cheap completion
card builders used by draw planning.

`shop_flow.py` owns the top-level shop decision flow used by
`BasicStrategyBot`: replacement sells, best visible shop action selection,
information-first pack sequencing before planned joker buys, pressure-forced
spend actions, end-shop annotations, and shop decision audit payloads.

`shop_items.py` finds visible shop, voucher, booster, and pack items for an
action and formats action/item payloads for traces. It does not score actions.

`shop_jokers.py` owns joker valuation for shop buys, replacement sells, owned
joker value, pressure-driven role bonuses, future score headroom, Joker Stencil
slot projections, Red Card caps, Hallucination utility, and conflict penalties
for narrow or unsupported joker plans.

`shop_cards.py` owns shop and pack-card valuation for non-jokers: planets,
Black Hole, tarots, spectral-adjacent tarot support values, and playing cards.
It also computes planet capacity gain and hand/joker alignment so consumable
pricing can respond to the current build instead of static card names alone.

`shop_money.py` owns shop money policy: interest caps, money-scaling reserves,
late pressure reserve caps, spendable money, money-gain value, and cost/interest
penalties used by buys, vouchers, packs, rerolls, and audit payloads.

`shop_packs.py` owns booster-pack valuation and late-shop pack gating. It
scores pack kinds, rare-hand pack support, late pack-open limits, capacity gain
from celestial/buffoon/arcana/standard/spectral packs, and the minimum capacity
gain needed before spending late money on another pack.

`shop_forecast.py` estimates upcoming blind pressure for shop decisions. It
parses upcoming boss metadata, projects next/final required scores, applies boss
capacity factors, handles exact Needle/Water shop-pressure state shaping, and
models final-boss fragility.

`shop_pressure.py` combines forecast data, current sample build score, hand
realism, target safety, and capacity safety into the `_ShopPressure` object used
by reroll, pack, voucher, and buy/sell policy.

`shop_reroll.py` owns reroll and late-spend policy: minimum reroll bank, early
reroll allowance, escalating reroll costs, visible early power paths, late
reroll caps, pressure-driven extra rerolls, and bank-conversion closer mode.

`shop_safety.py` contains early-shop safety predicates and adjustments. It
penalizes unsupported early planets/tarots, economy-only or narrow jokers before
the run has real scoring, tracks money-scaling joker presence, and exposes
hand-support checks shared by joker and tarot valuation.

`shop_vouchers.py` owns voucher valuation. It blocks vouchers that are already
owned, hard-denied, or not useful for the current boss/shop pressure, then
applies pressure-aware adjustments for hand/discard count, hand size, shop
slots, discounts, rerolls, editions, tarot/planet generators, interest caps,
Retcon, Antimatter, and Observatory.

`shop_values.py` is the shop valuation glue used by the main shop action flow.
It scores legal buy/open/reroll actions, computes shop/card/pack thresholds,
applies spectral card values, scaling-commitment bonuses, visible safety-pack
reroll vetoes, and hidden-target pack guards.

`utils.py` contains tiny general-purpose helpers that do not belong to a domain
module yet, such as conservative integer parsing.

`winning_economy.py` handles known-deck discard lines where the current best
play already clears the blind but a discard can still improve round-end value.
It compares the baseline clear value against projected clear hands after
drawing gold cards, blue/gold seals, and discard-triggered cash effects.

## Dependency Direction

Keep extracted helper modules mostly one-way:

`basic_strategy_bot.py` may import from helper modules. Helper modules should
avoid importing `basic_strategy_bot.py`. When a helper would need live scoring
or pressure logic from the main bot, leave that helper in `basic_strategy_bot.py`
until the whole scoring area is split together.

This keeps the refactor mechanical and makes behavior-preserving checks easier:
move one cluster, import it back through `basic_strategy_bot.py`, run targeted
tests, then compare deterministic seed summaries.
