"""Shared helpers for :mod:`balatro_ai.bots.basic_strategy_bot`.

The public bot class still lives in ``basic_strategy_bot.py``. This package
contains the smaller helper modules extracted from that file. The old private
names are imported back through ``basic_strategy_bot.py`` so existing tests,
debug probes, and trace tooling can continue to use ``strategy._foo`` while the
implementation becomes easier to navigate.

Module map:

* ``actions``: generic ``Action`` construction, annotation, and memory keys.
* ``ante_one_hunt``: ante-1 small-blind discard hunt and near-clear upgrades.
* ``banner_policy``: Banner discard-veto EV policy.
* ``blind_setup``: tempo-spending blind setup plays/discards for joker value.
* ``blind_reasons``: trace/reason string formatters for blind decisions.
* ``blind_solver``: shared blind-state solution used by tactics and shop pressure.
* ``blind_state``: small shared blind-state predicates.
* ``blind_tactics``: top-level blind play/discard action router.
* ``build_profile``: joker role scoring, durability, and build-profile policy.
* ``build_scoring``: representative sample-hand build score projections.
* ``cache``: per-decision caches used during one ``choose_action`` call.
* ``cards``: card payload adapters, card categories, slot limits, rank/suit
  parsing, editions, and shop-card-to-joker conversion.
* ``data``: static strategy tables and hand/joker/voucher valuation constants.
* ``decision_context``: per-action derived context and lazy evaluator seam.
* ``discard_policy``: generic discard selection and hand-pace policy.
* ``discard_state``: state projection, draw counts, and discard-trigger effects.
* ``draw_evaluation``: straight/flush/rank/full-house draw evaluation helpers.
* ``draw_math``: pure draw probabilities and straight/rank utility math.
* ``economy_hunt``: blind-side economy value helpers for held/drawn cards.
* ``hand_models``: small dataclasses shared by play/discard planning.
* ``hand_preferences``: preferred hand, archetype, and dedicated-plan helpers.
* ``hand_value``: card keep scores, kept-hand potential, and long-term card
  value helpers.
* ``held_consumables``: held consumable use policy in shop flow.
* ``jokers``: joker metadata, effect text, disabled/sticker checks, role
  classification, and simple joker-derived state helpers.
* ``joker_ordering``: joker permutation search and rearrange action selection.
* ``pack_choice``: booster-opened pick/skip action selection.
* ``pack_targets``: pack-card pickability and tarot target-index selection.
* ``play_scoring``: play action scoring, hand sequencing, and play candidates.
* ``preferred_hunt``: preferred-hand hunt discard/redraw policy.
* ``profile``: shop pressure, build profile, and audit payload containers.
* ``rare_hands``: rare-hand plans, support gaps, and rank-target helpers.
* ``run_plan``: run-level archetype, commitment, shop posture, reroll budget,
  and pack/voucher permission signals derived from the current state; hard
  voucher blocks remain authoritative.
* ``score_projection``: projected discard scores and optimistic completion
  scoring helpers.
* ``shop_cards``: non-joker shop card valuation for planets, tarots, and cards.
* ``shop_flow``: top-level shop action selection and audit payloads.
* ``shop_forecast``: upcoming-blind and boss-pressure forecast helpers.
* ``shop_items``: lookup and payload helpers for visible shop/pack items.
* ``shop_jokers``: joker buy, replacement, owned-value, and role valuation.
* ``shop_money``: shop reserves, interest caps, and spend penalties.
* ``shop_packs``: booster pack valuation, Standard-pack payoff checks, and
  late-pack opening policy.
* ``shop_pressure``: shop pressure calculation from forecast and build score.
* ``shop_reroll``: reroll limits, late spend mode, and bank conversion policy.
* ``shop_safety``: early-shop safety adjustments and hand-support checks.
* ``shop_vouchers``: voucher gating and pressure-aware voucher valuation.
* ``shop_values``: buy/open/reroll action valuation and shop thresholds.
* ``utils``: tiny general-purpose helpers shared across modules.
* ``winning_economy``: winning known-deck discard lines for extra economy.
"""
