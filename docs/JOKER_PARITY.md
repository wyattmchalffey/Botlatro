# Joker parity map

Source checked: local `.balatro-headless/Balatro.exe` Lua, mainly `game.lua` for the 150 joker definitions and `card.lua` for `Card:calculate_joker`.

This file separates scoring parity from forward-sim parity. A joker can be "score covered" while still needing simulator work, because the evaluator may use the visible/current value from the real game but the simulator still has to update that value during search rollouts.

## Status legend

- `score covered`: `evaluate_played_cards` models the scoring effect when the required visible state is present.
- `sim covered`: `forward_sim` now updates the deterministic visible state for play/discard transitions.
- `needs sim`: deterministic, but the forward simulator still has to mutate state, money, cards, slots, packs, or shops.
- `stochastic hook`: search must inject/sample the random outcome instead of pretending it is deterministic.
- `not scoring`: no direct hand-score effect; relevant to shop/run planning, legality, or economy.

## Source coverage check

The local source has 150 real jokers. Our canonical name/rarity table also has 150 and currently has no missing or extra names.

## Score covered

These have direct scoring formulas implemented, or current-value scoring implemented when the visible/current value is present in joker metadata/effect text:

`Abstract Joker`, `Acrobat`, `Ancient Joker`, `Arrowhead`, `Banner`, `Baron`, `Baseball Card`, `Blackboard`, `Blue Joker`, `Blueprint`, `Bootstraps`, `Brainstorm`, `Bull`, `Caino`, `Campfire`, `Card Sharp`, `Castle`, `Cavendish`, `Ceremonial Dagger`, `Clever Joker`, `Constellation`, `Crafty Joker`, `Crazy Joker`, `Devious Joker`, `Droll Joker`, `Driver's License`, `Dusk`, `Erosion`, `Even Steven`, `Fibonacci`, `Flash Card`, `Flower Pot`, `Fortune Teller`, `Four Fingers`, `Glass Joker`, `Gluttonous Joker`, `Golden Ticket`, `Greedy Joker`, `Green Joker`, `Gros Michel`, `Hack`, `Half Joker`, `Hanging Chad`, `Hiker`, `Hit the Road`, `Hologram`, `Ice Cream`, `Joker`, `Joker Stencil`, `Jolly Joker`, `Loyalty Card`, `Lucky Cat`, `Lusty Joker`, `Mad Joker`, `Madness`, `Midas Mask`, `Mime`, `Mystic Summit`, `Obelisk`, `Odd Todd`, `Onyx Agate`, `Pareidolia`, `Photograph`, `Popcorn`, `Raised Fist`, `Ramen`, `Red Card`, `Ride the Bus`, `Rough Gem`, `Runner`, `Scary Face`, `Scholar`, `Seeing Double`, `Seltzer`, `Shoot the Moon`, `Shortcut`, `Sly Joker`, `Smeared Joker`, `Smiley Face`, `Sock and Buskin`, `Spare Trousers`, `Splash`, `Square Joker`, `Steel Joker`, `Stone Joker`, `Stuntman`, `Supernova`, `Swashbuckler`, `The Duo`, `The Family`, `The Idol`, `The Order`, `The Tribe`, `The Trio`, `Throwback`, `Triboulet`, `Vampire`, `Walkie Talkie`, `Wee Joker`, `Wily Joker`, `Wrathful Joker`, `Yorick`, `Zany Joker`.

Current-value caveat: `Campfire`, `Castle`, `Ceremonial Dagger`, `Constellation`, `Flash Card`, `Fortune Teller`, `Glass Joker`, `Hit the Road`, `Hologram`, `Lucky Cat`, `Madness`, `Obelisk`, `Red Card`, `Stone Joker`, `Throwback`, `Vampire`, and `Yorick` need their live current values supplied by state/replay/sim.

## Sim covered for transitions

These deterministic visible counters are now updated by `forward_sim`:

- On blind start: `Burglar`, `Chicot`, `Cartomancer` with injected tarot output, `Certificate` with injected sealed card output, `Marble Joker` with injected Stone-card output, and `Riff-raff` with injected common-joker output.
- On play: `Green Joker`, `Ride the Bus`, `Square Joker`, `Runner`, `Spare Trousers`, `Wee Joker`, `Ice Cream`, `Seltzer`, `Loyalty Card` when countdown metadata is present, `Vampire` XMult gain, `Lucky Cat` with injected Lucky-card trigger count, `DNA`, `Sixth Sense` with injected spectral output, `8 Ball`, `Seance`, `Superposition`, and `Vagabond` with injected tarot output, `Space Joker` with injected hand-level upgrade, `Mr. Bones` save/removal, plus `Hiker`, `Midas Mask`, and `Vampire` persistent played-card mutations recorded in the simulator `played_pile`.
- On discard: `Ramen`, `Green Joker`, `Hit the Road`, `Burnt Joker` hand-level upgrade on first discard, `Trading Card` first-discard money and destroyed card zone tracking, `Castle` when its current target suit is visible, `Yorick` when its countdown metadata is present, `Faceless Joker`, and `Mail-In Rebate` when its current target rank is visible.
- On cash-out/end of round: `Campfire` boss reset, `Rocket` boss upgrade, `Popcorn` decay/removal, `Turtle Bean` decay/removal, `Invisible Joker` round counter, `Egg` sell-value growth, `Hit the Road` reset, `To Do List` with injected next target, and injected stochastic extinction removal for `Gros Michel`/`Cavendish`.
- On cash-out money rows: blind reward, unused-hand money, optional unused-discard money, interest, `Golden Joker`, `Rocket`, `Delayed Gratification`, `Cloud 9` when its current nine count or deck contents are visible, `Satellite` when its unique planet count is visible, and `Gift Card` joker sell-value growth plus a consumable sell-value bonus tracker.
- In shop/pack transitions: generic buy/sell/reroll/open-pack/choose-pack/end-shop state updates, `Astronomer` free planet/celestial purchases, `Flash Card` reroll growth, `Campfire` selling growth, `Diet Cola` Double Tag creation, `Hallucination` with injected tarot output, `Luchador` boss-disable selling, `Perkeo` with injected negative-consumable copy at end shop, passive modifier add/remove for `Credit Card`, `Drunkard`, `Juggler`, `Merry Andy`, `Oops! All 6s`, `Showman`, `Stuntman`, `To the Moon`, `Troubadour`, and `Turtle Bean`, free reroll consumption when the state exposes it, and injected reroll/pack contents.

## Needs simulator work next

Deterministic play/discard card mutation or deck mutation:

None currently listed.

Deck/reshuffle integration now consumes simulator `played_pile` and `discard_pile` cards at cash-out, while inferred deck models remove cards in `played_pile`, `discard_pile`, and `destroyed_cards`.

Deterministic blind-select, hand-size, hand-count, or legality effects:

None currently listed.

Deterministic shop, pack, consumable, tag, cash-out, or economy effects:

None currently listed.

End-of-round deterministic decays/resets still need cash-out simulation:

None currently listed.

## Stochastic hooks

These should not be hard-coded as average values in the simulator. Search should inject/sample their outcome:

`8 Ball`, `Bloodstone`, `Business Card`, `Cavendish` extinction, `Gros Michel` extinction, `Lucky Cat`, `Misprint`, `Reserved Parking`, `Space Joker`.

The simulator/evaluator now expose deterministic injection hooks for all of the above. They still remain stochastic at policy time: replay diff or search must provide the observed/sampled trigger counts instead of averaging them inside the simulator.

`Hallucination` is stochastic, but the simulator now has an injected-output hook for the created tarot card.

`Oops! All 6s` is now recorded as a deterministic probability multiplier, but the stochastic hooks still need to consume that multiplier when their outcomes are injected or sampled.

## Full canonical joker list

`8 Ball`, `Abstract Joker`, `Acrobat`, `Ancient Joker`, `Arrowhead`, `Astronomer`, `Banner`, `Baron`, `Baseball Card`, `Blackboard`, `Bloodstone`, `Blue Joker`, `Blueprint`, `Bootstraps`, `Brainstorm`, `Bull`, `Burglar`, `Burnt Joker`, `Business Card`, `Caino`, `Campfire`, `Card Sharp`, `Cartomancer`, `Castle`, `Cavendish`, `Ceremonial Dagger`, `Certificate`, `Chaos the Clown`, `Chicot`, `Clever Joker`, `Cloud 9`, `Constellation`, `Crafty Joker`, `Crazy Joker`, `Credit Card`, `Delayed Gratification`, `Devious Joker`, `Diet Cola`, `DNA`, `Driver's License`, `Droll Joker`, `Drunkard`, `Dusk`, `Egg`, `Erosion`, `Even Steven`, `Faceless Joker`, `Fibonacci`, `Flash Card`, `Flower Pot`, `Fortune Teller`, `Four Fingers`, `Gift Card`, `Glass Joker`, `Gluttonous Joker`, `Golden Joker`, `Golden Ticket`, `Greedy Joker`, `Green Joker`, `Gros Michel`, `Hack`, `Half Joker`, `Hallucination`, `Hanging Chad`, `Hiker`, `Hit the Road`, `Hologram`, `Ice Cream`, `Invisible Joker`, `Joker`, `Joker Stencil`, `Jolly Joker`, `Juggler`, `Loyalty Card`, `Luchador`, `Lucky Cat`, `Lusty Joker`, `Mad Joker`, `Madness`, `Mail-In Rebate`, `Marble Joker`, `Matador`, `Merry Andy`, `Midas Mask`, `Mime`, `Misprint`, `Mr. Bones`, `Mystic Summit`, `Obelisk`, `Odd Todd`, `Onyx Agate`, `Oops! All 6s`, `Pareidolia`, `Perkeo`, `Photograph`, `Popcorn`, `Raised Fist`, `Ramen`, `Red Card`, `Reserved Parking`, `Ride the Bus`, `Riff-raff`, `Rocket`, `Rough Gem`, `Runner`, `Satellite`, `Scary Face`, `Scholar`, `Seance`, `Seeing Double`, `Seltzer`, `Shoot the Moon`, `Shortcut`, `Showman`, `Sixth Sense`, `Sly Joker`, `Smeared Joker`, `Smiley Face`, `Sock and Buskin`, `Space Joker`, `Spare Trousers`, `Splash`, `Square Joker`, `Steel Joker`, `Stone Joker`, `Stuntman`, `Supernova`, `Superposition`, `Swashbuckler`, `The Duo`, `The Family`, `The Idol`, `The Order`, `The Tribe`, `The Trio`, `Throwback`, `To Do List`, `To the Moon`, `Trading Card`, `Triboulet`, `Troubadour`, `Turtle Bean`, `Vagabond`, `Vampire`, `Walkie Talkie`, `Wee Joker`, `Wily Joker`, `Wrathful Joker`, `Yorick`, `Zany Joker`.
